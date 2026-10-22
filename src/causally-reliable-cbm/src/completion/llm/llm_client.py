from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from mistralai import Mistral
from openai import OpenAI


class llm_client:
    """Provider-independent client used by the graph completion code.

    Qwen and DeepSeek expose OpenAI-compatible chat-completions endpoints. The
    optional JSONL log doubles as a response cache, so a restarted experiment
    does not pay for already completed votes.
    """

    def __init__(
        self,
        LLM="gpt-4o",
        temperature=0,
        max_tries=1,
        provider=None,
        base_url=None,
        api_key_env=None,
        max_tokens=64,
        max_retries=2,
        timeout=120,
        thinking=False,
        log_file=None,
    ):
        self.LLM = LLM
        self.temperature = temperature
        self.max_tries = max_tries
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking = thinking
        self.provider = provider or self._infer_provider(LLM)
        self.log_file = log_file or os.environ.get("C2BM_LLM_LOG_FILE")
        self._response_cache = {}
        self._load_response_cache()

        if self.provider in {"openai", "qwen", "deepseek", "openai_compatible"}:
            default_key_env = {
                "openai": "OPENAI_API_KEY",
                "qwen": "DASHSCOPE_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
            }[self.provider]
            key_env = api_key_env or default_key_env
            api_key = os.environ.get(key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key: set {key_env} in /home/src/Reliable/.env.llm"
                )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
        elif self.provider == "mistral":
            api_key = os.environ.get("MISTRALAIKEY")
            if not api_key:
                raise RuntimeError("Missing API key: set MISTRALAIKEY")
            self.client = Mistral(api_key=api_key)
        else:
            raise ValueError(f"provider ({self.provider}) is not implemented")

    @staticmethod
    def _infer_provider(model):
        model_lower = model.lower()
        if "qwen" in model_lower:
            return "qwen"
        if "deepseek" in model_lower:
            return "deepseek"
        if "gpt" in model_lower:
            return "openai"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        return "openai_compatible"

    def _extra_body(self):
        if self.provider == "qwen":
            return {"enable_thinking": bool(self.thinking)}
        if self.provider == "deepseek":
            return {"thinking": {"type": "enabled" if self.thinking else "disabled"}}
        return None

    @staticmethod
    def _requires_answer_tag(question):
        return bool(re.search(r"<answer>", question, flags=re.IGNORECASE))

    def _system_message(self, question):
        if self._requires_answer_tag(question):
            return (
                "You analyze causal relationships. You may reason step-by-step, "
                "but your final line must be exactly one tag: <answer>A</answer>, "
                "<answer>B</answer>, or <answer>C</answer>."
            )
        return "You are a helpful AI assistant."

    def _cache_key(self, question, temperature):
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.LLM,
                "temperature": temperature,
                "question": question,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_response_cache(self):
        if not self.log_file:
            return
        path = Path(self.log_file)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                key = (event.get("cache_key"), str(event.get("cache_slot", "default")))
                if event.get("event") in {"response", "vote_cache"} and "raw_response" in event:
                    self._response_cache[key] = event["raw_response"]
                elif event.get("event") == "cache_invalidate":
                    self._response_cache.pop(key, None)

    def _append_event(self, event):
        if not self.log_file:
            return
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "requested_model": self.LLM,
            **event,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _invoke_once(self, question, temperature, cache_slot="default"):
        cache_key = self._cache_key(question, temperature)
        response_cache_key = (cache_key, str(cache_slot))
        if response_cache_key in self._response_cache:
            self._append_event(
                {
                    "event": "cache_hit",
                    "cache_key": cache_key,
                    "cache_slot": cache_slot,
                }
            )
            return self._response_cache[response_cache_key]

        started = time.monotonic()
        try:
            if self.provider in {"openai", "qwen", "deepseek", "openai_compatible"}:
                request_kwargs = {
                    "model": self.LLM,
                    "messages": [
                        {"role": "system", "content": self._system_message(question)},
                        {"role": "user", "content": question},
                    ],
                    "temperature": temperature,
                    "max_tokens": self.max_tokens,
                }
                extra_body = self._extra_body()
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                response = self.client.chat.completions.create(**request_kwargs)
                answer = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage is not None else None
                actual_model = response.model
            else:
                response = self.client.chat.complete(
                    model=self.LLM,
                    messages=[
                        {"role": "system", "content": self._system_message(question)},
                        {"role": "user", "content": question},
                    ],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage is not None else None
                actual_model = self.LLM
        except Exception as exc:
            self._append_event(
                {
                    "event": "error",
                    "cache_key": cache_key,
                    "cache_slot": cache_slot,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        event = {
            "event": "response",
            "cache_key": cache_key,
            "cache_slot": cache_slot,
            "actual_model": actual_model,
            "temperature": temperature,
            "thinking": self.thinking,
            "latency_seconds": round(time.monotonic() - started, 3),
            "usage": usage,
            "prompt": question,
            "raw_response": answer,
        }
        self._append_event(event)
        self._response_cache[response_cache_key] = answer
        return answer

    def invoke(self, question, temperature=0):
        return self._invoke_once(question, temperature, cache_slot="default")

    def _invalidate_cached_response(self, question, temperature, cache_slot):
        cache_key = self._cache_key(question, temperature)
        response_cache_key = (cache_key, str(cache_slot))
        self._response_cache.pop(response_cache_key, None)
        self._append_event(
            {
                "event": "cache_invalidate",
                "cache_key": cache_key,
                "cache_slot": cache_slot,
            }
        )

    def _cache_completed_vote(self, question, temperature, vote_index, raw_answer):
        cache_key = self._cache_key(question, temperature)
        cache_slot = f"vote-{vote_index}"
        self._response_cache[(cache_key, cache_slot)] = raw_answer
        self._append_event(
            {
                "event": "vote_cache",
                "cache_key": cache_key,
                "cache_slot": cache_slot,
                "raw_response": raw_answer,
            }
        )

    @staticmethod
    def parse_answer(answer):
        if not answer:
            return None
        patterns = (
            r"<answer>\s*([ABC])\s*</answer>",
            r"<answer>\s*([ABC])\b",
            r"(?:final\s+answer|correct\s+option|therefore)\s*[:：]?\s*(?:option\s*)?([ABC])\b",
            r"\boption\s*([ABC])\b",
        )
        for pattern in patterns:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        tail = answer.strip()[-120:]
        match = re.search(r"\b([ABC])\b\s*$", tail, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None

    def invoke_with_retry(self, question):
        """Collect ``max_tries`` valid votes and return their majority answer."""
        votes = []
        for vote_index in range(self.max_tries):
            cache_key = self._cache_key(question, self.temperature)
            completed_vote_key = (cache_key, f"vote-{vote_index}")
            completed_vote = self._response_cache.get(completed_vote_key)
            parsed_completed_vote = self.parse_answer(completed_vote or "")
            if parsed_completed_vote in {"A", "B", "C"}:
                votes.append(parsed_completed_vote)
                continue

            parsed = None
            raw_answer = ""
            for retry_index in range(self.max_retries + 1):
                cache_slot = f"vote-{vote_index}-attempt-{retry_index}"
                vote_prompt = question
                if retry_index > 0:
                    vote_prompt = (
                        f"{question}\n\n"
                        "Your previous reply did not include a valid answer tag. "
                        "Keep the reasoning short and end with exactly one of "
                        "<answer>A</answer>, <answer>B</answer>, or <answer>C</answer>."
                    )
                raw_answer = self._invoke_once(
                    vote_prompt,
                    self.temperature,
                    cache_slot=cache_slot,
                )
                parsed = self.parse_answer(raw_answer)
                if parsed in {"A", "B", "C"}:
                    break
                self._append_event(
                    {
                        "event": "invalid_answer",
                        "vote_index": vote_index,
                        "retry_index": retry_index,
                        "raw_response": raw_answer,
                    }
                )
                self._invalidate_cached_response(
                    question,
                    self.temperature,
                    cache_slot,
                )
            if parsed not in {"A", "B", "C"}:
                raise ValueError(
                    f"Model {self.LLM} did not return a valid <answer>A|B|C</answer> "
                    f"after {self.max_retries + 1} attempts for vote {vote_index}."
                )
            self._cache_completed_vote(
                question,
                self.temperature,
                vote_index,
                raw_answer,
            )
            votes.append(parsed)

        counts = Counter(votes)
        winner = counts.most_common(1)[0][0]
        self._append_event(
            {
                "event": "vote_summary",
                "votes": votes,
                "counts": dict(counts),
                "winner": winner,
            }
        )
        return winner

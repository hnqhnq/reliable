#!/usr/bin/env python3
"""Make one cached, strictly parsed request to an OpenAI-compatible provider."""

import argparse
import os
import sys


ROOT_DIR = "/home/src/Reliable"
C2BM_DIR = os.path.join(ROOT_DIR, "src", "causally-reliable-cbm")
sys.path.insert(0, C2BM_DIR)

from src.completion.llm.llm_client import llm_client  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("qwen", "deepseek"))
    parser.add_argument("log_file")
    args = parser.parse_args()

    if args.provider == "qwen":
        model = os.environ.get("QWEN_MODEL", "qwen3.7-flash-2026-07-15")
        base_url = os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        key_env = "DASHSCOPE_API_KEY"
    else:
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        key_env = "DEEPSEEK_API_KEY"

    client = llm_client(
        LLM=model,
        provider=args.provider,
        base_url=base_url,
        api_key_env=key_env,
        temperature=0,
        max_tries=1,
        max_retries=2,
        max_tokens=64,
        thinking=False,
        log_file=args.log_file,
    )
    answer = client.invoke_with_retry(
        "This is an API connection and output-format test. Return exactly "
        "<answer>A</answer> and no other text."
    )
    if answer != "A":
        raise RuntimeError(f"Unexpected parsed answer: {answer}")
    print(f"{args.provider}: connection and response format OK (model={model})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Low-cost BC-LLM compatibility demo: one image and one text prompt."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys


ROOT_DIR = Path("/home/src/Reliable")
PROJECT_DIR = ROOT_DIR / "src" / "bc-llm"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "llm-api-main"))

from lab_llm.constants import convert_to_llm_type  # noqa: E402
from lab_llm.dataset import ImageDataset  # noqa: E402
from lab_llm.duckdb_handler import DuckDBHandler  # noqa: E402
from lab_llm.error_callback_handler import ErrorCallbackHandler  # noqa: E402
from lab_llm.llm_api import LLMApi  # noqa: E402
from lab_llm.llm_cache import LLMCache  # noqa: E402
from src.llm_response_types import CandidateConcepts, ProtoConceptExtract  # noqa: E402


def atomic_json_dump(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=run_dir / "bc_api_demo.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("bc_llm_api_demo")

    extraction_model = os.environ.get(
        "QWEN_EXTRACTION_MODEL", "qwen3.7-flash-2026-07-15"
    )
    iter_model = os.environ.get("QWEN_ITER_MODEL", "qwen3.8-flash")
    cache = LLMCache(DuckDBHandler(str(run_dir / "api_cache.duckdb")))
    error_handler = ErrorCallbackHandler(logger)
    extraction_llm = LLMApi(
        cache,
        seed=10,
        model_type=convert_to_llm_type(extraction_model),
        error_handler=error_handler,
        logging=logger,
        timeout=120,
    )
    iter_llm = LLMApi(
        cache,
        seed=10,
        model_type=convert_to_llm_type(iter_model),
        error_handler=error_handler,
        logging=logger,
        timeout=120,
    )

    image_root = (
        PROJECT_DIR
        / "exp_cub_birds_existing"
        / "data"
        / "CUB_200_2011"
        / "CUB_200_2011"
        / "images"
    )
    image_paths = sorted(image_root.glob("*/*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No CUB image found below {image_root}")
    image_path = image_paths[0]

    extraction_prompt = (
        "Describe visible bird attributes using at least three short keyphrases. "
        "Return valid JSON with fields reasoning and keyphrases."
    )
    image_results = asyncio.run(
        extraction_llm.get_outputs(
            ImageDataset([str(image_path)], extraction_prompt),
            batch_size=1,
            max_new_tokens=300,
            is_image=True,
            response_model=ProtoConceptExtract,
        )
    )
    image_result = image_results[0]
    if image_result is None or not image_result.keyphrases:
        raise RuntimeError("Qwen image structured-output demo returned no keyphrases")

    concept_prompt = """Propose exactly two binary, visually answerable bird concepts.
Return valid JSON in this form:
{
  "reasoning": "short reason",
  "concepts": [
    {"concept": "Does the bird ...?", "words": ["short phrase"]},
    {"concept": "Does the bird ...?", "words": ["short phrase"]}
  ]
}
"""
    concept_result = iter_llm.get_output(
        concept_prompt,
        max_new_tokens=500,
        response_model=CandidateConcepts,
    )
    if concept_result is None or len(concept_result.concepts) < 2:
        raise RuntimeError("Qwen concept-proposal structured-output demo failed")

    summary = {
        "status": "SUCCEEDED",
        "image": str(image_path),
        "extraction_model": extraction_model,
        "num_keyphrases": len(image_result.keyphrases),
        "iter_model": iter_model,
        "num_candidate_concepts": len(concept_result.concepts),
        "checks": {
            "qwen_image_input": True,
            "pydantic_structured_extraction": True,
            "pydantic_structured_concept_proposal": True,
            "duckdb_response_cache": True,
        },
    }
    atomic_json_dump(summary, run_dir / "bc_api_demo_summary.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

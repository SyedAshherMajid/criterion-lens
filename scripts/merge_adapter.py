"""Merge a trained PEFT adapter into its base model and optionally publish it."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)
_UPLOAD_IGNORES = [
    "training_args.bin",
    "log_history.json",
    "train_config.json",
    "trainable_params.json",
    "truncation_log.json",
    "train_results.json",
    "all_results.json",
]


def _read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("model_name_or_path"):
        raise ValueError(f"Training config has no base model name: {config_path}")
    return config


def _write_model_card(output_dir: Path, config: dict[str, Any]) -> None:
    model_id = config["model_name_or_path"]
    card = f"""---
license: apache-2.0
base_model: {model_id}
tags:
  - criterion-screening
  - evidence-grounded
---

# CriterionLens merged model

This repository contains the merged weights for a research prototype that
matches patient-note evidence to clinical-trial eligibility criteria. It is
not a medical device and must not be used to make clinical decisions.

The LoRA adapter was trained on the training partition only. Validation
metrics will be added after the fixed validation evaluation is complete.
No test-set metrics are included.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def merge_run(run_dir: Path, output_dir: Path) -> Path:
    """Merge a local adapter directory into a standalone model directory."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged model: {output_dir}")
    config = _read_config(run_dir)
    if not (run_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter config is missing from {run_dir}")
    if not (run_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights are missing from {run_dir}")

    dtype_name = config.get("torch_dtype_used", config.get("torch_dtype", "float16"))
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported merge dtype: {dtype_name}")

    base = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        dtype=dtypes[dtype_name],
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    adapted = PeftModel.from_pretrained(base, str(run_dir), is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir), use_fast=True)
    tokenizer.save_pretrained(output_dir)
    _write_model_card(output_dir, config)
    LOGGER.info("Merged %s into %s", run_dir, output_dir)
    return output_dir


def publish_merged(output_dir: Path, repo_id: str) -> str:
    """Publish a merged model after verifying the authenticated Hub owner."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; load a write-enabled token before uploading")
    if "/" not in repo_id:
        raise ValueError("repo_id must use the owner/name form")

    owner = repo_id.split("/", maxsplit=1)[0]
    api = HfApi(token=token)
    identity = api.whoami()
    if str(identity["name"]).casefold() != owner.casefold():
        raise ValueError(
            f"Authenticated Hugging Face user {identity['name']} does not own {repo_id}"
        )
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_dir,
        ignore_patterns=_UPLOAD_IGNORES,
        commit_message="Publish merged CriterionLens model",
    )
    LOGGER.info("Uploaded merged model to %s at commit %s", repo_id, commit.oid)
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = merge_run(args.run, args.output_dir)
    if args.repo:
        publish_merged(output_dir, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

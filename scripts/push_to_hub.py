"""Publish a trained LoRA adapter and its model card to the Hugging Face Hub."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IGNORED_UPLOAD_PATTERNS = [
    "checkpoint-*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
    "trainer_state.json",
    "train_config.json",
    "trainable_params.json",
    "log_history.json",
    "truncation_log.json",
    "train_results.json",
]


def build_model_card(
    config: dict[str, Any],
    trainable_params: dict[str, Any],
    split_manifest: dict[str, Any],
) -> str:
    """Render a model card with verified provenance and explicit metric placeholders."""
    split_rows = "\n".join(
        f"| {name} | {split_manifest[name]['n_patients']} | "
        f"{split_manifest[name]['n_annotations']} |"
        for name in ("train", "val", "test")
    )
    return f"""---
license: apache-2.0
base_model: {config["model_name_or_path"]}
library_name: peft
tags:
  - criterion-screening
  - evidence-grounded
  - lora
---

# CriterionLens LoRA adapter

This adapter is a research prototype for matching patient-note evidence to
clinical-trial eligibility criteria. It is not a medical device and must not
be used to make clinical decisions.

## Base model and training

- Base model: `{config["model_name_or_path"]}`
- Dataset: `ncbi/TrialGPT-Criterion-Annotations`
- Dataset revision: `{config["dataset_revision"]}`
- Maximum sequence length: {config["max_seq_len"]}
- Epochs: {config["num_train_epochs"]}
- LoRA rank / alpha: {config["lora"]["r"]} / {config["lora"]["lora_alpha"]}
- Trainable parameters: {trainable_params["trainable_parameters"]:,} of
  {trainable_params["total_parameters"]:,}
- Targets: evidence IDs and expert eligibility labels only; GPT-4 explanations
  are excluded from inputs and training targets.

## Fixed patient-level split

| Split | Patients | Annotations |
|---|---:|---:|
{split_rows}

## Metrics

Validation macro-F1: **TBD**

Evidence micro-F1: **TBD**

Metrics are intentionally placeholders until validation evaluation is complete.
No test-set metrics are included in this model card.
"""


def write_model_card(run_dir: Path, config: dict[str, Any]) -> Path:
    params = json.loads((run_dir / "trainable_params.json").read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (PROJECT_ROOT / str(config["split_manifest"])).read_text(encoding="utf-8")
    )
    card_path = run_dir / "README.md"
    card_path.write_text(
        build_model_card(config, params, split_manifest),
        encoding="utf-8",
    )
    return card_path


def push_run(run_dir: Path, repo_id: str) -> str:
    """Create or update a model repo and upload the final adapter files."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; load a write-enabled token before uploading")

    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    if "/" not in repo_id:
        raise ValueError("repo_id must use the owner/name form")
    owner = repo_id.split("/", maxsplit=1)[0]
    api = HfApi(token=token)
    identity = api.whoami()
    if str(identity["name"]).casefold() != owner.casefold():
        raise ValueError(
            f"Authenticated Hugging Face user {identity['name']} does not own {repo_id}"
        )

    write_model_card(run_dir, config)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=run_dir,
        ignore_patterns=_IGNORED_UPLOAD_PATTERNS,
        commit_message="Upload trained CriterionLens LoRA adapter",
    )
    LOGGER.info("Uploaded adapter to %s at commit %s", repo_id, commit.oid)
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    push_run(args.run, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

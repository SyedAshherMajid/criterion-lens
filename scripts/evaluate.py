"""Evaluate one model on a fixed validation or sealed test split."""

import argparse
import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from criterion_lens.data.load import PROJECT_ROOT, load_annotations
from criterion_lens.data.sentences import split_sentences
from criterion_lens.eval.evidence_metrics import evidence_metrics
from criterion_lens.eval.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    save_confusion_matrices,
)
from criterion_lens.inference.predict import Decoder, predict_criterion
from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
TEST_ACCESS_LOG = PROJECT_ROOT / "results" / "TEST_ACCESS_LOG.md"


def _git_output(arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Git command failed: git {' '.join(arguments)}")
    return result.stdout.strip()


def record_test_access(
    *,
    split: str,
    final: bool,
    model: str,
    adapter: str | None,
    decoding_config: dict[str, Any],
) -> None:
    """Enforce the test-set seal and append an immutable access record."""
    if split != "test":
        if final:
            raise ValueError("--final is valid only with --split test")
        return
    if not final:
        raise ValueError("Test evaluation requires --final")
    if TEST_ACCESS_LOG.exists():
        existing_entries = TEST_ACCESS_LOG.read_text(encoding="utf-8").splitlines()
        if any(line.startswith("|") and "| test |" in line for line in existing_entries):
            raise RuntimeError("The final test split has already been accessed")

    tag = _git_output(["describe", "--exact-match", "--tags", "HEAD"])
    if tag != "v1.0-frozen":
        raise RuntimeError("Final test evaluation requires HEAD tagged v1.0-frozen")
    commit = _git_output(["rev-parse", "HEAD"])
    TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not TEST_ACCESS_LOG.exists():
        TEST_ACCESS_LOG.write_text(
            "# Final test-set access log\n\n"
            "| Timestamp (UTC) | Split | Git commit | Tag | Model | Adapter | Decoding config |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = json.dumps(decoding_config, sort_keys=True, separators=(",", ":"))
    cells = (
        timestamp,
        "test",
        commit,
        tag,
        model,
        adapter or "none",
        config,
    )
    safe_cells = [cell.replace("|", r"\|") for cell in cells]
    with TEST_ACCESS_LOG.open("a", encoding="utf-8") as access_log:
        access_log.write("| " + " | ".join(safe_cells) + " |\n")


def load_model(model_id: str, adapter_id: str | None) -> tuple[Any, Any, torch.device]:
    """Load a base model and optional PEFT adapter on the available device."""
    load_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer {model_id} has no pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _decoder_for_mode(decoding: str) -> Decoder | None:
    if decoding == "no_cache":
        return None
    if decoding == "greedy_kv":
        from criterion_lens.decoding.kv_cache import greedy_kv_cache

        return lambda model, input_ids, max_new_tokens, eos_id: greedy_kv_cache(
            model,
            input_ids,
            max_new=max_new_tokens,
            eos_id=eos_id,
        )
    raise ValueError(f"Unsupported decoding mode: {decoding}")


def evaluate_records(
    records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    split: str,
    model_id: str,
    adapter_id: str | None,
    decoding: str,
    max_new_tokens: int,
    output_dir: Path,
    decoder: Decoder | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run predictions and write per-record, aggregate, and confusion outputs."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if not records:
        raise ValueError(f"The {split} split contains no annotations")

    prediction_rows: list[dict[str, Any]] = []
    for row in records:
        criterion_type = str(row["criterion_type"])
        annotation_id = row["annotation_id"]
        result = predict_criterion(
            model,
            tokenizer,
            str(row["note"]),
            f"{criterion_type[:3]}-{annotation_id}",
            criterion_type,
            str(row["criterion_text"]),
            decoding=decoding,
            max_new_tokens=max_new_tokens,
            decoder=decoder,
        )
        prediction_row = asdict(result)
        prediction_row["annotation_id"] = int(annotation_id)
        prediction_row["patient_id"] = str(row["patient_id"])
        prediction_row["trial_id"] = str(row["trial_id"])
        prediction_row["gold"] = {
            "label": str(row["expert_eligibility"]),
            "evidence_ids": sorted(set(row["expert_sentences"])),
        }
        prediction_rows.append(prediction_row)

    gold_labels = [row["gold"]["label"] for row in prediction_rows]
    predicted_labels = [row["label"] for row in prediction_rows]
    criterion_types = [row["criterion_type"] for row in prediction_rows]
    gold_evidence = [row["gold"]["evidence_ids"] for row in prediction_rows]
    predicted_evidence = [row["evidence_ids"] for row in prediction_rows]
    sentence_counts = [
        len(split_sentences(str(row["note"]))) for row in records
    ]
    classification = classification_metrics(
        gold_labels,
        predicted_labels,
        criterion_types,
        parse_failures=[not row["parse_ok"] for row in prediction_rows],
        fallback_used=[row["fallback_used"] for row in prediction_rows],
        invalid_labels=[row["invalid_label"] for row in prediction_rows],
    )
    evidence = evidence_metrics(
        gold_evidence,
        predicted_evidence,
        sentence_counts,
        criterion_types,
        gold_labels,
    )
    decoding_config = {
        "mode": decoding,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    metrics: dict[str, Any] = {
        "split": split,
        "model": model_id,
        "adapter": adapter_id,
        "device": str(device or getattr(model, "device", "unknown")),
        "n_annotations": len(prediction_rows),
        "decoding": decoding_config,
        "classification": classification,
        "evidence": evidence,
    }
    if split == "test":
        metrics["bootstrap_95_ci"] = bootstrap_confidence_intervals(
            gold_labels,
            predicted_labels,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as predictions_file:
        for prediction_row in prediction_rows:
            predictions_file.write(
                json.dumps(prediction_row, ensure_ascii=False, allow_nan=False) + "\n"
            )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_confusion_matrices(classification, output_dir)
    LOGGER.info("Wrote %d %s predictions to %s", len(prediction_rows), split, output_dir)
    return metrics


def run_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if arguments.final and arguments.split != "test":
        raise ValueError("--final is valid only with --split test")

    decoder = _decoder_for_mode(arguments.decoding)
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    model, tokenizer, device = load_model(arguments.model, arguments.adapter)
    record_test_access(
        split=arguments.split,
        final=arguments.final,
        model=arguments.model,
        adapter=arguments.adapter,
        decoding_config={
            "mode": arguments.decoding,
            "max_new_tokens": arguments.max_new_tokens,
            "do_sample": False,
        },
    )
    frame = load_annotations(revision=manifest["dataset_revision"])
    patient_ids = set(manifest[arguments.split]["patient_ids"])
    selected = frame.loc[frame["patient_id"].isin(patient_ids)]
    expected_count = int(manifest[arguments.split]["n_annotations"])
    if len(selected) != expected_count:
        raise ValueError(
            f"{arguments.split} split has {len(selected)} rows; "
            f"manifest expects {expected_count}"
        )
    return evaluate_records(
        selected.to_dict("records"),
        model,
        tokenizer,
        split=arguments.split,
        model_id=arguments.model,
        adapter_id=arguments.adapter,
        decoding=arguments.decoding,
        max_new_tokens=arguments.max_new_tokens,
        output_dir=arguments.out,
        decoder=decoder,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument(
        "--decoding",
        choices=("no_cache", "greedy_kv"),
        default="no_cache",
    )
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_evaluation(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

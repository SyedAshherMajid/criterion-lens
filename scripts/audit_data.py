"""Audit annotation labels, evidence IDs, and text lengths without model evaluation."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse input manifest, tokenizer, and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/split_manifest.json"),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def _plot_label_distribution(audit: dict[str, Any], output_path: Path) -> None:
    """Plot observed expert-label counts by criterion type."""
    by_type = audit["label_counts"]["by_criterion_type"]
    labels = sorted({label for counts in by_type.values() for label in counts})
    criterion_types = list(by_type)
    positions = np.arange(len(labels))
    width = 0.8 / max(len(criterion_types), 1)

    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    for index, criterion_type in enumerate(criterion_types):
        counts = by_type[criterion_type]
        values = [counts.get(label, 0) for label in labels]
        axis.bar(positions + index * width, values, width, label=criterion_type)
    axis.set_xticks(positions + width * max(len(criterion_types) - 1, 0) / 2)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Annotations")
    axis.set_title("Expert label distribution by criterion type")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_note_token_lengths(audit: dict[str, Any], output_path: Path) -> None:
    """Plot the generated mean and 95th-percentile note-token lengths by split."""
    summaries = audit["lengths"]["notes_by_split"]
    split_names = [name for name in ("train", "val", "test") if name in summaries]
    means = [summaries[name]["tokens"]["mean"] for name in split_names]
    percentiles = [summaries[name]["tokens"]["p95"] for name in split_names]
    positions = np.arange(len(split_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(positions - width / 2, means, width, label="Mean")
    axis.bar(positions + width / 2, percentiles, width, label="95th percentile")
    axis.set_xticks(positions)
    axis.set_xticklabels(split_names)
    axis.set_ylabel("Qwen2.5 tokenizer tokens per patient note")
    axis.set_title("Patient-note token lengths by split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Load the fixed split and write descriptive JSON and figure artifacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    from criterion_lens.data.load import load_annotations

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    frame = load_annotations(revision=split_manifest["dataset_revision"])
    if frame.attrs["dataset_revision"] != split_manifest["dataset_revision"]:
        raise ValueError("Loaded dataset revision differs from the split manifest")

    from transformers import AutoTokenizer

    from criterion_lens.data.audit import audit_annotations

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    audit = audit_annotations(frame, split_manifest, tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _plot_label_distribution(audit, args.figures_dir / "label_dist.png")
    _plot_note_token_lengths(audit, args.figures_dir / "note_length_tokens.png")

    LOGGER.info(
        "Audited %d annotations and %d patients at revision %s",
        audit["n_annotations"],
        audit["n_patients"],
        audit["dataset_revision"],
    )
    LOGGER.info(
        "Label/type inconsistencies=%d; out-of-range evidence IDs=%d; note parse failures=%d",
        audit["label_type_inconsistencies"]["count"],
        audit["evidence"]["out_of_range_sentence_id_count"],
        audit["evidence"]["note_parse_failures"],
    )
    LOGGER.info("Wrote audit JSON to %s and figures to %s", args.output, args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

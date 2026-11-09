"""Load and sanitize TrialGPT criterion annotations."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
GPT4_COLUMNS = {
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
    "training",
}
REQUIRED_COLUMNS = [
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "annotations.parquet"
LOGGER = logging.getLogger(__name__)


def _parse_expert_sentences(value: Any, annotation_id: Any) -> list[int]:
    """Parse evidence IDs while rejecting malformed or non-integer targets."""
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or any(
        not isinstance(sentence_id, int) or isinstance(sentence_id, bool)
        for sentence_id in parsed
    ):
        raise ValueError(f"Invalid expert_sentences for annotation {annotation_id}")
    return parsed


def load_annotations(revision: str | None = None) -> pd.DataFrame:
    """Load a pinned annotation snapshot and remove leakage-prone fields immediately.

    The unused GPT-4 outputs and source `training` flag are removed from the
    Hugging Face Dataset before conversion, so downstream code only sees expert
    annotations and the fixed patient-level split is the sole training selector.
    """
    dataset_info = HfApi().dataset_info(DATASET_ID, revision=revision)
    dataset_revision = dataset_info.sha
    if not dataset_revision:
        raise RuntimeError(f"Hugging Face returned no revision for {DATASET_ID}")

    dataset = load_dataset(DATASET_ID, split="train", revision=dataset_revision)
    columns_to_drop = sorted(set(dataset.column_names) & GPT4_COLUMNS)
    if columns_to_drop:
        dataset = dataset.remove_columns(columns_to_drop)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataset.column_names))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    frame = dataset.to_pandas()[REQUIRED_COLUMNS].copy()
    empty_criteria = frame["criterion_text"].isna() | (
        frame["criterion_text"].astype("string").str.strip() == ""
    )
    dropped_count = int(empty_criteria.sum())
    if dropped_count:
        LOGGER.warning("Dropping %d annotations with null or empty criterion_text", dropped_count)
        frame = frame.loc[~empty_criteria].copy()

    frame["criterion_type"] = frame["criterion_type"].astype("string").str.strip().str.lower()
    frame["expert_eligibility"] = (
        frame["expert_eligibility"].astype("string").str.strip().str.lower()
    )
    frame["expert_sentences"] = [
        _parse_expert_sentences(value, annotation_id)
        for value, annotation_id in zip(
            frame["expert_sentences"], frame["annotation_id"], strict=True
        )
    ]
    frame.attrs["dataset_revision"] = dataset_revision
    frame.attrs["dropped_empty_criterion_count"] = dropped_count

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE_PATH, index=False)
    LOGGER.info(
        "Loaded %d annotations from dataset revision %s; dropped %d empty criteria",
        len(frame),
        dataset_revision,
        dropped_count,
    )
    return frame

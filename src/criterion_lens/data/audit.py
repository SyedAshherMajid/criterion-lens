"""Compute descriptive dataset statistics without evaluating model predictions."""

from typing import Any

import numpy as np
import pandas as pd

from criterion_lens.data.prompts import ALLOWED_LABELS, build_messages, format_target
from criterion_lens.data.sentences import parse_numbered_note


def _counts(values: pd.Series) -> dict[str, int]:
    normalized = values.astype("string").fillna("<missing>")
    return {
        str(label): int(count)
        for label, count in normalized.value_counts().sort_index().items()
    }


def _summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "mean": round(float(array.mean()), 2),
        "median": round(float(np.median(array)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "max": int(array.max()),
    }


def audit_annotations(
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Summarize labels, evidence ranges, and note/criterion lengths per split."""
    split_by_patient: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for patient_id in split_manifest[split_name]["patient_ids"]:
            if patient_id in split_by_patient:
                raise ValueError(f"Patient {patient_id} appears in multiple splits")
            split_by_patient[patient_id] = split_name

    unknown_patients = set(frame["patient_id"]) - set(split_by_patient)
    if unknown_patients:
        raise ValueError("Split manifest does not include every patient in the dataset")

    audit_frame = frame.copy()
    audit_frame["split"] = audit_frame["patient_id"].map(split_by_patient)
    label_counts_by_type = {
        str(criterion_type): _counts(group["expert_eligibility"])
        for criterion_type, group in audit_frame.groupby("criterion_type", dropna=False)
    }
    label_counts_by_split = {
        split_name: _counts(group["expert_eligibility"])
        for split_name, group in audit_frame.groupby("split")
    }

    allowed_by_type = {
        criterion_type: set(labels) for criterion_type, labels in ALLOWED_LABELS.items()
    }
    def is_inconsistent(row: pd.Series) -> bool:
        criterion_type = row["criterion_type"]
        label = row["expert_eligibility"]
        if pd.isna(criterion_type) or pd.isna(label):
            return True
        return label not in allowed_by_type.get(str(criterion_type), set())

    inconsistent = audit_frame.loc[audit_frame.apply(is_inconsistent, axis=1)]

    note_sentence_ids: dict[tuple[str, str], set[int]] = {}
    parsed_notes: dict[tuple[str, str], list[tuple[int, str]]] = {}
    note_lengths: list[dict[str, Any]] = []
    note_parse_failures = 0
    unique_notes = audit_frame[["patient_id", "note", "split"]].drop_duplicates()
    for patient_id, note, split_name in unique_notes.itertuples(index=False, name=None):
        if not isinstance(note, str):
            note_parse_failures += 1
            continue
        try:
            sentences = parse_numbered_note(note)
        except ValueError:
            note_parse_failures += 1
            continue
        parsed_notes[(patient_id, note)] = sentences
        note_sentence_ids[(patient_id, note)] = {sentence_id for sentence_id, _ in sentences}
        note_lengths.append(
            {
                "split": split_name,
                "sentences": len(sentences),
                "words": len(note.split()),
                "tokens": len(tokenizer.encode(note, add_special_tokens=False)),
            }
        )

    out_of_range_annotations: list[int] = []
    out_of_range_ids = 0
    for row in audit_frame.itertuples(index=False):
        valid_ids = note_sentence_ids.get((row.patient_id, row.note))
        if valid_ids is None:
            continue
        invalid_ids = [
            sentence_id
            for sentence_id in row.expert_sentences
            if sentence_id not in valid_ids
        ]
        if invalid_ids:
            out_of_range_annotations.append(int(row.annotation_id))
            out_of_range_ids += len(invalid_ids)

    empty_evidence_by_label: dict[str, dict[str, int | float]] = {}
    for label, group in audit_frame.groupby("expert_eligibility", dropna=False):
        label_name = "<missing>" if pd.isna(label) else str(label)
        empty_count = int(group["expert_sentences"].map(len).eq(0).sum())
        empty_evidence_by_label[label_name] = {
            "n_annotations": int(len(group)),
            "n_empty": empty_count,
            "empty_rate": round(empty_count / len(group), 4) if len(group) else 0.0,
        }

    criteria_per_patient = audit_frame.groupby("patient_id").size().astype(int).tolist()
    criteria_per_trial = audit_frame.groupby("trial_id").size().astype(int).tolist()
    note_summary_by_split: dict[str, dict[str, dict[str, int | float]]] = {}
    for split_name in ("train", "val", "test"):
        split_notes = [row for row in note_lengths if row["split"] == split_name]
        note_summary_by_split[split_name] = {
            field: _summary([int(row[field]) for row in split_notes])
            for field in ("sentences", "words", "tokens")
        }

    criterion_tokens = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in audit_frame["criterion_text"].astype(str)
    ]
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_token_id is None or end_token_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not recognize the <|im_end|> token")
    prompt_target_lengths: list[dict[str, Any]] = []
    allowed_labels = {
        label for labels in ALLOWED_LABELS.values() for label in labels
    }
    for row in audit_frame.itertuples(index=False):
        sentences = parsed_notes.get((row.patient_id, row.note))
        if (
            sentences is None
            or not isinstance(row.criterion_type, str)
            or row.criterion_type not in ALLOWED_LABELS
            or not isinstance(row.expert_eligibility, str)
            or row.expert_eligibility not in allowed_labels
            or not isinstance(row.criterion_text, str)
        ):
            continue
        messages = build_messages(
            sentences, row.criterion_type, row.criterion_text
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        target_ids = tokenizer.encode(
            format_target(row.expert_sentences, row.expert_eligibility),
            add_special_tokens=False,
        )
        prompt_target_lengths.append(
            {
                "split": row.split,
                "tokens": len(prompt_ids) + len(target_ids) + 1,
            }
        )
    prompt_target_by_split = {
        split_name: _summary(
            [
                int(row["tokens"])
                for row in prompt_target_lengths
                if row["split"] == split_name
            ]
        )
        for split_name in ("train", "val", "test")
    }
    return {
        "dataset_id": "ncbi/TrialGPT-Criterion-Annotations",
        "dataset_revision": frame.attrs.get("dataset_revision"),
        "n_annotations": int(len(audit_frame)),
        "n_patients": int(audit_frame["patient_id"].nunique()),
        "n_trials": int(audit_frame["trial_id"].nunique()),
        "dropped_empty_criterion_count": int(
            frame.attrs.get("dropped_empty_criterion_count", 0)
        ),
        "label_counts": {
            "overall": _counts(audit_frame["expert_eligibility"]),
            "by_criterion_type": label_counts_by_type,
            "by_split": label_counts_by_split,
        },
        "label_type_inconsistencies": {
            "count": int(len(inconsistent)),
            "annotation_ids": [int(value) for value in inconsistent["annotation_id"]],
        },
        "evidence": {
            "empty_evidence_by_label": empty_evidence_by_label,
            "note_parse_failures": note_parse_failures,
            "out_of_range_annotation_count": len(out_of_range_annotations),
            "out_of_range_annotation_ids": out_of_range_annotations,
            "out_of_range_sentence_id_count": out_of_range_ids,
        },
        "lengths": {
            "notes": {
                field: _summary([int(row[field]) for row in note_lengths])
                for field in ("sentences", "words", "tokens")
            },
            "notes_by_split": note_summary_by_split,
            "criterion_tokens": _summary(criterion_tokens),
            "prompt_and_target_tokens": {
                "unmeasurable_count": len(audit_frame) - len(prompt_target_lengths),
                "overall": _summary(
                    [int(row["tokens"]) for row in prompt_target_lengths]
                ),
                "by_split": prompt_target_by_split,
            },
        },
        "criteria_per_patient": _summary(criteria_per_patient),
        "criteria_per_trial": _summary(criteria_per_trial),
    }

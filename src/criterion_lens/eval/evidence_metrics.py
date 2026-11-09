"""Evidence-set metrics with explicit empty-gold and hallucinated-ID conventions."""

from collections.abc import Sequence
from typing import Any

from criterion_lens.data.prompts import ALLOWED_LABELS


def _nonnegative_integer(value: object, name: str) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must contain integers")
    return value >= 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    true_positives = false_positives = false_negatives = 0
    gold_nonempty_scores: list[tuple[float, float, float]] = []
    all_record_scores: list[tuple[float, float, float]] = []
    empty_gold_records: list[dict[str, Any]] = []
    predicted_id_count = invalid_id_count = 0

    for record in records:
        gold = record["gold"]
        predicted = record["predicted"]
        true_positive = len(gold & predicted)
        false_positive = len(predicted - gold)
        false_negative = len(gold - predicted)
        true_positives += true_positive
        false_positives += false_positive
        false_negatives += false_negative
        predicted_id_count += len(predicted) + len(record["invalid_ids"])
        invalid_id_count += len(record["invalid_ids"])

        precision = _ratio(true_positive, len(predicted))
        recall = _ratio(true_positive, len(gold))
        scores = (precision, recall, _f1(precision, recall))
        if gold:
            gold_nonempty_scores.append(scores)
            all_record_scores.append(scores)
        elif predicted:
            empty_gold_records.append(record)
            all_record_scores.append((0.0, 0.0, 0.0))
        else:
            empty_gold_records.append(record)
            all_record_scores.append((1.0, 1.0, 1.0))

    micro_precision = _ratio(true_positives, true_positives + false_positives)
    micro_recall = _ratio(true_positives, true_positives + false_negatives)
    macro = (
        tuple(
            sum(scores[index] for scores in gold_nonempty_scores)
            / len(gold_nonempty_scores)
            for index in range(3)
        )
        if gold_nonempty_scores
        else (0.0, 0.0, 0.0)
    )
    macro_all = (
        tuple(
            sum(scores[index] for scores in all_record_scores) / len(all_record_scores)
            for index in range(3)
        )
        if all_record_scores
        else (0.0, 0.0, 0.0)
    )
    return {
        "n_records": len(records),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": _f1(micro_precision, micro_recall),
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
        },
        "macro_nonempty_gold": {
            "n_records": len(gold_nonempty_scores),
            "precision": macro[0],
            "recall": macro[1],
            "f1": macro[2],
        },
        "macro_all_records_secondary": {
            "precision": macro_all[0],
            "recall": macro_all[1],
            "f1": macro_all[2],
        },
        "empty_gold": {
            "n_records": len(empty_gold_records),
            "abstention_accuracy": (
                sum(not record["predicted"] for record in empty_gold_records)
                / len(empty_gold_records)
                if empty_gold_records
                else None
            ),
            "mean_false_citations": (
                sum(len(record["predicted"]) for record in empty_gold_records)
                / len(empty_gold_records)
                if empty_gold_records
                else None
            ),
        },
        "invalid_ids": {
            "count": invalid_id_count,
            "rate_of_predicted_ids": _ratio(invalid_id_count, predicted_id_count),
        },
        "hallucinated_id_rate": _ratio(invalid_id_count, predicted_id_count),
    }


def evidence_metrics(
    gold_evidence: Sequence[Sequence[int]],
    predicted_evidence: Sequence[Sequence[int]],
    sentence_counts: Sequence[int],
    criterion_types: Sequence[str],
    gold_labels: Sequence[str],
) -> dict[str, Any]:
    """Compute micro/macro evidence scores and grouped empty-gold diagnostics."""
    size = len(gold_evidence)
    if size == 0:
        raise ValueError("At least one record is required")
    if not all(
        len(values) == size
        for values in (
            predicted_evidence,
            sentence_counts,
            criterion_types,
            gold_labels,
        )
    ):
        raise ValueError("All evidence and metadata sequences must have equal lengths")

    records: list[dict[str, Any]] = []
    for index, (gold_ids, predicted_ids, count, criterion_type, gold_label) in enumerate(
        zip(
            gold_evidence,
            predicted_evidence,
            sentence_counts,
            criterion_types,
            gold_labels,
            strict=True,
        )
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"sentence_counts[{index}] must be a non-negative integer")
        if criterion_type not in ("inclusion", "exclusion"):
            raise ValueError(f"Unsupported criterion type at index {index}")
        if gold_label not in ALLOWED_LABELS[criterion_type]:
            raise ValueError(f"Invalid gold label at index {index}")
        gold: set[int] = set()
        for sentence_id in gold_ids:
            if not _nonnegative_integer(sentence_id, "gold_evidence"):
                raise ValueError("gold evidence IDs must be non-negative")
            if sentence_id >= count:
                raise ValueError(f"Gold evidence ID at index {index} is outside the note")
            gold.add(sentence_id)
        predicted: set[int] = set()
        invalid_ids: set[int] = set()
        for sentence_id in predicted_ids:
            is_nonnegative = _nonnegative_integer(sentence_id, "predicted_evidence")
            if not is_nonnegative or sentence_id >= count:
                invalid_ids.add(sentence_id)
            else:
                predicted.add(sentence_id)
        records.append(
            {
                "gold": gold,
                "predicted": predicted,
                "invalid_ids": invalid_ids,
                "criterion_type": criterion_type,
                "gold_label": gold_label,
            }
        )

    return {
        "overall": _summarize(records),
        "by_criterion_type": {
            criterion_type: _summarize(
                [record for record in records if record["criterion_type"] == criterion_type]
            )
            for criterion_type in ("inclusion", "exclusion")
        },
        "by_gold_label": {
            label: _summarize(
                [record for record in records if record["gold_label"] == label]
            )
            for label in sorted({record["gold_label"] for record in records})
        },
        "conventions": {
            "micro": "Sum TP, FP, and FN across records; empty-gold/empty-pred records add none.",
            "primary_macro": (
                "Average per-record scores only where gold evidence is non-empty; "
                "empty predictions score zero."
            ),
            "empty_gold": "Report abstention accuracy and mean valid false citations separately.",
            "secondary_macro": (
                "Include every record; empty-gold/empty-pred scores 1, "
                "empty-gold/non-empty-pred scores 0."
            ),
            "invalid_ids": (
                "Predicted IDs outside [0, sentence_count) are removed and counted separately."
            ),
        },
    }

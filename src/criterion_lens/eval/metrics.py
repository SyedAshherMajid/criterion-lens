"""Classification metrics for criterion-level eligibility predictions."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from criterion_lens.data.prompts import ALLOWED_LABELS

_ALL_LABELS = (
    "included",
    "not included",
    "not enough information",
    "not applicable",
    "excluded",
    "not excluded",
)
_UNIFIED_LABELS = (
    "criterion met",
    "criterion not met",
    "not enough information",
    "not applicable",
)
_UNIFIED_MAP = {
    "included": "criterion met",
    "excluded": "criterion met",
    "not included": "criterion not met",
    "not excluded": "criterion not met",
    "not enough information": "not enough information",
    "not applicable": "not applicable",
}


def _validate_records(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    criterion_types: Sequence[str],
) -> None:
    if len(y_true) == 0:
        raise ValueError("At least one record is required")
    if not (len(y_true) == len(y_pred) == len(criterion_types)):
        raise ValueError("Labels and criterion_types must have equal lengths")
    for index, (gold, predicted, criterion_type) in enumerate(
        zip(y_true, y_pred, criterion_types, strict=True)
    ):
        if criterion_type not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported criterion type at index {index}: {criterion_type}")
        allowed = ALLOWED_LABELS[criterion_type]
        if gold not in allowed or predicted not in allowed:
            raise ValueError(
                f"Gold and predicted labels at index {index} must be valid "
                f"{criterion_type} labels"
            )


def _diagnostic_rates(
    indices: Sequence[int],
    parse_failures: Sequence[bool] | None,
    fallback_used: Sequence[bool] | None,
    invalid_labels: Sequence[bool] | None,
) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for name, flags in (
        ("parse_failure_rate", parse_failures),
        ("fallback_usage_rate", fallback_used),
        ("invalid_label_rate", invalid_labels),
    ):
        rates[name] = (
            None
            if flags is None or not indices
            else float(np.mean([flags[index] for index in indices]))
        )
    return rates


def _classification_summary(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    indices: Sequence[int],
    parse_failures: Sequence[bool] | None,
    fallback_used: Sequence[bool] | None,
    invalid_labels: Sequence[bool] | None,
) -> dict[str, Any]:
    if not indices:
        zeros = np.zeros((len(labels), len(labels)), dtype=np.int64).tolist()
        return {
            "n": 0,
            "labels": list(labels),
            "accuracy": None,
            "macro_f1": None,
            "fixed_label_macro_f1": None,
            "per_class": {
                label: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 0.0,
                }
                for label in labels
            },
            "confusion_matrix": zeros,
            "confusion_matrix_row_normalized": zeros,
            **_diagnostic_rates(
                indices, parse_failures, fallback_used, invalid_labels
            ),
        }

    gold = [y_true[index] for index in indices]
    predicted = [y_pred[index] for index in indices]
    observed_labels = sorted(set(gold) | set(predicted))
    report = classification_report(
        gold,
        predicted,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    counts = confusion_matrix(gold, predicted, labels=list(labels))
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_sums,
        out=np.zeros(counts.shape, dtype=np.float64),
        where=row_sums != 0,
    )
    return {
        "n": len(indices),
        "labels": list(labels),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=observed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "fixed_label_macro_f1": float(
            f1_score(gold, predicted, labels=list(labels), average="macro", zero_division=0)
        ),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in labels
        },
        "confusion_matrix": counts.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
        **_diagnostic_rates(
            indices, parse_failures, fallback_used, invalid_labels
        ),
    }


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    criterion_types: Sequence[str],
    *,
    parse_failures: Sequence[bool] | None = None,
    fallback_used: Sequence[bool] | None = None,
    invalid_labels: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Report six-label, type-specific, and supplementary unified metrics."""
    _validate_records(y_true, y_pred, criterion_types)
    for name, flags in (
        ("parse_failures", parse_failures),
        ("fallback_used", fallback_used),
        ("invalid_labels", invalid_labels),
    ):
        if flags is not None and len(flags) != len(y_true):
            raise ValueError(f"{name} must have the same length as labels")

    all_indices = list(range(len(y_true)))
    inclusion_indices = [
        index for index, kind in enumerate(criterion_types) if kind == "inclusion"
    ]
    exclusion_indices = [
        index for index, kind in enumerate(criterion_types) if kind == "exclusion"
    ]
    unified_true = [_UNIFIED_MAP[label] for label in y_true]
    unified_pred = [_UNIFIED_MAP[label] for label in y_pred]
    return {
        "overall": _classification_summary(
            y_true,
            y_pred,
            _ALL_LABELS,
            all_indices,
            parse_failures,
            fallback_used,
            invalid_labels,
        ),
        "inclusion": _classification_summary(
            y_true,
            y_pred,
            ALLOWED_LABELS["inclusion"],
            inclusion_indices,
            parse_failures,
            fallback_used,
            invalid_labels,
        ),
        "exclusion": _classification_summary(
            y_true,
            y_pred,
            ALLOWED_LABELS["exclusion"],
            exclusion_indices,
            parse_failures,
            fallback_used,
            invalid_labels,
        ),
        "unified_supplementary": _classification_summary(
            unified_true,
            unified_pred,
            _UNIFIED_LABELS,
            all_indices,
            parse_failures,
            fallback_used,
            invalid_labels,
        ),
    }


def bootstrap_confidence_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap criterion-level accuracy and fixed six-label macro-F1 intervals."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Non-empty y_true and equally sized y_pred are required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    gold = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(n_resamples, dtype=np.float64)
    macro_f1_samples = np.empty(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        selected = rng.integers(0, len(gold), size=len(gold))
        sample_gold = gold[selected]
        sample_predicted = predicted[selected]
        accuracy_samples[sample_index] = accuracy_score(
            sample_gold, sample_predicted
        )
        macro_f1_samples[sample_index] = f1_score(
            sample_gold,
            sample_predicted,
            labels=list(_ALL_LABELS),
            average="macro",
            zero_division=0,
        )
    return {
        "confidence_level": 0.95,
        "n_resamples": n_resamples,
        "seed": seed,
        "accuracy": np.percentile(accuracy_samples, [2.5, 97.5]).tolist(),
        "macro_f1": np.percentile(macro_f1_samples, [2.5, 97.5]).tolist(),
    }


def save_confusion_matrices(
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Save count and row-normalized confusion matrices for each label view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for view_name in ("overall", "inclusion", "exclusion"):
        summary = metrics[view_name]
        labels = (
            _ALL_LABELS
            if view_name == "overall"
            else ALLOWED_LABELS[view_name]
        )
        for matrix_key, suffix, value_format in (
            ("confusion_matrix", "counts", "d"),
            ("confusion_matrix_row_normalized", "normalized", ".2f"),
        ):
            matrix = np.asarray(summary[matrix_key])
            figure = Figure(figsize=(8, 6))
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            image = axis.imshow(matrix, cmap="Blues", vmin=0)
            figure.colorbar(image, ax=axis)
            axis.set(
                xticks=np.arange(len(labels)),
                yticks=np.arange(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                xlabel="Predicted label",
                ylabel="Gold label",
                title=f"{view_name.title()} confusion matrix ({suffix})",
            )
            for tick in axis.get_xticklabels():
                tick.set_rotation(35)
                tick.set_ha("right")
            for row in range(len(labels)):
                for column in range(len(labels)):
                    axis.text(
                        column,
                        row,
                        format(matrix[row, column], value_format),
                        ha="center",
                        va="center",
                    )
            figure.tight_layout()
            path = output_dir / f"confusion_{view_name}_{suffix}.png"
            figure.savefig(path, dpi=160)
            figure.clear()
            paths.append(path)
    return paths

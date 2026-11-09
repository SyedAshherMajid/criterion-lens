import pytest

from criterion_lens.eval.evidence_metrics import evidence_metrics
from criterion_lens.eval.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    save_confusion_matrices,
)


def test_classification_metrics_include_type_views_and_diagnostic_rates() -> None:
    metrics = classification_metrics(
        ["included", "not included", "excluded", "not excluded"],
        ["included", "not included", "not excluded", "excluded"],
        ["inclusion", "inclusion", "exclusion", "exclusion"],
        parse_failures=[False, True, False, False],
        fallback_used=[False, True, True, False],
        invalid_labels=[False, False, False, True],
    )

    assert metrics["overall"]["n"] == 4
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["macro_f1"] == 0.5
    assert metrics["overall"]["fixed_label_macro_f1"] == pytest.approx(1 / 3)
    assert metrics["overall"]["parse_failure_rate"] == 0.25
    assert metrics["overall"]["fallback_usage_rate"] == 0.5
    assert metrics["overall"]["invalid_label_rate"] == 0.25
    assert metrics["inclusion"]["accuracy"] == 1
    assert metrics["exclusion"]["accuracy"] == 0
    assert metrics["unified_supplementary"]["accuracy"] == 0.5
    assert metrics["overall"]["confusion_matrix"][0][0] == 1
    assert metrics["overall"]["confusion_matrix_row_normalized"][2] == [0] * 6


def test_classification_metrics_support_a_missing_criterion_type() -> None:
    metrics = classification_metrics(["included"], ["included"], ["inclusion"])

    assert metrics["exclusion"]["n"] == 0
    assert metrics["exclusion"]["accuracy"] is None
    assert metrics["exclusion"]["parse_failure_rate"] is None


def test_classification_metrics_reject_incompatible_labels() -> None:
    with pytest.raises(ValueError, match="valid inclusion labels"):
        classification_metrics(["excluded"], ["included"], ["inclusion"])


def test_bootstrap_confidence_intervals_are_repeatable() -> None:
    labels = ["included", "not included", "excluded", "not excluded"]
    predictions = ["included", "included", "excluded", "not excluded"]

    first = bootstrap_confidence_intervals(labels, predictions, n_resamples=200)
    second = bootstrap_confidence_intervals(labels, predictions, n_resamples=200)

    assert first == second
    assert first["confidence_level"] == 0.95
    assert first["n_resamples"] == 200
    assert first["accuracy"][0] <= 0.5 <= first["accuracy"][1]
    assert all(0 <= bound <= 1 for bound in first["accuracy"])


def test_confusion_matrix_images_are_written(tmp_path) -> None:
    metrics = classification_metrics(
        ["included", "excluded"],
        ["included", "not excluded"],
        ["inclusion", "exclusion"],
    )

    paths = save_confusion_matrices(metrics, tmp_path)

    assert len(paths) == 6
    assert all(path.is_file() for path in paths)


def test_evidence_metrics_cover_empty_gold_and_prediction_combinations() -> None:
    metrics = evidence_metrics(
        gold_evidence=[(0, 1), (2,), (), (), (0,)],
        predicted_evidence=[(0, 2), (), (1,), (), (99, -1)],
        sentence_counts=[3, 3, 2, 2, 1],
        criterion_types=[
            "inclusion",
            "inclusion",
            "exclusion",
            "exclusion",
            "exclusion",
        ],
        gold_labels=[
            "included",
            "included",
            "excluded",
            "not enough information",
            "excluded",
        ],
    )

    overall = metrics["overall"]
    assert overall["micro"] == {
        "precision": pytest.approx(1 / 3),
        "recall": pytest.approx(1 / 4),
        "f1": pytest.approx(2 / 7),
        "tp": 1,
        "fp": 2,
        "fn": 3,
    }
    assert overall["macro_nonempty_gold"]["n_records"] == 3
    assert overall["macro_nonempty_gold"]["precision"] == pytest.approx(1 / 6)
    assert overall["macro_nonempty_gold"]["recall"] == pytest.approx(1 / 6)
    assert overall["macro_nonempty_gold"]["f1"] == pytest.approx(1 / 6)
    assert overall["macro_all_records_secondary"]["f1"] == pytest.approx(0.3)
    assert overall["empty_gold"]["abstention_accuracy"] == 0.5
    assert overall["empty_gold"]["mean_false_citations"] == 0.5
    assert overall["invalid_ids"]["count"] == 2
    assert overall["hallucinated_id_rate"] == 0.4
    assert metrics["by_criterion_type"]["inclusion"]["n_records"] == 2
    assert metrics["by_gold_label"]["not enough information"]["n_records"] == 1


def test_evidence_metrics_reject_out_of_range_gold_and_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="outside the note"):
        evidence_metrics([(2,)], [()], [2], ["inclusion"], ["included"])
    with pytest.raises(ValueError, match="Invalid gold label"):
        evidence_metrics([()], [()], [1], ["inclusion"], ["excluded"])

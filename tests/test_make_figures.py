import json

import pytest

from criterion_lens.eval.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
)
from scripts.make_figures import _load_evaluations, _make_architecture_figures


def test_architecture_diagrams_are_written_to_both_figure_directories(tmp_path):
    results_figures = tmp_path / "results" / "figures"
    report_figures = tmp_path / "report" / "figures"

    _make_architecture_figures(results_figures, report_figures)

    expected = {
        "system_pipeline.png",
        "finetuning_pipeline.png",
        "kv_cache_flow.png",
        "tiled_attention_flow.png",
        "speculative_decoding_flow.png",
    }
    assert {path.name for path in results_figures.iterdir()} == expected
    assert {path.name for path in report_figures.iterdir()} == expected


def test_load_evaluations_bootstraps_saved_predictions(tmp_path):
    run_dir = tmp_path / "eval" / "val_ft_1p5b"
    run_dir.mkdir(parents=True)
    rows = [
        {"gold": {"label": "included"}, "label": "included", "criterion_type": "inclusion"},
        {"gold": {"label": "excluded"}, "label": "excluded", "criterion_type": "exclusion"},
        {
            "gold": {"label": "not included"},
            "label": "not included",
            "criterion_type": "inclusion",
        },
        {
            "gold": {"label": "not excluded"},
            "label": "excluded",
            "criterion_type": "exclusion",
        },
    ]
    classification = classification_metrics(
        [row["gold"]["label"] for row in rows],
        [row["label"] for row in rows],
        [row["criterion_type"] for row in rows],
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "split": "val",
                "classification": classification,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    evaluations = _load_evaluations(tmp_path)

    assert evaluations["val_ft_1p5b"]["bootstrap_95_ci"] == (
        bootstrap_confidence_intervals(
            [row["gold"]["label"] for row in rows],
            [row["label"] for row in rows],
        )
    )


def test_load_evaluations_rejects_mismatched_predictions(tmp_path):
    run_dir = tmp_path / "eval" / "val_ft_1p5b"
    run_dir.mkdir(parents=True)
    classification = classification_metrics(["included"], ["included"], ["inclusion"])
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "split": "val",
                "classification": classification,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "gold": {"label": "included"},
                "label": "not included",
                "criterion_type": "inclusion",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Prediction metrics do not match saved metrics"):
        _load_evaluations(tmp_path)

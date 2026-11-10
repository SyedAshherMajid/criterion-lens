from scripts import error_analysis


def _record(
    *,
    annotation_id: int,
    label: str = "included",
    evidence_ids: list[int] | None = None,
    unsupported: bool = False,
) -> dict:
    return {
        "annotation_id": annotation_id,
        "criterion_type": "inclusion",
        "criterion_text": "The patient must be at least 18 years old.",
        "label": label,
        "evidence_ids": evidence_ids or [],
        "unsupported_decision": unsupported,
        "invalid_ids_dropped": [],
        "gold": {"label": "not enough information", "evidence_ids": [0]},
    }


def test_error_analysis_flags_distinct_decision_and_evidence_failures() -> None:
    record = _record(annotation_id=3, evidence_ids=[8], unsupported=True)

    categories = error_analysis._categories(record)

    assert categories == [
        "missing_information_guess",
        "unsupported_decision",
        "missed_gold_evidence",
        "evidence_set_discrepancy",
    ]


def test_error_analysis_merges_models_without_copying_note_text() -> None:
    record = _record(annotation_id=3)
    result = error_analysis.analyze_predictions(
        {"ft_1p5b": [record], "ft_0p5b": [record]},
        max_cases=10,
    )

    assert result["unique_candidate_annotations"] == 1
    assert [row["model"] for row in result["cases"][0]["model_outputs"]] == [
        "ft_1p5b",
        "ft_0p5b",
    ]
    assert "note" not in result["cases"][0]

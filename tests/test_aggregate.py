import pytest

from criterion_lens.inference.aggregate import aggregate_trial
from criterion_lens.inference.schema import CriterionResult


def _result(
    criterion_id: str,
    criterion_type: str,
    label: str,
    evidence_ids: list[int] | None = None,
) -> CriterionResult:
    ids = evidence_ids or []
    return CriterionResult(
        criterion_id=criterion_id,
        criterion_type=criterion_type,
        criterion_text="A sample criterion",
        label=label,
        evidence_ids=ids,
        evidence_text=[f"Evidence {sentence_id}" for sentence_id in ids],
        explanation="Grounded explanation",
        missing_information=label == "not enough information" or not ids,
        unsupported_decision=label in {
            "included",
            "not included",
            "excluded",
            "not excluded",
        }
        and not ids,
        invalid_ids_dropped=[],
        raw_output="{}",
        parse_ok=True,
        fallback_used=False,
        decoding={"mode": "no_cache", "max_new_tokens": 64},
    )


@pytest.mark.parametrize(
    ("criteria", "expected"),
    [
        (
            [_result("inc-1", "inclusion", "not included", [0])],
            "likely ineligible",
        ),
        (
            [
                _result("inc-1", "inclusion", "included", [0]),
                _result("exc-1", "exclusion", "excluded", [1]),
            ],
            "likely ineligible",
        ),
        (
            [
                _result("inc-1", "inclusion", "included", [0]),
                _result("exc-1", "exclusion", "not excluded", [1]),
            ],
            "potentially eligible (preliminary)",
        ),
        (
            [_result("inc-1", "inclusion", "not enough information")],
            "needs manual review",
        ),
        (
            [_result("inc-1", "inclusion", "not applicable")],
            "needs manual review",
        ),
    ],
)
def test_aggregate_trial_verdict_branches(
    criteria: list[CriterionResult], expected: str
) -> None:
    aggregate = aggregate_trial(criteria)

    assert aggregate["verdict"] == expected
    assert aggregate["is_ground_truth"] is False
    assert "not a ground-truth" in aggregate["interpretation"]


def test_absent_evidence_is_not_a_proof_of_failure() -> None:
    aggregate = aggregate_trial(
        [_result("inc-1", "inclusion", "not included", evidence_ids=[])]
    )

    assert aggregate["verdict"] == "needs manual review"
    assert aggregate["counts"]["not_met"] == 0
    assert aggregate["counts"]["unsupported"] == 1
    assert aggregate["unsupported_criteria"] == ["inc-1"]
    assert aggregate["missing_information_criteria"] == ["inc-1"]


def test_not_applicable_is_ignored_and_counts_are_reported() -> None:
    aggregate = aggregate_trial(
        [
            _result("inc-1", "inclusion", "included", [0]),
            _result("inc-2", "inclusion", "not applicable"),
            _result("exc-1", "exclusion", "not excluded", [1]),
        ]
    )

    assert aggregate["verdict"] == "potentially eligible (preliminary)"
    assert aggregate["counts"] == {
        "met": 2,
        "not_met": 0,
        "not_enough_information": 0,
        "not_applicable": 1,
        "unsupported": 0,
    }


def test_aggregate_rejects_duplicate_ids_and_incompatible_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate criterion_id"):
        aggregate_trial(
            [
                _result("inc-1", "inclusion", "included", [0]),
                _result("inc-1", "inclusion", "included", [1]),
            ]
        )
    with pytest.raises(ValueError, match="Invalid label"):
        aggregate_trial([_result("inc-1", "inclusion", "excluded", [0])])

"""Rule-based aggregation of criterion results into a preliminary trial verdict."""

from collections.abc import Sequence
from typing import Any

from criterion_lens.data.prompts import ALLOWED_LABELS
from criterion_lens.inference.schema import CriterionResult

_NEI = "not enough information"
_NOT_APPLICABLE = "not applicable"
_DECISIVE_LABELS = {
    "included",
    "not included",
    "excluded",
    "not excluded",
}


def aggregate_trial(criteria: Sequence[CriterionResult]) -> dict[str, Any]:
    """Apply the documented screening rule; the verdict is not ground truth."""
    seen_ids: set[str] = set()
    counts = {
        "met": 0,
        "not_met": 0,
        "not_enough_information": 0,
        "not_applicable": 0,
        "unsupported": 0,
    }
    missing_information: list[str] = []
    unsupported_criteria: list[str] = []
    inclusion_count = 0
    evidence_backed_failure = False
    needs_review = False

    for result in criteria:
        if result.criterion_id in seen_ids:
            raise ValueError(f"Duplicate criterion_id: {result.criterion_id}")
        seen_ids.add(result.criterion_id)
        if result.criterion_type not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported criterion type: {result.criterion_type}")
        if result.label not in ALLOWED_LABELS[result.criterion_type]:
            raise ValueError(
                f"Invalid label {result.label!r} for {result.criterion_type} criterion"
            )
        if result.label == _NOT_APPLICABLE:
            counts["not_applicable"] += 1
            continue
        if result.criterion_type == "inclusion":
            inclusion_count += 1

        has_evidence = bool(result.evidence_ids)
        decisive = result.label in _DECISIVE_LABELS
        unsupported = decisive and not has_evidence
        if result.label == _NEI:
            counts["not_enough_information"] += 1
            needs_review = True
        elif unsupported:
            counts["unsupported"] += 1
            unsupported_criteria.append(result.criterion_id)
            needs_review = True
        elif result.label in ("included", "not excluded"):
            counts["met"] += 1
        else:
            counts["not_met"] += 1

        if result.label == _NEI or not has_evidence:
            missing_information.append(result.criterion_id)
        if result.criterion_type == "inclusion" and result.label == "not included":
            evidence_backed_failure = evidence_backed_failure or has_evidence
        if result.criterion_type == "exclusion" and result.label == "excluded":
            evidence_backed_failure = evidence_backed_failure or has_evidence

    if evidence_backed_failure:
        verdict = "likely ineligible"
    elif needs_review or inclusion_count == 0:
        verdict = "needs manual review"
    else:
        verdict = "potentially eligible (preliminary)"

    return {
        "verdict": verdict,
        "is_ground_truth": False,
        "interpretation": (
            "Rule-based screening summary only; not a ground-truth eligibility decision."
        ),
        "counts": counts,
        "missing_information_criteria": missing_information,
        "unsupported_criteria": unsupported_criteria,
    }

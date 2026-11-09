"""Grounded explanation templates for validated criterion results."""

from collections.abc import Sequence

_DECISIVE_LABELS = {
    "included",
    "not included",
    "excluded",
    "not excluded",
}


def build_explanation(
    label: str,
    criterion_type: str,
    evidence_ids: Sequence[int],
    evidence_text: Sequence[str],
) -> str:
    """Create an explanation using only the label and cited note sentences."""
    if len(evidence_ids) != len(evidence_text):
        raise ValueError("evidence_ids and evidence_text must have equal lengths")
    if label == "not enough information":
        related = (
            f"; closest related sentence(s): {', '.join(map(str, evidence_ids))}"
            if evidence_ids
            else ""
        )
        return (
            "The note does not state the information needed to judge this criterion"
            f"{related}. Absence of information is not treated as proof of eligibility "
            "or ineligibility."
        )
    if label == "not applicable":
        if evidence_ids:
            return (
                "The criterion does not apply to this patient given sentence(s) "
                f"{', '.join(map(str, evidence_ids))}."
            )
        return "The criterion does not apply to this patient."

    evidence_clause = ""
    if evidence_ids:
        citations = "; ".join(
            f"{sentence_id}: “{text}”"
            for sentence_id, text in zip(evidence_ids, evidence_text, strict=True)
        )
        evidence_clause = (
            f" based on sentence(s) {', '.join(map(str, evidence_ids))}: {citations}"
        )
    explanation = (
        f"Judged '{label}' for this {criterion_type} criterion{evidence_clause}. "
        "The decision relies only on these statements."
    )
    if label in _DECISIVE_LABELS and not evidence_ids:
        explanation += (
            " [UNSUPPORTED] No supporting sentence was cited; "
            "treat this decision as unverified."
        )
    return explanation

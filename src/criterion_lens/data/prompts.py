"""Build expert-label prompts without exposing GPT-4 annotation fields."""

import json
from collections.abc import Sequence

SYSTEM_PROMPT = (
    "You are a clinical-trial screening assistant. Judge ONE eligibility criterion against a "
    "patient note. Use only facts stated in the note. Cite the IDs of the note sentences that "
    'support your judgment. If the note does not contain the needed information, answer "not '
    'enough information". If the criterion cannot apply to this patient (e.g., wrong sex, '
    "pregnancy criterion for a male), answer \"not applicable\". Respond with one JSON object only."
)

ALLOWED_LABELS = {
    "inclusion": (
        "included",
        "not included",
        "not enough information",
        "not applicable",
    ),
    "exclusion": (
        "excluded",
        "not excluded",
        "not enough information",
        "not applicable",
    ),
}
_ALL_LABELS = frozenset(label for labels in ALLOWED_LABELS.values() for label in labels)


def build_messages(
    sentences: Sequence[tuple[int, str]], criterion_type: str, criterion_text: str
) -> list[dict[str, str]]:
    """Build the system/user messages from note evidence and one criterion only."""
    criterion_type = criterion_type.strip().lower()
    if criterion_type not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported criterion_type: {criterion_type}")

    note_lines = []
    seen_ids: set[int] = set()
    for sentence_id, sentence_text in sentences:
        if not isinstance(sentence_id, int) or isinstance(sentence_id, bool) or sentence_id < 0:
            raise ValueError("Sentence IDs must be non-negative integers")
        if sentence_id in seen_ids:
            raise ValueError(f"Duplicate sentence ID: {sentence_id}")
        if not isinstance(sentence_text, str) or not sentence_text.strip():
            raise ValueError(f"Sentence {sentence_id} must have non-empty text")
        seen_ids.add(sentence_id)
        note_lines.append(f"{sentence_id}. {sentence_text.strip()}")

    allowed_labels = " | ".join(f'"{label}"' for label in ALLOWED_LABELS[criterion_type])
    note_text = "\n".join(note_lines)
    user_message = (
        "Patient note (sentences are numbered):\n"
        f"{note_text}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n"
        f"Allowed labels: {allowed_labels}\n\n"
        'Return: {"evidence": [<sentence ids>], "label": "<one allowed label>"}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def format_target(evidence_ids: Sequence[int], label: str) -> str:
    """Serialize a normalized expert target as compact evidence-first JSON."""
    if label not in _ALL_LABELS:
        raise ValueError(f"Unsupported label: {label}")
    if any(
        not isinstance(sentence_id, int)
        or isinstance(sentence_id, bool)
        or sentence_id < 0
        for sentence_id in evidence_ids
    ):
        raise ValueError("Evidence IDs must be non-negative integers")
    return json.dumps(
        {"evidence": sorted(set(evidence_ids)), "label": label},
        ensure_ascii=False,
    )

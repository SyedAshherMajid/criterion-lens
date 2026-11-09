"""Parse and validate structured model outputs."""

import json
import re
from typing import Any

from criterion_lens.data.prompts import ALLOWED_LABELS
from criterion_lens.inference.schema import ParsedOutput

_LABEL_PATTERN = re.compile(r"""["']?label["']?\s*:\s*["']([^"']+)["']""", re.IGNORECASE)
_EVIDENCE_PATTERN = re.compile(
    r"""["']?evidence["']?\s*:\s*\[([^\]]*)\]""", re.IGNORECASE
)
_INTEGER_PATTERN = re.compile(r"-?\d+\Z")


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _regex_fields(text: str) -> tuple[str | None, list[int], bool] | None:
    label_match = _LABEL_PATTERN.search(text)
    evidence_match = _EVIDENCE_PATTERN.search(text)
    if label_match is None or evidence_match is None:
        return None

    evidence_text = evidence_match.group(1).strip()
    if not evidence_text:
        return label_match.group(1), [], True

    pieces = [piece.strip() for piece in evidence_text.split(",")]
    integer_pieces = [piece for piece in pieces if _INTEGER_PATTERN.fullmatch(piece)]
    return (
        label_match.group(1),
        [int(piece) for piece in integer_pieces],
        len(integer_pieces) == len(pieces),
    )


def parse_output(text: str, criterion_type: str, n_sentences: int) -> ParsedOutput:
    """Extract a label and evidence IDs, dropping and recording IDs out of range."""
    normalized_type = criterion_type.strip().lower()
    if normalized_type not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported criterion_type: {criterion_type}")
    if not isinstance(n_sentences, int) or isinstance(n_sentences, bool) or n_sentences < 0:
        raise ValueError("n_sentences must be a non-negative integer")

    payload = _first_json_object(text)
    raw_label: str | None = None
    candidate_ids: list[int] = []
    fields_valid = False

    if payload is not None:
        payload_label = payload.get("label")
        payload_evidence = payload.get("evidence")
        if isinstance(payload_evidence, list):
            candidate_ids = [
                value
                for value in payload_evidence
                if isinstance(value, int) and not isinstance(value, bool)
            ]
        fields_valid = (
            isinstance(payload_label, str)
            and isinstance(payload_evidence, list)
            and len(candidate_ids) == len(payload_evidence)
        )
        if isinstance(payload_label, str):
            raw_label = payload_label

    if not fields_valid:
        regex_result = _regex_fields(text)
        if regex_result is not None:
            raw_label, candidate_ids, fields_valid = regex_result

    normalized_label = raw_label.strip().lower() if raw_label is not None else None
    label = (
        normalized_label
        if normalized_label in ALLOWED_LABELS[normalized_type]
        else None
    )
    invalid_label = (
        normalized_label is not None
        and normalized_label not in ALLOWED_LABELS[normalized_type]
    )
    in_range_ids = sorted({value for value in candidate_ids if 0 <= value < n_sentences})
    invalid_ids = sorted({value for value in candidate_ids if value < 0 or value >= n_sentences})
    return ParsedOutput(
        label=label,
        evidence_ids=in_range_ids,
        invalid_ids_dropped=invalid_ids,
        raw_output=text,
        parse_ok=fields_valid and label is not None and not invalid_ids,
        invalid_label=invalid_label,
    )

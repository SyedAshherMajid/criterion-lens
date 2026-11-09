"""Typed result records for criterion-level inference."""

from dataclasses import dataclass
from typing import Any, Literal

CriterionType = Literal["inclusion", "exclusion"]


@dataclass
class ParsedOutput:
    """Validated fields extracted from one model response."""

    label: str | None
    evidence_ids: list[int]
    invalid_ids_dropped: list[int]
    raw_output: str
    parse_ok: bool
    invalid_label: bool


@dataclass
class CriterionResult:
    """Complete user-facing result for one eligibility criterion."""

    criterion_id: str
    criterion_type: CriterionType
    criterion_text: str
    label: str
    evidence_ids: list[int]
    evidence_text: list[str]
    explanation: str
    missing_information: bool
    unsupported_decision: bool
    invalid_ids_dropped: list[int]
    raw_output: str
    parse_ok: bool
    invalid_label: bool
    fallback_used: bool
    decoding: dict[str, Any]

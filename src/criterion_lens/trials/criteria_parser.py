"""Parse human-readable inclusion and exclusion criteria sections."""

import re
from dataclasses import dataclass
from typing import Literal

CriterionType = Literal["inclusion", "exclusion"]
_HEADING = re.compile(r"^\s*(inclusion|exclusion)\s+criteria\s*:?\s*$", re.IGNORECASE)
_ITEM = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class CriterionInput:
    criterion_type: CriterionType
    criterion_text: str


def parse_criteria_text(text: str) -> list[CriterionInput]:
    """Split headed criteria text into typed items; never infer a missing section type."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Criteria text must not be empty")

    criteria: list[CriterionInput] = []
    active_type: CriterionType | None = None
    current: list[str] = []

    def finish_item() -> None:
        if active_type is not None and current:
            criteria.append(CriterionInput(active_type, " ".join(current).strip()))
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = _HEADING.fullmatch(line)
        if heading:
            finish_item()
            active_type = (
                "inclusion" if heading.group(1).lower() == "inclusion" else "exclusion"
            )
            continue
        if active_type is None:
            raise ValueError(
                "Criteria items must follow an Inclusion Criteria or "
                "Exclusion Criteria heading"
            )
        item = _ITEM.sub("", line).strip()
        if not item:
            continue
        if _ITEM.match(line):
            finish_item()
        current.append(item)

    finish_item()
    if not criteria:
        raise ValueError("No criteria items found under Inclusion Criteria or Exclusion Criteria")
    return criteria

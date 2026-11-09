"""Export provenance-aware screening results as JSON, Markdown, and HTML."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

from criterion_lens.inference.schema import CriterionResult

DISCLAIMER = (
    "CriterionLens is a research prototype. This preliminary screening summary is "
    "not medical advice and is not a ground-truth eligibility decision. A qualified "
    "human must review the source records and trial criteria."
)


@dataclass(frozen=True)
class ReportExports:
    json_text: str
    markdown_text: str
    html_text: str
    saved_files: dict[str, Path]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CriterionLens Screening Summary",
        "",
        f"**Preliminary outcome:** {payload['aggregation']['verdict']}",
        "",
        payload["aggregation"]["interpretation"],
        "",
        f"**Generated (UTC):** {payload['created_at_utc']}",
        f"**Patient note SHA-256:** `{payload['patient_note_sha256']}`",
        f"**Criteria SHA-256:** `{payload['criteria_sha256']}`",
        "",
        "## Criterion results",
        "",
    ]
    for result in payload["criteria"]:
        evidence = ", ".join(str(item) for item in result["evidence_ids"]) or "None"
        lines.extend(
            [
                f"### {result['criterion_type'].title()}: {result['criterion_text']}",
                "",
                f"- **Label:** {result['label']}",
                f"- **Evidence sentence IDs:** {evidence}",
                f"- **Explanation:** {result['explanation']}",
                f"- **Parse successful:** {result['parse_ok']}",
                f"- **Fallback used:** {result['fallback_used']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            "```json",
            json.dumps(payload["provenance"], indent=2),
            "```",
            "",
        ]
    )
    lines.extend(["## Disclaimer", "", DISCLAIMER, ""])
    return "\n".join(lines)


def export_screening_report(
    *,
    patient_note: str,
    criteria: Sequence[CriterionResult],
    aggregation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_dir: Path | None = None,
) -> ReportExports:
    """Create three reports without embedding the original patient note."""
    criterion_inputs = [
        {
            "criterion_type": result.criterion_type,
            "criterion_text": result.criterion_text,
        }
        for result in criteria
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient_note_sha256": _sha256(patient_note),
        "criteria_sha256": _sha256(
            json.dumps(criterion_inputs, ensure_ascii=False, sort_keys=True)
        ),
        "aggregation": dict(aggregation),
        "criteria": [asdict(result) for result in criteria],
        "provenance": dict(provenance),
        "disclaimer": DISCLAIMER,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown_text = _markdown(payload)
    template = Environment(
        autoescape=select_autoescape(("html", "xml")),
    ).from_string(
        files("criterion_lens.report")
        .joinpath("templates", "report.html.j2")
        .read_text(encoding="utf-8")
    )
    html_text = template.render(report=payload)

    saved_files: dict[str, Path] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension, content in (
            ("json", json_text),
            ("md", markdown_text),
            ("html", html_text),
        ):
            path = output_dir / f"screening_report.{extension}"
            path.write_text(content, encoding="utf-8")
            saved_files[extension] = path
    return ReportExports(json_text, markdown_text, html_text, saved_files)

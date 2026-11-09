import hashlib
import json
from pathlib import Path

from criterion_lens.inference.schema import CriterionResult
from criterion_lens.report.export import DISCLAIMER, export_screening_report


def _result() -> CriterionResult:
    return CriterionResult(
        criterion_id="c1",
        criterion_type="inclusion",
        criterion_text="<script>alert('x')</script>",
        label="included",
        evidence_ids=[0],
        evidence_text=["Age is 42."],
        explanation="The cited sentence supports the criterion.",
        missing_information=False,
        unsupported_decision=False,
        invalid_ids_dropped=[],
        raw_output='{"evidence":[0],"label":"included"}',
        parse_ok=True,
        invalid_label=False,
        fallback_used=False,
        decoding={"total_s": 0.1},
    )


def test_export_screening_report_writes_provenance_json_markdown_and_html(
    tmp_path: Path,
) -> None:
    result = export_screening_report(
        patient_note="Private patient note.",
        criteria=[_result()],
        aggregation={
            "verdict": "potentially eligible (preliminary)",
            "interpretation": "Not a ground-truth decision.",
        },
        provenance={"model_id": "test/model", "decoding": "greedy_kv"},
        output_dir=tmp_path,
    )

    payload = json.loads(result.json_text)
    assert payload["patient_note_sha256"] == hashlib.sha256(
        b"Private patient note."
    ).hexdigest()
    assert payload["provenance"]["model_id"] == "test/model"
    assert "Private patient note" not in result.json_text
    assert DISCLAIMER in result.markdown_text
    assert "Private patient note" not in result.markdown_text
    assert "&lt;script&gt;" in result.html_text
    assert "<script>alert" not in result.html_text
    assert DISCLAIMER in result.html_text
    assert {key: path.name for key, path in result.saved_files.items()} == {
        "json": "screening_report.json",
        "md": "screening_report.md",
        "html": "screening_report.html",
    }
    assert all(path.is_file() for path in result.saved_files.values())

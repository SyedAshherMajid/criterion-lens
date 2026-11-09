import pytest

from criterion_lens.inference.parse import parse_output
from criterion_lens.inference.schema import CriterionResult, ParsedOutput


def test_parse_output_reads_json_and_canonicalizes_evidence() -> None:
    output = '```json\n{"evidence": [2, 0, 2], "label": "Included"}\n```'

    parsed = parse_output(output, "inclusion", n_sentences=3)

    assert parsed == ParsedOutput(
        label="included",
        evidence_ids=[0, 2],
        invalid_ids_dropped=[],
        raw_output=output,
        parse_ok=True,
        invalid_label=False,
    )


def test_parse_output_uses_regex_for_malformed_json() -> None:
    output = '{"label": "not excluded", "evidence": [1, 3]'

    parsed = parse_output(output, "exclusion", n_sentences=4)

    assert parsed.label == "not excluded"
    assert parsed.evidence_ids == [1, 3]
    assert parsed.parse_ok


def test_parse_output_drops_and_reports_out_of_range_ids() -> None:
    output = '{"evidence": [-1, 0, 4], "label": "included"}'

    parsed = parse_output(output, "inclusion", n_sentences=2)

    assert parsed.label == "included"
    assert parsed.evidence_ids == [0]
    assert parsed.invalid_ids_dropped == [-1, 4]
    assert not parsed.parse_ok


def test_parse_output_marks_explicit_invalid_labels() -> None:
    parsed = parse_output(
        '{"evidence": [], "label": "made up"}',
        "inclusion",
        n_sentences=1,
    )

    assert parsed.label is None
    assert parsed.invalid_label


@pytest.mark.parametrize(
    ("output", "expected_label"),
    [
        ("not JSON", None),
        ('{"evidence": [], "label": "excluded"}', None),
        ('{"evidence": "0", "label": "included"}', "included"),
        ('{"evidence": [true], "label": "included"}', "included"),
        ('{"evidence": [], "label": "made up"}', None),
    ],
)
def test_parse_output_marks_malformed_or_type_incompatible_outputs(
    output: str, expected_label: str | None
) -> None:
    parsed = parse_output(output, "inclusion", n_sentences=3)

    assert parsed.label == expected_label
    assert not parsed.parse_ok


def test_parse_output_rejects_invalid_parser_arguments() -> None:
    with pytest.raises(ValueError, match="Unsupported criterion_type"):
        parse_output('{"evidence": [], "label": "included"}', "other", 3)
    with pytest.raises(ValueError, match="non-negative"):
        parse_output('{"evidence": [], "label": "included"}', "inclusion", -1)
    with pytest.raises(ValueError, match="non-negative"):
        parse_output('{"evidence": [], "label": "included"}', "inclusion", True)


def test_criterion_result_schema_accepts_complete_inference_record() -> None:
    result = CriterionResult(
        criterion_id="inc-3",
        criterion_type="inclusion",
        criterion_text="Age over 18",
        label="included",
        evidence_ids=[0],
        evidence_text=["The patient is 58."],
        explanation="Based on sentence 0.",
        missing_information=False,
        unsupported_decision=False,
        invalid_ids_dropped=[],
        raw_output='{"evidence": [0], "label": "included"}',
        parse_ok=True,
        invalid_label=False,
        fallback_used=False,
        decoding={"mode": "greedy", "max_new_tokens": 64},
    )

    assert result.evidence_ids == [0]

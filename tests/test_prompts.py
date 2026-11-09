import json

import pytest

from criterion_lens.data.prompts import ALLOWED_LABELS, build_messages, format_target


def test_allowed_labels_keep_inclusion_and_exclusion_polarities() -> None:
    assert ALLOWED_LABELS["inclusion"] == (
        "included",
        "not included",
        "not enough information",
        "not applicable",
    )
    assert ALLOWED_LABELS["exclusion"] == (
        "excluded",
        "not excluded",
        "not enough information",
        "not applicable",
    )


def test_build_messages_contains_only_the_permitted_inputs() -> None:
    messages = build_messages(
        [(0, "Age is 58."), (1, "No prior surgery.")], "inclusion", "Age > 18"
    )
    combined = "\n".join(message["content"] for message in messages)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Age is 58." in combined
    assert "Criterion type: inclusion" in combined
    assert "Age > 18" in combined
    assert "gpt4_explanation" not in combined


def test_format_target_sorts_and_deduplicates_evidence_ids() -> None:
    target = format_target([2, 0, 2], "included")

    assert json.loads(target) == {"evidence": [0, 2], "label": "included"}


def test_prompt_rejects_unknown_criterion_types_and_labels() -> None:
    with pytest.raises(ValueError, match="Unsupported criterion_type"):
        build_messages([], "other", "criterion")
    with pytest.raises(ValueError, match="Unsupported label"):
        format_target([], "gpt4_eligibility")

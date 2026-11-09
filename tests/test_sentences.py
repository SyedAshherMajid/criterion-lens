import pytest

from criterion_lens.data.sentences import parse_numbered_note, split_sentences


def test_parse_numbered_note_preserves_ids_and_decimal_values() -> None:
    note = "0. Takes 3.5 mg daily. 1. Follow-up with Dr. Smith was unremarkable."

    assert parse_numbered_note(note) == [
        (0, "Takes 3.5 mg daily."),
        (1, "Follow-up with Dr. Smith was unremarkable."),
    ]


def test_parse_numbered_note_rejects_noncontiguous_ids() -> None:
    with pytest.raises(ValueError, match="contiguous from zero"):
        parse_numbered_note("0. First sentence. 2. Skipped sentence.")


def test_parse_numbered_note_rejects_unexpected_prefix() -> None:
    with pytest.raises(ValueError, match="before the first"):
        parse_numbered_note("Header 0. First sentence.")


def test_split_sentences_uses_pysbd_for_unseen_free_text() -> None:
    sentences = split_sentences("Dr. Smith prescribed 3.5 mg daily. The patient is stable.")

    assert sentences == [
        (0, "Dr. Smith prescribed 3.5 mg daily."),
        (1, "The patient is stable."),
    ]


def test_split_sentences_uses_numbered_input_and_handles_empty_text() -> None:
    assert split_sentences("0. First sentence. 1. Second sentence.") == [
        (0, "First sentence."),
        (1, "Second sentence."),
    ]
    assert split_sentences(" \n\t ") == []

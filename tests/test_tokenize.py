import pytest

from criterion_lens.data.prompts import build_messages, format_target
from criterion_lens.data.sentences import parse_numbered_note
from criterion_lens.data.tokenize import encode_example


class TokenizerStub:
    unk_token_id = 0
    end_token_id = 10_000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        assert add_generation_prompt
        prompt_length = len(messages[1]["content"].split()) + 8
        return list(range(1, prompt_length + 1))

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return self.end_token_id

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            "<|im_end|>" if token_id == self.end_token_id else chr(token_id)
            for token_id in token_ids
        )


def _example() -> dict[str, object]:
    return {
        "annotation_id": 17,
        "note": "0. Age is 58. 1. Diagnosis documented. 2. No other disease.",
        "criterion_type": "inclusion",
        "criterion_text": "Age over 18",
        "expert_sentences": [0],
        "expert_eligibility": "included",
    }


def test_encode_example_masks_prompt_and_keeps_exact_target_and_end_token() -> None:
    tokenizer = TokenizerStub()
    example = _example()

    encoded = encode_example(tokenizer, example, max_len=1000)
    prompt_ids = tokenizer.apply_chat_template(
        build_messages(
            parse_numbered_note(str(example["note"])),
            str(example["criterion_type"]),
            str(example["criterion_text"]),
        ),
        tokenize=True,
        add_generation_prompt=True,
    )
    target = format_target([0], "included")
    response_labels = [
        token_id for token_id in encoded["labels"] if token_id != -100
    ]

    assert encoded["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert tokenizer.decode(response_labels) == target + "<|im_end|>"
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


def test_encode_example_truncates_only_trailing_non_evidence_sentences() -> None:
    tokenizer = TokenizerStub()
    example = _example()
    full_length = len(encode_example(tokenizer, example, max_len=1000)["input_ids"])
    truncation_log: list[dict[str, object]] = []

    encoded = encode_example(
        tokenizer,
        example,
        max_len=full_length - 1,
        truncation_log=truncation_log,
    )

    assert len(encoded["input_ids"]) <= full_length - 1
    assert truncation_log == [
        {
            "annotation_id": 17,
            "removed_sentence_ids": [2],
            "removed_gold_evidence": False,
        }
    ]


def test_encode_example_refuses_to_truncate_gold_evidence() -> None:
    tokenizer = TokenizerStub()
    example = _example()
    example["expert_sentences"] = [0, 1, 2]
    full_length = len(encode_example(tokenizer, example, max_len=1000)["input_ids"])

    with pytest.raises(ValueError, match="truncation would remove gold evidence"):
        encode_example(tokenizer, example, max_len=full_length - 1)


def test_encode_example_rejects_missing_gold_evidence_ids() -> None:
    example = _example()
    example["expert_sentences"] = [99]

    with pytest.raises(ValueError, match="references missing evidence sentence IDs"):
        encode_example(TokenizerStub(), example, max_len=1000)

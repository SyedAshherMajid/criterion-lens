import os

import pytest

from criterion_lens.decoding.speculative import validate_tokenizer_compatibility


class _Tokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        *,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        self._vocab = vocab
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = None
        self.unk_token_id = 0
        self.additional_special_tokens_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_tokenizer_compatibility_checks_vocab_and_special_ids() -> None:
    tokenizer = _Tokenizer({"<unk>": 0, "<s>": 1, "</s>": 2})
    compatible = _Tokenizer({"<unk>": 0, "<s>": 1, "</s>": 2})

    assert validate_tokenizer_compatibility(tokenizer, compatible) == 3
    with pytest.raises(ValueError, match="token-to-ID"):
        validate_tokenizer_compatibility(
            tokenizer,
            _Tokenizer({"<unk>": 0, "<s>": 2, "</s>": 1}),
        )
    with pytest.raises(ValueError, match="special-token"):
        validate_tokenizer_compatibility(
            tokenizer,
            _Tokenizer({"<unk>": 0, "<s>": 1, "</s>": 2}, eos_token_id=1),
        )


@pytest.mark.slow
def test_real_qwen_tokenizers_are_compatible() -> None:
    if os.environ.get("CRITERION_LENS_RUN_SLOW_TOKENIZER_TESTS") != "1":
        pytest.skip("Set CRITERION_LENS_RUN_SLOW_TOKENIZER_TESTS=1 to load real tokenizers")
    from transformers import AutoTokenizer

    target = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
    draft = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
    vocab_size = validate_tokenizer_compatibility(target, draft)

    assert vocab_size == len(target)

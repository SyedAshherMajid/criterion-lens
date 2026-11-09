from types import SimpleNamespace

import pytest
import torch

from criterion_lens.inference.predict import predict_criterion


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 0
    generated_text = ""
    label_token_ids = {
        "included": 3,
        "not included": 4,
        "not enough information": 5,
        "not applicable": 6,
    }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert messages[1]["content"].startswith("Patient note")
        assert tokenize and add_generation_prompt
        return torch.tensor([[1, 2]])

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return self.eos_token_id

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return self.generated_text

    def encode(self, text: str, *, add_special_tokens):
        assert not add_special_tokens
        for label, token_id in self.label_token_ids.items():
            if f'"label": "{label}"' in text:
                return [token_id]
        raise AssertionError(f"Unknown continuation: {text}")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.preferred_label_token = 4
        self.forward_batch_sizes: list[int] = []

    def generate(self, input_ids, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is False
        return torch.cat(
            [input_ids, torch.tensor([[9]], device=input_ids.device)],
            dim=1,
        )

    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        self.forward_batch_sizes.append(input_ids.shape[0])
        logits = torch.full(
            (*input_ids.shape, 10),
            -5.0,
            device=input_ids.device,
        )
        logits[:, 1, self.preferred_label_token] = 2.0
        return SimpleNamespace(logits=logits)


def _predict(tokenizer: _Tokenizer, model: _Model):
    return predict_criterion(
        model,
        tokenizer,
        "0. The patient is 58. 1. The patient reports no fever.",
        "inc-1",
        "inclusion",
        "Age over 18",
    )


def test_predict_criterion_returns_exact_cited_evidence() -> None:
    tokenizer = _Tokenizer()
    tokenizer.generated_text = '{"evidence": [0], "label": "included"}'
    model = _Model()

    result = _predict(tokenizer, model)

    assert result.label == "included"
    assert result.evidence_ids == [0]
    assert result.evidence_text == ["The patient is 58."]
    assert result.parse_ok
    assert not result.fallback_used
    assert not result.unsupported_decision
    assert "The patient is 58." in result.explanation
    assert result.decoding == {
        "mode": "no_cache",
        "max_new_tokens": 64,
        "do_sample": False,
    }


def test_predict_criterion_scores_fallback_labels_in_one_batch() -> None:
    tokenizer = _Tokenizer()
    tokenizer.generated_text = "malformed output"
    model = _Model()

    result = _predict(tokenizer, model)

    assert result.label == "not included"
    assert result.fallback_used
    assert not result.parse_ok
    assert result.unsupported_decision
    assert "[UNSUPPORTED]" in result.explanation
    assert model.forward_batch_sizes == [4]


def test_predict_criterion_preserves_invalid_ids_and_uses_fallback() -> None:
    tokenizer = _Tokenizer()
    tokenizer.generated_text = '{"evidence": [9], "label": "included"}'
    model = _Model()

    result = _predict(tokenizer, model)

    assert result.label == "not included"
    assert result.evidence_ids == []
    assert result.invalid_ids_dropped == [9]
    assert result.fallback_used


def test_predict_criterion_accepts_a_custom_decoder() -> None:
    tokenizer = _Tokenizer()
    tokenizer.generated_text = '{"evidence": [], "label": "not enough information"}'
    model = _Model()

    result = predict_criterion(
        model,
        tokenizer,
        "0. No patient age is listed.",
        "inc-1",
        "inclusion",
        "Age over 18",
        decoding="kv_cache",
        decoder=lambda *args, **kwargs: SimpleNamespace(
            tokens=torch.tensor([9]),
            prefill_s=0.1,
            decode_s=0.2,
        ),
    )

    assert result.label == "not enough information"
    assert result.decoding["mode"] == "kv_cache"
    assert result.decoding["prefill_s"] == 0.1
    assert result.decoding["decode_s"] == 0.2


def test_predict_criterion_rejects_invalid_arguments() -> None:
    tokenizer = _Tokenizer()
    model = _Model()
    with pytest.raises(ValueError, match="Unsupported criterion_type"):
        predict_criterion(model, tokenizer, "note", "inc-1", "other", "criterion")
    with pytest.raises(ValueError, match="positive"):
        predict_criterion(
            model,
            tokenizer,
            "note",
            "inc-1",
            "inclusion",
            "criterion",
            max_new_tokens=0,
        )
    with pytest.raises(ValueError, match="decoder callback"):
        predict_criterion(
            model,
            tokenizer,
            "note",
            "inc-1",
            "inclusion",
            "criterion",
            decoding="speculative",
        )

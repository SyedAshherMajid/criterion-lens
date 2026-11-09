import json
import sys
from pathlib import Path

import pytest
import torch

from scripts import evaluate


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 0
    generated_text = '{"evidence": [0], "label": "included"}'

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        return torch.tensor([[1, 2]])

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return self.eos_token_id

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return self.generated_text


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[9]])], dim=1)


def _record() -> dict:
    return {
        "annotation_id": 101,
        "patient_id": "patient-1",
        "trial_id": "NCT00000001",
        "note": "0. The patient is 58.",
        "criterion_type": "inclusion",
        "criterion_text": "Age over 18",
        "expert_eligibility": "included",
        "expert_sentences": [0],
    }


def test_evaluate_records_writes_predictions_metrics_and_confusion_figures(
    tmp_path: Path,
) -> None:
    metrics = evaluate.evaluate_records(
        [_record()],
        _Model(),
        _Tokenizer(),
        split="val",
        model_id="test/model",
        adapter_id=None,
        decoding="no_cache",
        max_new_tokens=64,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    prediction = json.loads(
        (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert prediction["gold"] == {"label": "included", "evidence_ids": [0]}
    assert prediction["evidence_text"] == ["The patient is 58."]
    assert saved_metrics["classification"]["overall"]["accuracy"] == 1
    assert saved_metrics["evidence"]["overall"]["micro"]["f1"] == 1
    assert metrics["n_annotations"] == 1
    assert len(list(tmp_path.glob("confusion_*.png"))) == 6


def test_test_access_requires_final_flag_and_frozen_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_log = tmp_path / "results" / "TEST_ACCESS_LOG.md"
    monkeypatch.setattr(evaluate, "TEST_ACCESS_LOG", access_log)
    monkeypatch.setattr(
        evaluate,
        "_git_output",
        lambda arguments: "v1.0-frozen" if "describe" in arguments else "abc123",
    )

    with pytest.raises(ValueError, match="requires --final"):
        evaluate.record_test_access(
            split="test",
            final=False,
            model="test/model",
            adapter=None,
            decoding_config={"mode": "no_cache"},
        )

    evaluate.record_test_access(
        split="test",
        final=True,
        model="test/model",
        adapter="test/adapter",
        decoding_config={"mode": "greedy_kv", "max_new_tokens": 64},
    )
    log = access_log.read_text(encoding="utf-8")
    assert "abc123" in log
    assert "test/model" in log
    assert "test/adapter" in log
    with pytest.raises(RuntimeError, match="already been accessed"):
        evaluate.record_test_access(
            split="test",
            final=True,
            model="test/model",
            adapter=None,
            decoding_config={"mode": "no_cache"},
        )


def test_test_access_rejects_nonfrozen_head_without_creating_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_log = tmp_path / "results" / "TEST_ACCESS_LOG.md"
    monkeypatch.setattr(evaluate, "TEST_ACCESS_LOG", access_log)
    monkeypatch.setattr(evaluate, "_git_output", lambda arguments: "main")

    with pytest.raises(RuntimeError, match="v1.0-frozen"):
        evaluate.record_test_access(
            split="test",
            final=True,
            model="test/model",
            adapter=None,
            decoding_config={"mode": "no_cache"},
        )
    assert not access_log.exists()


def test_cli_defaults_to_safe_greedy_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--model", "test/model", "--split", "val", "--out", "out"],
    )

    args = evaluate.parse_args()

    assert args.decoding == "no_cache"
    assert args.max_new_tokens == 64
    assert not args.final

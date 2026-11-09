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
    frozen_commit = "a" * 40
    monkeypatch.setattr(
        evaluate,
        "_git_output",
        lambda arguments: "v1.0-frozen" if "describe" in arguments else frozen_commit,
    )

    with pytest.raises(ValueError, match="requires --final"):
        evaluate.record_test_access(
            split="test",
            final=False,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter=None,
            decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
        )

    evaluate.record_test_access(
        split="test",
        final=True,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        adapter=None,
        decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
    )
    evaluate.record_test_access(
        split="test",
        final=True,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        adapter="Ashher123/criterion-lens-qwen2.5-1.5b-lora",
        decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
    )
    log = access_log.read_text(encoding="utf-8")
    assert frozen_commit in log
    assert "Qwen/Qwen2.5-1.5B-Instruct" in log
    assert "criterion-lens-qwen2.5-1.5b-lora" in log
    with pytest.raises(RuntimeError, match="already been accessed"):
        evaluate.record_test_access(
            split="test",
            final=True,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter=None,
            decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
        )


def test_test_access_in_kaggle_uses_validated_freeze_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_log = tmp_path / "results" / "TEST_ACCESS_LOG.md"
    freeze_manifest = tmp_path / "freeze_manifest.json"
    commit = "b" * 40
    freeze_manifest.write_text(
        json.dumps({"tag": "v1.0-frozen", "commit": commit}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate, "TEST_ACCESS_LOG", access_log)
    monkeypatch.setattr(evaluate, "FREEZE_MANIFEST", freeze_manifest)
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)

    def missing_git(_arguments: list[str]) -> str:
        raise RuntimeError("not a Git checkout")

    monkeypatch.setattr(evaluate, "_git_output", missing_git)
    evaluate.record_test_access(
        split="test",
        final=True,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter=None,
        decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
    )

    assert commit in access_log.read_text(encoding="utf-8")


def test_test_access_rejects_unplanned_model_or_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_log = tmp_path / "results" / "TEST_ACCESS_LOG.md"
    monkeypatch.setattr(evaluate, "TEST_ACCESS_LOG", access_log)
    monkeypatch.setattr(
        evaluate,
        "_git_output",
        lambda arguments: "v1.0-frozen" if "describe" in arguments else "c" * 40,
    )

    with pytest.raises(ValueError, match="four frozen model variants"):
        evaluate.record_test_access(
            split="test",
            final=True,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter="Ashher123/unplanned-adapter",
            decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
        )
    with pytest.raises(ValueError, match="greedy no-cache"):
        evaluate.record_test_access(
            split="test",
            final=True,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter=None,
            decoding_config={"mode": "greedy_kv", "max_new_tokens": 64, "do_sample": False},
        )
    assert not access_log.exists()


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
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter=None,
            decoding_config=evaluate.FINAL_TEST_DECODING_CONFIG,
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

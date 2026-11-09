import csv
import json

import pandas as pd
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.bench_speculative import (
    _write_results,
    load_validation_prompts,
    run_benchmark,
    synthetic_prompts,
)


def _tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(55)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()


def test_synthetic_speculative_benchmark_writes_results(tmp_path) -> None:
    target = _tiny_model()
    draft = _tiny_model()
    prompts = synthetic_prompts(count=1, length=8, vocab_size=97)

    result = run_benchmark(
        target,
        draft,
        prompts,
        target_label="tiny",
        draft_label="tiny",
        output_lengths=[4],
        k_values=[2],
        eos_id=None,
        vocab_size=97,
        input_source="synthetic",
        warmup=0,
        repeats=1,
    )
    _write_results(result, tmp_path)

    payload = json.loads((tmp_path / "speculative.json").read_text(encoding="utf-8"))
    with (tmp_path / "speculative.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert payload["input_source"] == "synthetic"
    assert payload["results"][0]["sequence_equality_rate"] == 1.0
    assert payload["results"][0]["acceptance_rate"] == 1.0
    assert rows[0]["k"] == "2"
    assert float(rows[0]["speedup"]) > 0


def test_validation_prompt_loader_reads_only_val_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.bench_speculative as benchmark

    revision = "test-revision"
    manifest_path = tmp_path / "manifest.json"
    audit_path = tmp_path / "audit.json"
    annotations_path = tmp_path / "annotations.parquet"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_revision": revision,
                "val": {"patient_ids": ["val-patient"], "n_annotations": 1},
            }
        ),
        encoding="utf-8",
    )
    audit_path.write_text(json.dumps({"dataset_revision": revision}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "annotation_id": "val-1",
                "patient_id": "val-patient",
                "note": "0. A validation-only note.",
                "criterion_type": "inclusion",
                "criterion_text": "The patient is an adult.",
                "gpt4_explanation": "must not be loaded",
            },
            {
                "annotation_id": "test-1",
                "patient_id": "test-patient",
                "note": "0. A sealed test note.",
                "criterion_type": "inclusion",
                "criterion_text": "The patient is an adult.",
                "gpt4_explanation": "must not be loaded",
            },
        ]
    ).to_parquet(annotations_path, index=False)
    monkeypatch.setattr(benchmark, "SPLIT_MANIFEST", manifest_path)
    monkeypatch.setattr(benchmark, "DATA_AUDIT", audit_path)
    monkeypatch.setattr(benchmark, "ANNOTATIONS_PATH", annotations_path)

    class Tokenizer:
        chat_template = "present"

        def apply_chat_template(self, messages, **kwargs):
            assert "A validation-only note." in messages[1]["content"]
            assert "sealed test note" not in messages[1]["content"]
            return torch.tensor([[3, 4, 5]])

    prompts = load_validation_prompts(Tokenizer(), count=1)

    assert prompts[0].tolist() == [[3, 4, 5]]

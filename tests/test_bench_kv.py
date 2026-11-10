import csv
import json

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.bench_kv import _write_results, benchmark_model


def test_tiny_benchmark_writes_structured_json_and_csv(tmp_path) -> None:
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    result = benchmark_model(
        model,
        model_label="tiny",
        model_id="tiny-random-qwen2",
        device=torch.device("cpu"),
        dtype_name="float32",
        input_lengths=[8],
        output_lengths=[2],
        warmup=0,
        repeats=1,
    )
    _write_results(result, tmp_path)

    payload = json.loads((tmp_path / "kv_cache.json").read_text(encoding="utf-8"))
    with (tmp_path / "kv_cache.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert payload["input_source"] == "deterministic_synthetic_token_ids"
    assert payload["precision_checks"] == [
        {
            "precision": "float32",
            "input_length": 8,
            "output_length": 2,
            "tokens_equal": True,
        }
    ]
    assert rows[0]["input_length"] == "8"
    assert rows[0]["output_length"] == "2"
    assert json.loads(rows[0]["kv_cache"])["peak_memory_bytes"] is None
    assert json.loads(rows[0]["cache_growth"]) == [
        {"cache_bytes": 2048, "cache_len": 8},
        {"cache_bytes": 2304, "cache_len": 9},
    ]

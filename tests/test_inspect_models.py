from types import SimpleNamespace

from scripts.inspect_models import config_fields


def test_config_fields_extract_architecture_and_infer_head_dim() -> None:
    config = SimpleNamespace(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=None,
        hidden_size=512,
        intermediate_size=1024,
        vocab_size=128,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )

    spec = config_fields("example/model", config)

    assert spec["layers"] == 2
    assert spec["query_heads"] == 8
    assert spec["kv_heads"] == 2
    assert spec["head_dim"] == 64
    assert spec["vocab_size"] == 128
    assert spec["context_length"] == 2048
    assert spec["parameters"] is None

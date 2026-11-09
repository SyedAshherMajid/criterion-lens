import json

import pytest

from scripts import merge_adapter


class _MergedModel:
    def save_pretrained(self, output_dir, *, safe_serialization):
        assert safe_serialization is True
        (output_dir / "model.safetensors").write_bytes(b"merged")


class _PeftModel:
    def merge_and_unload(self, *, safe_merge):
        assert safe_merge is True
        return _MergedModel()


class _Tokenizer:
    def save_pretrained(self, output_dir):
        (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_merge_run_saves_merged_model_tokenizer_and_safe_card(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "adapter"
    run_dir.mkdir()
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "torch_dtype_used": "float16",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "Qwen/Qwen2.5-0.5B-Instruct"
            assert kwargs["device_map"] == "cpu"
            return object()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, use_fast):
            assert path == str(run_dir)
            assert use_fast is True
            return _Tokenizer()

    class _AdapterLoader:
        @staticmethod
        def from_pretrained(base_model, path, *, is_trainable):
            assert path == str(run_dir)
            assert is_trainable is False
            return _PeftModel()

    monkeypatch.setattr(merge_adapter, "AutoModelForCausalLM", _AutoModel)
    monkeypatch.setattr(merge_adapter, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr(merge_adapter, "PeftModel", _AdapterLoader)
    output_dir = tmp_path / "merged"

    assert merge_adapter.merge_run(run_dir, output_dir) == output_dir
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "tokenizer.json").is_file()
    card = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "training partition only" in card
    assert "No test-set metrics are included" in card


def test_merge_run_refuses_to_overwrite_existing_output(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        merge_adapter.merge_run(tmp_path / "missing", output_dir)

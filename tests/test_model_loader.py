from types import SimpleNamespace

import pytest
import torch

from criterion_lens.modeling import load_model as loader


@pytest.fixture(autouse=True)
def clear_model_cache():
    loader.clear_model_cache()
    yield
    loader.clear_model_cache()


class FakeModel:
    def __init__(self) -> None:
        self.device = None
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self


def test_load_model_uses_environment_and_caches_model(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_env", lambda: 0)
    monkeypatch.setenv("CL_MODEL", "base-model")
    monkeypatch.setenv("CL_TOKENIZER", "tokenizer")
    monkeypatch.setenv("CL_DEVICE", "cpu")
    monkeypatch.setenv("CL_DTYPE", "float32")
    calls = {"model": 0, "tokenizer": 0}
    model = FakeModel()
    tokenizer = object()

    def load_model(model_id, *, dtype):
        calls["model"] += 1
        assert model_id == "base-model"
        assert dtype is torch.float32
        return model

    def load_tokenizer(tokenizer_id, *, use_fast):
        calls["tokenizer"] += 1
        assert tokenizer_id == "tokenizer"
        assert use_fast
        return tokenizer

    monkeypatch.setattr(
        loader,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setattr(
        loader,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=load_tokenizer),
    )

    loaded = loader.load_model()

    assert loader.load_model() is loaded
    assert loaded.model is model
    assert loaded.tokenizer is tokenizer
    assert loaded.device == torch.device("cpu")
    assert loaded.dtype is torch.float32
    assert not model.training
    assert calls == {"model": 1, "tokenizer": 1}


def test_load_model_applies_optional_adapter(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_env", lambda: 0)
    base = FakeModel()
    adapted = FakeModel()
    adapter_calls = []
    monkeypatch.setattr(
        loader,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: base),
    )
    monkeypatch.setattr(
        loader,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        loader,
        "PeftModel",
        SimpleNamespace(
            from_pretrained=lambda model, adapter_id: (
                adapter_calls.append((model, adapter_id)) or adapted
            )
        ),
    )

    loaded = loader.load_model(
        "base-model",
        adapter_id="adapter",
        device="cpu",
        dtype="float32",
    )

    assert adapter_calls == [(base, "adapter")]
    assert loaded.model is adapted


def test_load_model_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_env", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        loader.load_model("base-model", device="cuda", dtype="float32")

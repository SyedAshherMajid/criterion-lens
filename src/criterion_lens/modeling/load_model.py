"""Load and cache the configured language model and tokenizer."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from criterion_lens.utils.env import load_env

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Any
    model_id: str
    device: torch.device
    dtype: torch.dtype


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported inference device: {device}")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    try:
        return _DTYPES[value]
    except KeyError as error:
        raise ValueError(
            f"Unsupported inference dtype {value!r}; choose auto, float16, bfloat16, or float32"
        ) from error


@lru_cache(maxsize=4)
def _load_cached(
    model_id: str,
    tokenizer_id: str,
    adapter_id: str | None,
    device_name: str,
    dtype_name: str,
) -> LoadedModel:
    device = _resolve_device(device_name)
    dtype = _resolve_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)
    model = model.to(device).eval()
    return LoadedModel(model, tokenizer, model_id, device, dtype)


def load_model(
    model_id: str | None = None,
    *,
    adapter_id: str | None = None,
    tokenizer_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> LoadedModel:
    """Load the selected model once, using environment defaults when omitted."""
    load_env()
    resolved_model_id = model_id or os.environ.get("CL_MODEL")
    if not resolved_model_id:
        raise RuntimeError("Set CL_MODEL or pass model_id to load_model")
    resolved_tokenizer_id = tokenizer_id or os.environ.get(
        "CL_TOKENIZER",
        resolved_model_id,
    )
    return _load_cached(
        resolved_model_id,
        resolved_tokenizer_id,
        adapter_id,
        device or os.environ.get("CL_DEVICE", "auto"),
        dtype or os.environ.get("CL_DTYPE", "auto"),
    )


def load_draft_model(
    model_id: str | None = None,
    *,
    tokenizer_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> LoadedModel:
    """Load the configured draft model for speculative decoding."""
    load_env()
    resolved_model_id = model_id or os.environ.get("CL_DRAFT_MODEL")
    if not resolved_model_id:
        raise RuntimeError("Set CL_DRAFT_MODEL or pass model_id to load_draft_model")
    return load_model(
        resolved_model_id,
        tokenizer_id=tokenizer_id or resolved_model_id,
        device=device,
        dtype=dtype,
    )


def clear_model_cache() -> None:
    """Release cached model references, primarily for model switching and tests."""
    _load_cached.cache_clear()

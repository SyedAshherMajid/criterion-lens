"""Compatibility helpers for measuring and cropping Hugging Face KV caches."""

from typing import Any

import torch


def _cache_layers(cache: Any) -> list[Any] | None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return list(layers)
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache, strict=True))
    return None


def _layer_tensors(layer: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(layer, tuple) and len(layer) == 2:
        keys, values = layer
    else:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
    if keys is not None and not isinstance(keys, torch.Tensor):
        raise TypeError("Cache keys must be torch.Tensor values")
    if values is not None and not isinstance(values, torch.Tensor):
        raise TypeError("Cache values must be torch.Tensor values")
    return keys, values


def cache_seq_len(cache: Any) -> int:
    """Return the sequence length from the installed Cache API or cache tensors."""
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    layers = _cache_layers(cache)
    if layers is None:
        raise TypeError("Unsupported cache: no get_seq_length, layers, or key_cache")
    for layer in layers:
        keys, _ = _layer_tensors(layer)
        if keys is not None:
            return int(keys.shape[-2])
    return 0


def cache_bytes(cache: Any) -> int:
    """Count allocated key/value tensor bytes across cache layers."""
    layers = _cache_layers(cache)
    if layers is None:
        raise TypeError("Unsupported cache: expected layers or key_cache/value_cache")
    total = 0
    for layer in layers:
        keys, values = _layer_tensors(layer)
        for tensor in (keys, values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def crop_cache(cache: Any, max_length: int) -> Any:
    """Crop a mutable HF cache in place and return the same cache object."""
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    if cache_seq_len(cache) <= max_length:
        return cache

    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(max_length)
        return cache

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is None or value_cache is None:
        raise TypeError("Cache does not support crop and has no legacy key/value lists")
    for index, (keys, values) in enumerate(zip(key_cache, value_cache, strict=True)):
        if keys is not None:
            key_cache[index] = keys[..., :max_length, :]
        if values is not None:
            value_cache[index] = values[..., :max_length, :]
    return cache

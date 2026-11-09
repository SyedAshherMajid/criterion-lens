import pytest
import torch
from transformers.cache_utils import DynamicCache

from criterion_lens.decoding.cache_utils import cache_bytes, cache_seq_len, crop_cache


def _filled_cache() -> DynamicCache:
    cache = DynamicCache()
    for layer_index in range(2):
        key = torch.zeros(2, 3, 4, 5, dtype=torch.float16)
        value = torch.zeros_like(key)
        cache.update(key, value, layer_index)
    return cache


def test_dynamic_cache_sequence_length_and_byte_formula() -> None:
    cache = _filled_cache()

    assert cache_seq_len(cache) == 4
    expected_bytes = 2 * 2 * 2 * 3 * 4 * 5 * torch.tensor([], dtype=torch.float16).element_size()
    assert cache_bytes(cache) == expected_bytes


def test_crop_cache_uses_transformers_api_and_mutates_in_place() -> None:
    cache = _filled_cache()

    cropped = crop_cache(cache, 2)

    assert cropped is cache
    assert cache_seq_len(cache) == 2
    assert cache_bytes(cache) == 2 * 2 * 2 * 3 * 2 * 5 * 2
    assert crop_cache(cache, 10) is cache


def test_empty_dynamic_cache_has_zero_length_and_bytes() -> None:
    cache = DynamicCache()

    assert cache_seq_len(cache) == 0
    assert cache_bytes(cache) == 0


def test_legacy_key_value_lists_are_supported() -> None:
    class LegacyCache:
        def __init__(self) -> None:
            self.key_cache = [torch.zeros(1, 2, 5, 4)]
            self.value_cache = [torch.zeros(1, 2, 5, 4)]

    cache = LegacyCache()

    assert cache_seq_len(cache) == 5
    assert cache_bytes(cache) == 2 * 1 * 2 * 5 * 4 * 4
    crop_cache(cache, 3)
    assert cache_seq_len(cache) == 3


def test_cache_helpers_reject_invalid_or_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        crop_cache(_filled_cache(), -1)
    with pytest.raises(TypeError, match="Unsupported cache"):
        cache_bytes(object())
    with pytest.raises(TypeError, match="Unsupported cache"):
        cache_seq_len(object())

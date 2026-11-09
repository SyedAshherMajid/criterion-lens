import pytest
import torch

from criterion_lens.attention.reference import sdpa_reference
from criterion_lens.attention.tiled import tiled_attention

SHAPES = [
    (1, 1, 8, 8, 4),
    (2, 4, 64, 64, 32),
    (1, 2, 1, 77, 64),
    (3, 8, 129, 129, 64),
    (1, 12, 512, 512, 128),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_tiled_attention_matches_fp64_reference(
    shape: tuple[int, int, int, int, int],
    causal: bool,
    dtype: torch.dtype,
    block_size: int,
) -> None:
    batch, heads, query_length, key_length, head_dim = shape
    kv_heads = max(heads // 2, 1)
    generator = torch.Generator().manual_seed(
        batch + heads + query_length + key_length + head_dim + block_size
    )
    q = torch.randn(
        batch, heads, query_length, head_dim, generator=generator, dtype=dtype
    )
    k = torch.randn(
        batch, kv_heads, key_length, head_dim, generator=generator, dtype=dtype
    )
    v = torch.randn(
        batch, kv_heads, key_length, head_dim, generator=generator, dtype=dtype
    )

    reference = sdpa_reference(q.double(), k.double(), v.double(), causal=causal)
    actual = tiled_attention(
        q,
        k,
        v,
        causal=causal,
        block_q=block_size,
        block_k=block_size,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 2e-3

    assert actual.dtype == dtype
    assert torch.allclose(actual.double(), reference, atol=tolerance, rtol=tolerance)


def test_tiled_attention_supports_padding_masks_and_fully_masked_rows() -> None:
    torch.manual_seed(12)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 7, 8)
    v = torch.randn(2, 2, 7, 6)
    padding_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )
    padding_mask[0, :5] = False

    reference = sdpa_reference(q.double(), k.double(), v.double(), False, padding_mask)
    actual = tiled_attention(
        q,
        k,
        v,
        causal=False,
        block_q=3,
        block_k=4,
        attn_mask=padding_mask,
    )

    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    assert torch.allclose(actual.double(), reference, atol=1e-5, rtol=1e-5)


def test_tiled_attention_does_not_call_fused_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("tiled attention must not call scaled_dot_product_attention")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 1, 7, 8)
    v = torch.randn(1, 1, 7, 8)

    result = tiled_attention(q, k, v, causal=True, block_q=2, block_k=3)

    assert result.shape == q.shape[:-1] + (v.shape[-1],)


def test_attention_rejects_invalid_dimensions_and_block_sizes() -> None:
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="divisible"):
        sdpa_reference(q, k, v, causal=False)
    with pytest.raises(ValueError, match="block_q"):
        tiled_attention(q[:, :2], k, v, causal=False, block_q=0)
    with pytest.raises(ValueError, match="attn_mask"):
        tiled_attention(
            q[:, :2],
            k,
            v,
            causal=False,
            attn_mask=torch.ones(3, dtype=torch.bool),
        )

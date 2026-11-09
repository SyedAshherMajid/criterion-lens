"""Memory-efficient blockwise attention using online softmax accumulation."""

import torch

from criterion_lens.attention.reference import (
    _apply_mask,
    _compute_dtype,
    _prepare_mask,
    _validate_inputs,
)


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    block_q: int = 64,
    block_k: int = 64,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute attention in query/key tiles without forming the full score matrix.

    For each query row, maintain the running maximum `m`, normalizer `l`, and
    weighted-value sum `acc`. A new key tile changes the maximum from `m` to
    `m_new`; its old contribution must be rescaled by
    `alpha = exp(m - m_new)` so that `l` and `acc` remain in the same
    normalization frame. Empty rows keep `l=0` and return zero. Causal tiles
    wholly above the diagonal are skipped; diagonal tiles apply the offset
    `Nk-Nq` mask, supporting one-token decode and non-square attention.

    The implementation accumulates in fp32 (fp64 for fp64 inputs) and retains
    only one `[B, H, block_q, block_k]` score tile at a time. GQA key/value
    heads are repeated to match query heads, and padding masks can be boolean
    keep masks or additive floating-point masks.
    """
    if not isinstance(block_q, int) or isinstance(block_q, bool) or block_q <= 0:
        raise ValueError("block_q must be a positive integer")
    if not isinstance(block_k, int) or isinstance(block_k, bool) or block_k <= 0:
        raise ValueError("block_k must be a positive integer")
    batch, heads, kv_heads, query_length, key_length = _validate_inputs(q, k, v)
    mask = _prepare_mask(
        attn_mask,
        batch=batch,
        heads=heads,
        query_length=query_length,
        key_length=key_length,
        device=q.device,
    )
    compute_dtype = _compute_dtype(q)
    repeat_factor = heads // kv_heads
    q_compute = q.to(dtype=compute_dtype)
    k_compute = k.to(dtype=compute_dtype).repeat_interleave(repeat_factor, dim=1)
    v_compute = v.to(dtype=compute_dtype).repeat_interleave(repeat_factor, dim=1)
    attention_scale = scale if scale is not None else q.shape[-1] ** -0.5
    output_blocks: list[torch.Tensor] = []
    causal_offset = key_length - query_length

    for query_start in range(0, query_length, block_q):
        query_end = min(query_start + block_q, query_length)
        query_tile = q_compute[:, :, query_start:query_end, :]
        rows = query_end - query_start
        running_max = torch.full(
            (batch, heads, rows, 1),
            -torch.inf,
            dtype=compute_dtype,
            device=q.device,
        )
        normalizer = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (batch, heads, rows, v.shape[-1]),
            dtype=compute_dtype,
            device=q.device,
        )

        for key_start in range(0, key_length, block_k):
            key_end = min(key_start + block_k, key_length)
            if causal and key_start > query_end - 1 + causal_offset:
                continue
            key_tile = k_compute[:, :, key_start:key_end, :]
            value_tile = v_compute[:, :, key_start:key_end, :]
            scores = torch.matmul(query_tile, key_tile.transpose(-2, -1))
            scores = scores * attention_scale
            scores = _apply_mask(
                scores,
                mask,
                causal=causal,
                query_start=query_start,
                key_start=key_start,
                total_query_length=query_length,
                total_key_length=key_length,
            )

            tile_max = scores.amax(dim=-1, keepdim=True)
            next_max = torch.maximum(running_max, tile_max)
            safe_next_max = torch.where(
                torch.isfinite(next_max),
                next_max,
                torch.zeros_like(next_max),
            )
            rescale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - safe_next_max),
                torch.zeros_like(running_max),
            )
            probabilities = torch.exp(scores - safe_next_max)
            probabilities = torch.where(
                torch.isfinite(scores),
                probabilities,
                torch.zeros_like(probabilities),
            )
            normalizer = rescale * normalizer + probabilities.sum(dim=-1, keepdim=True)
            accumulator = (
                rescale * accumulator + torch.matmul(probabilities, value_tile)
            )
            running_max = next_max

        output = accumulator / normalizer.clamp_min(torch.finfo(compute_dtype).tiny)
        output = torch.where(normalizer > 0, output, torch.zeros_like(output))
        output_blocks.append(output)

    return torch.cat(output_blocks, dim=-2).to(dtype=q.dtype)

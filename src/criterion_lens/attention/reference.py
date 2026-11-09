"""Numerically stable attention reference for grouped-query inputs."""

import torch


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [batch, heads, length, features]")
    batch, query_heads, query_length, query_dim = q.shape
    key_batch, kv_heads, key_length, key_dim = k.shape
    value_batch, value_heads, value_length, value_dim = v.shape
    if batch != key_batch or batch != value_batch:
        raise ValueError("q, k, and v must have the same batch size")
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by the KV head count")
    if kv_heads != value_heads or key_length != value_length:
        raise ValueError("k and v must have matching head and sequence dimensions")
    if query_dim != key_dim:
        raise ValueError("q and k must have the same head dimension")
    if min(batch, query_heads, kv_heads, query_length, key_length, query_dim, value_dim) <= 0:
        raise ValueError("attention dimensions must be positive")
    if not all(tensor.is_floating_point() for tensor in (q, k, v)):
        raise TypeError("q, k, and v must have floating-point dtypes")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same device")
    return batch, query_heads, kv_heads, query_length, key_length


def _compute_dtype(q: torch.Tensor) -> torch.dtype:
    return torch.float64 if q.dtype == torch.float64 else torch.float32


def _prepare_mask(
    attn_mask: torch.Tensor | None,
    *,
    batch: int,
    heads: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None
    if attn_mask.device != device:
        raise ValueError("attn_mask must be on the same device as q, k, and v")
    if attn_mask.ndim == 1:
        mask = attn_mask[None, None, None, :]
    elif attn_mask.ndim == 2:
        mask = attn_mask[:, None, None, :]
    elif attn_mask.ndim == 3:
        mask = attn_mask[:, None, :, :]
    elif attn_mask.ndim == 4:
        mask = attn_mask
    else:
        raise ValueError("attn_mask must have 1, 2, 3, or 4 dimensions")
    if mask.dtype == torch.bool or not mask.is_floating_point():
        mask = mask != 0
    try:
        return torch.broadcast_to(mask, (batch, heads, query_length, key_length))
    except RuntimeError as error:
        raise ValueError(
            "attn_mask must be broadcastable to [batch, heads, query_length, key_length]"
        ) from error


def _apply_mask(
    scores: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    causal: bool,
    query_start: int,
    key_start: int,
    total_query_length: int,
    total_key_length: int,
) -> torch.Tensor:
    query_length, key_length = scores.shape[-2:]
    if causal:
        query_positions = torch.arange(
            query_start,
            query_start + query_length,
            device=scores.device,
        )
        key_positions = torch.arange(
            key_start,
            key_start + key_length,
            device=scores.device,
        )
        allowed = key_positions[None, :] <= (
            query_positions[:, None] + total_key_length - total_query_length
        )
        scores = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
    if mask is not None:
        selected_mask = mask[
            ...,
            query_start : query_start + query_length,
            key_start : key_start + key_length,
        ]
        if selected_mask.dtype == torch.bool:
            scores = scores.masked_fill(~selected_mask, -torch.inf)
        else:
            scores = scores + selected_mask.to(dtype=scores.dtype)
    return scores


def _safe_softmax(scores: torch.Tensor) -> torch.Tensor:
    row_max = scores.amax(dim=-1, keepdim=True)
    safe_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exponentials = torch.exp(scores - safe_max)
    exponentials = torch.where(torch.isfinite(scores), exponentials, torch.zeros_like(scores))
    denominator = exponentials.sum(dim=-1, keepdim=True)
    return exponentials / denominator.clamp_min(torch.finfo(scores.dtype).tiny)


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute full attention scores, accumulating in fp32 or fp64.

    Shapes are `[B, H, Nq, d]`, `[B, Hkv, Nk, d]`, and
    `[B, Hkv, Nk, dv]`. Boolean and integer masks use nonzero for visible keys;
    floating masks are additive, as in PyTorch SDPA. Causal visibility uses the
    query offset `Nk - Nq`, which allows a short decode query to attend to the
    correct suffix of the key sequence. Fully masked query rows return zero.
    """
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
    scores = torch.matmul(q_compute, k_compute.transpose(-2, -1)) * attention_scale
    scores = _apply_mask(
        scores,
        mask,
        causal=causal,
        query_start=0,
        key_start=0,
        total_query_length=query_length,
        total_key_length=key_length,
    )
    probabilities = _safe_softmax(scores)
    output = torch.matmul(probabilities, v_compute)
    row_has_keys = torch.isfinite(scores).any(dim=-1, keepdim=True)
    output = torch.where(row_has_keys, output, torch.zeros_like(output))
    return output.to(dtype=q.dtype)

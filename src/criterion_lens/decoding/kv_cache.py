"""Greedy decoding with and without the Transformers-managed KV cache."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from criterion_lens.decoding.cache_utils import cache_bytes, cache_seq_len
from criterion_lens.decoding.timing import Timer


@dataclass
class GenResult:
    """Generated tokens, prefill/decode times, and per-token instrumentation."""

    tokens: torch.Tensor
    prefill_s: float
    decode_s: float
    per_step: list[dict[str, Any]]


def _validate_inputs(
    model: Any,
    input_ids: torch.Tensor,
    max_new: int,
    eos_id: int | None,
) -> None:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, non-empty prompt_length]")
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 0:
        raise ValueError("max_new must be a non-negative integer")
    if eos_id is not None and (
        not isinstance(eos_id, int) or isinstance(eos_id, bool) or eos_id < 0
    ):
        raise ValueError("eos_id must be a non-negative integer or None")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = input_ids.device
    if model_device != input_ids.device:
        raise ValueError(
            f"input_ids are on {input_ids.device}, but model parameters are on {model_device}"
        )


def _next_token(logits: torch.Tensor) -> tuple[torch.Tensor, float]:
    next_logits = logits[:, -1, :]
    if next_logits.shape[-1] < 2:
        raise ValueError("Greedy decoding requires a vocabulary with at least two tokens")
    top_values = torch.topk(next_logits.float(), k=2, dim=-1).values
    margin = float((top_values[:, 0] - top_values[:, 1]).item())
    return next_logits.argmax(dim=-1, keepdim=True), margin


def _step_record(
    step: int,
    token: torch.Tensor,
    margin: float,
    processed_tokens: int,
    cache: Any | None,
) -> dict[str, int | float]:
    return {
        "step": step,
        "token_id": int(token.item()),
        "logits_margin": margin,
        "processed_tokens": processed_tokens,
        "cache_len": cache_seq_len(cache) if cache is not None else 0,
        "cache_bytes": cache_bytes(cache) if cache is not None else 0,
    }


def _finish(tokens: list[torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
    if not tokens:
        return input_ids.new_empty((1, 0))
    return torch.cat(tokens, dim=1)


def greedy_no_cache(
    model: Any,
    input_ids: torch.Tensor,
    max_new: int,
    eos_id: int | None = None,
) -> GenResult:
    """Recompute the full prefix at every step as a no-cache greedy reference.

    Each forward pass receives explicit absolute `position_ids`. Its causal
    attention mask covers the complete prefix; no past key/value tensors are
    retained. The first forward is reported as prefill and subsequent full
    forwards as decode time.
    """
    _validate_inputs(model, input_ids, max_new, eos_id)
    if max_new == 0:
        return GenResult(input_ids.new_empty((1, 0)), 0.0, 0.0, [])

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    generated: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    prefill_s = 0.0
    decode_s = 0.0
    try:
        with torch.inference_mode():
            for step in range(max_new):
                context = torch.cat([input_ids, *generated], dim=1)
                length = context.shape[1]
                position_ids = torch.arange(
                    length,
                    dtype=torch.long,
                    device=input_ids.device,
                ).unsqueeze(0)
                with Timer(input_ids.device) as timer:
                    output = model(
                        input_ids=context,
                        attention_mask=torch.ones_like(context),
                        position_ids=position_ids,
                        use_cache=False,
                    )
                if timer.elapsed_s is None:
                    raise RuntimeError("Timer did not record an elapsed duration")
                if step == 0:
                    prefill_s = timer.elapsed_s
                else:
                    decode_s += timer.elapsed_s
                token, margin = _next_token(output.logits)
                generated.append(token)
                records.append(_step_record(step + 1, token, margin, length, None))
                if eos_id is not None and int(token.item()) == eos_id:
                    break
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    return GenResult(_finish(generated, input_ids), prefill_s, decode_s, records)


def greedy_kv_cache(
    model: Any,
    input_ids: torch.Tensor,
    max_new: int,
    eos_id: int | None = None,
) -> GenResult:
    """Prefill once, then decode with absolute positions and a DynamicCache.

    RoPE keys are rotated at their absolute positions during each model forward
    and stored in the Hugging Face cache. The first call uses positions
    `0..prompt_length-1`; each later one-token call uses the current cache length,
    not zero. The attention mask grows from the prompt length by one as each
    previously generated token is processed. A one-token query sees all cached
    keys, so the model's normal attention implementation handles its masking.
    Cache growth is linear; bytes are measured from allocated key/value tensors
    and can be checked against `2 * layers * batch * kv_heads * seq_len *
    head_dim * bytes_per_element`.
    """
    _validate_inputs(model, input_ids, max_new, eos_id)
    if max_new == 0:
        return GenResult(input_ids.new_empty((1, 0)), 0.0, 0.0, [])

    prompt_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(
        prompt_length,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    cache_position = torch.arange(
        prompt_length,
        dtype=torch.long,
        device=input_ids.device,
    )
    cache = DynamicCache(config=getattr(model, "config", None))
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    generated: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    decode_s = 0.0
    try:
        with torch.inference_mode():
            with Timer(input_ids.device) as timer:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                )
            if timer.elapsed_s is None:
                raise RuntimeError("Timer did not record an elapsed duration")
            prefill_s = timer.elapsed_s
            cache = output.past_key_values
            token, margin = _next_token(output.logits)
            generated.append(token)
            records.append(
                _step_record(
                    1,
                    token,
                    margin,
                    cache_seq_len(cache),
                    cache,
                )
            )

            for step in range(1, max_new):
                if eos_id is not None and int(generated[-1].item()) == eos_id:
                    break
                current_length = cache_seq_len(cache)
                position = torch.tensor(
                    [[current_length]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                cache_position = position.squeeze(0)
                attention_mask = torch.ones(
                    (1, current_length + 1),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                with Timer(input_ids.device) as timer:
                    output = model(
                        input_ids=generated[-1],
                        attention_mask=attention_mask,
                        position_ids=position,
                        cache_position=cache_position,
                        past_key_values=cache,
                        use_cache=True,
                    )
                if timer.elapsed_s is None:
                    raise RuntimeError("Timer did not record an elapsed duration")
                decode_s += timer.elapsed_s
                cache = output.past_key_values
                token, margin = _next_token(output.logits)
                generated.append(token)
                records.append(
                    _step_record(
                        step + 1,
                        token,
                        margin,
                        cache_seq_len(cache),
                        cache,
                    )
                )
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    return GenResult(_finish(generated, input_ids), prefill_s, decode_s, records)

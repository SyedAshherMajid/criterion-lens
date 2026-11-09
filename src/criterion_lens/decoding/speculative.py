"""Greedy speculative decoding with target/draft DynamicCache instances."""

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from criterion_lens.decoding.cache_utils import cache_seq_len, crop_cache
from criterion_lens.decoding.timing import Timer


@dataclass
class SpecResult:
    """Speculative output and instrumentation for every verification round."""

    tokens: torch.Tensor
    iterations: int
    proposed: int
    accepted: int
    accepted_per_iteration: list[int]
    target_forwards: int
    draft_forwards: int
    timings: dict[str, float]
    peak_memory_bytes: int | None
    cache_lengths: list[dict[str, int]]


def validate_tokenizer_compatibility(target_tokenizer: Any, draft_tokenizer: Any) -> int:
    """Require identical token-ID mappings and special-token IDs; return vocab size."""
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        raise ValueError("Target and draft tokenizers have different token-to-ID vocabularies")
    special_token_fields = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "additional_special_tokens_ids",
    )
    mismatches = [
        field
        for field in special_token_fields
        if getattr(target_tokenizer, field, None) != getattr(draft_tokenizer, field, None)
    ]
    if mismatches:
        raise ValueError(
            "Target and draft tokenizers have different special-token IDs: "
            + ", ".join(mismatches)
        )
    return len(target_vocab)


def _device_of(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _vocab_size(model: Any) -> int:
    config = getattr(model, "config", None)
    size = getattr(config, "vocab_size", None)
    if not isinstance(size, int) or size <= 1:
        raise ValueError("Target and draft models must define a vocabulary size greater than one")
    return size


def _cached_forward(
    model: Any,
    tokens: torch.Tensor,
    cache: Any,
    device: torch.device,
) -> tuple[Any, float]:
    start = cache_seq_len(cache)
    length = tokens.shape[1]
    position_ids = torch.arange(start, start + length, dtype=torch.long, device=device)[None, :]
    cache_position = torch.arange(start, start + length, dtype=torch.long, device=device)
    attention_mask = torch.ones(
        (1, start + length),
        dtype=tokens.dtype,
        device=device,
    )
    with Timer(device) as timer:
        output = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
    if timer.elapsed_s is None:
        raise RuntimeError("Timer did not record an elapsed duration")
    return output, timer.elapsed_s


def _argmax(logits: torch.Tensor, vocab_size: int, position: int = -1) -> torch.Tensor:
    available_logits = logits[:, position, :vocab_size]
    if available_logits.shape[-1] < 2:
        raise ValueError("Greedy decoding requires at least two shared vocabulary tokens")
    return available_logits.argmax(dim=-1, keepdim=True)


def _model_mode(model: Any) -> bool:
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    return was_training


def speculative_greedy(
    target: Any,
    draft: Any,
    input_ids: torch.Tensor,
    max_new: int,
    k: int,
    eos_id: int | None = None,
) -> SpecResult:
    """Generate target-greedy tokens using k draft proposals per verify call.

    The invariant is that the target cache contains `seq[:-1]`; `seq[-1]` is
    pending. The draft cache contains a prefix `seq[:d_len]` and may lag behind.
    Each round catches the draft up, proposes up to k tokens, and verifies
    `[seq[-1], d1, ..., dk]` in one target forward. The target's next-token
    argmaxes are aligned with each proposal; accept the longest matching prefix
    and append the following target argmax as the correction or bonus token.
    Both Hugging Face caches are cropped to remove rejected suffixes. When all
    proposals are accepted, the draft cache intentionally lacks the final
    proposal and catches up on the next round.

    The output equals target greedy decoding in exact arithmetic. Batched
    verification can use different floating-point reduction kernels than
    one-token decoding, so low-margin fp16 predictions may differ by hardware.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, non-empty prompt_length]")
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 0:
        raise ValueError("max_new must be a non-negative integer")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if eos_id is not None and (
        not isinstance(eos_id, int) or isinstance(eos_id, bool) or eos_id < 0
    ):
        raise ValueError("eos_id must be a non-negative integer or None")
    if max_new == 0:
        return SpecResult(
            tokens=input_ids.new_empty((1, 0)),
            iterations=0,
            proposed=0,
            accepted=0,
            accepted_per_iteration=[],
            target_forwards=0,
            draft_forwards=0,
            timings={"target_forward_s": 0.0, "draft_forward_s": 0.0, "total_s": 0.0},
            peak_memory_bytes=None,
            cache_lengths=[],
        )

    target_device = _device_of(target)
    draft_device = _device_of(draft)
    if target_device != draft_device:
        raise ValueError("Target and draft models must be on the same device")
    if input_ids.device != target_device:
        raise ValueError(
            f"input_ids are on {input_ids.device}, but models are on {target_device}"
        )
    vocab_size = min(_vocab_size(target), _vocab_size(draft))
    if eos_id is not None and eos_id >= vocab_size:
        raise ValueError("eos_id must be smaller than the shared vocabulary size")
    if input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
        raise ValueError("input_ids contain token IDs outside the shared vocabulary")

    target_cache = DynamicCache(config=target.config)
    draft_cache = DynamicCache(config=draft.config)
    target_training = _model_mode(target)
    draft_training = _model_mode(draft)
    sequence = input_ids
    generated: list[torch.Tensor] = []
    accepted_per_iteration: list[int] = []
    cache_lengths: list[dict[str, int]] = []
    target_forwards = 0
    draft_forwards = 0
    total_proposed = 0
    total_accepted = 0
    target_forward_s = 0.0
    draft_forward_s = 0.0
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    total_start = time.perf_counter()
    try:
        with torch.inference_mode():
            target_output, elapsed = _cached_forward(
                target,
                input_ids,
                target_cache,
                target_device,
            )
            target_forward_s += elapsed
            target_forwards += 1
            target_cache = target_output.past_key_values
            first_target_token = _argmax(target_output.logits, vocab_size)
            sequence = torch.cat([sequence, first_target_token], dim=1)
            generated.append(first_target_token)

            draft_output, elapsed = _cached_forward(
                draft,
                input_ids,
                draft_cache,
                draft_device,
            )
            draft_forward_s += elapsed
            draft_forwards += 1
            draft_cache = draft_output.past_key_values

            while len(generated) < max_new and (
                eos_id is None or int(generated[-1].item()) != eos_id
            ):
                target_length = cache_seq_len(target_cache)
                sequence_length = sequence.shape[1]
                if target_length != sequence_length - 1:
                    raise RuntimeError(
                        "Target cache invariant failed before verify: "
                        f"cache={target_length}, sequence={sequence_length}"
                    )
                draft_length = cache_seq_len(draft_cache)
                if draft_length >= sequence_length:
                    raise RuntimeError(
                        "Draft cache must lag the sequence so its pending suffix can be fed"
                    )
                proposal_count = min(k, max_new - len(generated))
                catch_up = sequence[:, draft_length:]
                draft_output, elapsed = _cached_forward(
                    draft,
                    catch_up,
                    draft_cache,
                    draft_device,
                )
                draft_forward_s += elapsed
                draft_forwards += 1
                draft_cache = draft_output.past_key_values
                proposals = [_argmax(draft_output.logits, vocab_size)]
                for _ in range(1, proposal_count):
                    draft_output, elapsed = _cached_forward(
                        draft,
                        proposals[-1],
                        draft_cache,
                        draft_device,
                    )
                    draft_forward_s += elapsed
                    draft_forwards += 1
                    draft_cache = draft_output.past_key_values
                    proposals.append(_argmax(draft_output.logits, vocab_size))

                proposal_tokens = torch.cat(proposals, dim=1)
                verify_tokens = torch.cat([sequence[:, -1:], proposal_tokens], dim=1)
                target_output, elapsed = _cached_forward(
                    target,
                    verify_tokens,
                    target_cache,
                    target_device,
                )
                target_forward_s += elapsed
                target_forwards += 1
                target_cache = target_output.past_key_values
                target_predictions = target_output.logits[:, :, :vocab_size].argmax(dim=-1)
                accepted = 0
                for index in range(proposal_count):
                    if int(proposals[index].item()) != int(target_predictions[0, index].item()):
                        break
                    accepted += 1
                    if eos_id is not None and int(proposals[index].item()) == eos_id:
                        break

                total_proposed += proposal_count
                total_accepted += accepted
                accepted_per_iteration.append(accepted)
                correction = target_predictions[:, accepted : accepted + 1]
                committed = torch.cat([proposal_tokens[:, :accepted], correction], dim=1)
                remaining = max_new - len(generated)
                committed = committed[:, :remaining]
                new_tokens: list[torch.Tensor] = []
                stop_for_eos = False
                for index in range(committed.shape[1]):
                    token = committed[:, index : index + 1]
                    new_tokens.append(token)
                    if eos_id is not None and int(token.item()) == eos_id:
                        stop_for_eos = True
                        break
                if not new_tokens:
                    raise RuntimeError("Speculative verify produced no committed token")
                committed_tokens = torch.cat(new_tokens, dim=1)
                sequence = torch.cat([sequence, committed_tokens], dim=1)
                generated.extend(
                    committed_tokens[:, index : index + 1]
                    for index in range(committed_tokens.shape[1])
                )

                target_length_after_verify = cache_seq_len(target_cache)
                draft_length_after_proposals = cache_seq_len(draft_cache)
                target_cache = crop_cache(target_cache, sequence.shape[1] - 1)
                draft_cache = crop_cache(
                    draft_cache,
                    min(draft_length_after_proposals, sequence.shape[1] - 1),
                )
                target_length_after_crop = cache_seq_len(target_cache)
                draft_length_after_crop = cache_seq_len(draft_cache)
                if target_length_after_crop != sequence.shape[1] - 1:
                    raise RuntimeError("Target cache invariant failed after crop")
                if draft_length_after_crop > sequence.shape[1]:
                    raise RuntimeError("Draft cache extends beyond the committed sequence")
                cache_lengths.append(
                    {
                        "iteration": len(accepted_per_iteration),
                        "sequence_before": sequence_length,
                        "target_before": target_length,
                        "draft_before": draft_length,
                        "proposal_count": proposal_count,
                        "accepted": accepted,
                        "target_after_verify": target_length_after_verify,
                        "draft_after_proposals": draft_length_after_proposals,
                        "sequence_after": sequence.shape[1],
                        "target_after_crop": target_length_after_crop,
                        "draft_after_crop": draft_length_after_crop,
                    }
                )
                if stop_for_eos:
                    break
    finally:
        if hasattr(target, "train"):
            target.train(target_training)
        if hasattr(draft, "train"):
            draft.train(draft_training)

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_memory = int(torch.cuda.max_memory_allocated(target_device))
    else:
        peak_memory = None
    total_s = time.perf_counter() - total_start
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    return SpecResult(
        tokens=torch.cat(generated, dim=1),
        iterations=len(accepted_per_iteration),
        proposed=total_proposed,
        accepted=total_accepted,
        accepted_per_iteration=accepted_per_iteration,
        target_forwards=target_forwards,
        draft_forwards=draft_forwards,
        timings={
            "target_forward_s": target_forward_s,
            "draft_forward_s": draft_forward_s,
            "total_s": total_s,
        },
        peak_memory_bytes=peak_memory,
        cache_lengths=cache_lengths,
    )

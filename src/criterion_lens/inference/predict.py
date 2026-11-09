"""Generate one criterion prediction with validated evidence and a scoring fallback."""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from criterion_lens.data.prompts import ALLOWED_LABELS, build_messages, format_target
from criterion_lens.data.sentences import split_sentences
from criterion_lens.inference.explain import build_explanation
from criterion_lens.inference.parse import parse_output
from criterion_lens.inference.schema import CriterionResult

Decoder = Callable[..., Any]


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _im_end_id(tokenizer: Any) -> int | None:
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return getattr(tokenizer, "eos_token_id", None)
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unknown_id:
        return getattr(tokenizer, "eos_token_id", None)
    return token_id


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Prediction expects a single generated sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def _decoder_metadata(result: Any) -> dict[str, Any]:
    fields = (
        "prefill_s",
        "decode_s",
        "iterations",
        "proposed",
        "accepted",
        "target_forwards",
        "draft_forwards",
    )
    return {
        field: getattr(result, field)
        for field in fields
        if getattr(result, field, None) is not None
    }


def _fallback_label(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    criterion_type: str,
    evidence_ids: Sequence[int],
) -> str:
    """Score all allowed JSON continuations in one causal-model forward pass."""
    if prompt_ids.shape[1] == 0:
        raise ValueError("Cannot score continuations for an empty prompt")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id for fallback scoring")

    labels = ALLOWED_LABELS[criterion_type]
    target_ids = [
        tokenizer.encode(
            format_target(evidence_ids, label),
            add_special_tokens=False,
        )
        for label in labels
    ]
    if any(not ids for ids in target_ids):
        raise ValueError("Tokenizer produced an empty fallback continuation")

    prompt_length = prompt_ids.shape[1]
    max_length = prompt_length + max(map(len, target_ids))
    input_ids = torch.full(
        (len(labels), max_length),
        fill_value=pad_token_id,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, continuation in enumerate(target_ids):
        length = prompt_length + len(continuation)
        input_ids[row, :prompt_length] = prompt_ids[0]
        input_ids[row, prompt_length:length] = torch.tensor(
            continuation,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        attention_mask[row, :length] = 1

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    scores = []
    for row, continuation in enumerate(target_ids):
        next_token_logits = logits[
            row,
            prompt_length - 1 : prompt_length + len(continuation) - 1,
        ]
        target = torch.tensor(
            continuation,
            dtype=torch.long,
            device=next_token_logits.device,
        )
        scores.append(
            torch.log_softmax(next_token_logits.float(), dim=-1)
            .gather(1, target.unsqueeze(1))
            .sum()
        )
    return labels[int(torch.stack(scores).argmax().item())]


def _generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    decoding: str,
    max_new_tokens: int,
    eos_id: int | None,
    decoder: Decoder | None,
) -> tuple[str, dict[str, Any]]:
    if decoder is not None:
        result = decoder(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
        )
        if isinstance(result, str):
            return result, {}
        tokens = getattr(result, "tokens", result)
        return tokenizer.decode(_token_ids(tokens), skip_special_tokens=True), (
            _decoder_metadata(result)
        )
    if decoding != "no_cache":
        raise ValueError(f"A decoder callback is required for decoding mode {decoding!r}")

    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[:, input_ids.shape[1] :]
    return tokenizer.decode(_token_ids(new_tokens), skip_special_tokens=True), {}


def predict_criterion(
    model: Any,
    tokenizer: Any,
    note: str,
    criterion_id: str,
    criterion_type: str,
    criterion_text: str,
    *,
    decoding: str = "no_cache",
    max_new_tokens: int = 64,
    decoder: Decoder | None = None,
) -> CriterionResult:
    """Predict one criterion; evidence and explanations remain note-grounded.

    Custom decoding modes receive a callback that returns generated token IDs or
    an object with generated IDs in `.tokens`. The default path is greedy and
    disables KV caching so it remains a clear reference implementation.
    """
    if criterion_type not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported criterion_type: {criterion_type}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    sentences = split_sentences(note)
    messages = build_messages(sentences, criterion_type, criterion_text)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    device = _model_device(model)
    input_ids = input_ids.to(device)
    eos_id = _im_end_id(tokenizer)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            raw_output, decoder_info = _generate(
                model,
                tokenizer,
                input_ids,
                decoding=decoding,
                max_new_tokens=max_new_tokens,
                eos_id=eos_id,
                decoder=decoder,
            )
            parsed = parse_output(raw_output, criterion_type, len(sentences))
            fallback_used = not parsed.parse_ok
            label = parsed.label
            if fallback_used:
                label = _fallback_label(
                    model,
                    tokenizer,
                    input_ids,
                    criterion_type,
                    parsed.evidence_ids,
                )
            if label is None:
                raise RuntimeError("Validated parsing and fallback produced no label")
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    evidence_text = [sentences[sentence_id][1] for sentence_id in parsed.evidence_ids]
    decisive = label in {"included", "not included", "excluded", "not excluded"}
    return CriterionResult(
        criterion_id=criterion_id,
        criterion_type=criterion_type,
        criterion_text=criterion_text,
        label=label,
        evidence_ids=parsed.evidence_ids,
        evidence_text=evidence_text,
        explanation=build_explanation(
            label,
            criterion_type,
            parsed.evidence_ids,
            evidence_text,
        ),
        missing_information=label == "not enough information" or not parsed.evidence_ids,
        unsupported_decision=decisive and not parsed.evidence_ids,
        invalid_ids_dropped=parsed.invalid_ids_dropped,
        raw_output=parsed.raw_output,
        parse_ok=parsed.parse_ok,
        invalid_label=parsed.invalid_label,
        fallback_used=fallback_used,
        decoding={
            "mode": decoding,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            **decoder_info,
        },
    )

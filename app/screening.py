"""Application-level screening orchestration and report creation."""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from criterion_lens.data.sentences import split_sentences
from criterion_lens.decoding.kv_cache import greedy_kv_cache
from criterion_lens.decoding.speculative import (
    speculative_greedy,
    validate_tokenizer_compatibility,
)
from criterion_lens.inference.aggregate import aggregate_trial
from criterion_lens.inference.predict import predict_criterion
from criterion_lens.modeling.load_model import LoadedModel, load_draft_model, load_model
from criterion_lens.report.export import export_screening_report
from criterion_lens.trials.criteria_parser import CriterionInput, parse_criteria_text
from criterion_lens.utils.env import load_env

MODEL_OPTIONS = {
    "Fine-tuned Qwen2.5 1.5B": "Ashher123/criterion-lens-qwen2.5-1.5b-merged",
    "Base Qwen2.5 1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Fine-tuned Qwen2.5 0.5B": "Ashher123/criterion-lens-qwen2.5-0.5b-merged",
    "Base Qwen2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
}
DISCLAIMER = (
    "Research prototype only. This is not medical advice or a ground-truth eligibility "
    "decision; a qualified human must review all source records."
)
ModelLoader = Callable[..., LoadedModel]


def screen_patient(
    note: str,
    criteria_text: str,
    *,
    model_id: str,
    decoding: str = "greedy_kv",
    max_criteria: int = 20,
    speculative_k: int = 4,
    criteria_rows: list[list[str]] | None = None,
    progress: Callable[..., Any] | None = None,
    write_reports: bool = True,
    model_loader: ModelLoader = load_model,
    draft_loader: ModelLoader = load_draft_model,
    output_root: Path = Path("reports") / "exports",
) -> dict[str, Any]:
    """Run criterion screening and return results plus saved provenance reports."""
    load_env()
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Patient note must not be empty")
    if not split_sentences(note):
        raise ValueError("Patient note must contain at least one sentence")
    if max_criteria <= 0:
        raise ValueError("max_criteria must be positive")
    if criteria_rows:
        criteria = []
        for row in criteria_rows:
            if len(row) < 2 or not row[0] or not row[1]:
                raise ValueError("Each editable criterion row needs a type and criterion text")
            if row[0] not in {"inclusion", "exclusion"}:
                raise ValueError(f"Unsupported criterion type in editable table: {row[0]}")
            criterion_type = "inclusion" if row[0] == "inclusion" else "exclusion"
            criteria.append(CriterionInput(criterion_type, row[1]))
    else:
        criteria = parse_criteria_text(criteria_text)
    if len(criteria) > max_criteria:
        raise ValueError(f"Parsed {len(criteria)} criteria; maximum allowed is {max_criteria}")
    if decoding not in {"no_cache", "greedy_kv", "speculative"}:
        raise ValueError(f"Unsupported decoding mode: {decoding}")
    if speculative_k <= 0:
        raise ValueError("speculative_k must be positive")

    loaded = model_loader(model_id)
    draft: LoadedModel | None = None
    shared_vocab_size: int | None = None
    if decoding == "speculative":
        dtype_name = str(loaded.dtype).removeprefix("torch.")
        draft = draft_loader(device=str(loaded.device), dtype=dtype_name)
        shared_vocab_size = validate_tokenizer_compatibility(
            loaded.tokenizer,
            draft.tokenizer,
        )

    decoder: Callable[..., Any] | None = None
    if decoding == "greedy_kv":
        def greedy_decoder(model, input_ids, *, max_new_tokens, eos_id):
            return greedy_kv_cache(
                model,
                input_ids,
                max_new=max_new_tokens,
                eos_id=eos_id,
            )

        decoder = greedy_decoder
    elif decoding == "speculative" and draft is not None:
        def speculative_decoder(model, input_ids, *, max_new_tokens, eos_id):
            return speculative_greedy(
                model,
                draft.model,
                input_ids,
                max_new_tokens,
                speculative_k,
                eos_id,
                vocab_size=shared_vocab_size,
            )

        decoder = speculative_decoder

    results = []
    for index, criterion in enumerate(criteria):
        results.append(predict_criterion(
            loaded.model,
            loaded.tokenizer,
            note,
            f"criterion-{index + 1}",
            criterion.criterion_type,
            criterion.criterion_text,
            decoding=decoding,
            decoder=decoder,
        ))
        if progress is not None:
            progress(
                (index + 1, len(criteria)),
                desc=f"Evaluated criterion {index + 1} of {len(criteria)}",
            )
    aggregation = aggregate_trial(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export = export_screening_report(
        patient_note=note,
        criteria=results,
        aggregation=aggregation,
        provenance={
            "app_version": "0.1.0",
            "model_id": loaded.model_id,
            "device": str(loaded.device),
            "dtype": str(loaded.dtype),
            "decoding": decoding,
            "speculative_k": speculative_k if decoding == "speculative" else None,
            "timings": [result.decoding for result in results],
        },
        output_dir=output_root / f"screening-{timestamp}" if write_reports else None,
    )
    return {
        "verdict": aggregation["verdict"],
        "rows": [
            [
                result.criterion_id,
                result.criterion_type,
                result.criterion_text,
                result.label,
                ", ".join(map(str, result.evidence_ids)),
                " ".join(result.evidence_text),
                result.explanation,
            ]
            for result in results
        ],
        "report_html": export.html_text,
        "report_json": export.json_text,
        "report_markdown": export.markdown_text,
        "saved_files": export.saved_files,
        "decoder_stats": [result.decoding for result in results],
        "aggregation": aggregation,
        "disclaimer": DISCLAIMER,
    }


def compare_decoding_modes(
    note: str,
    criteria_text: str,
    *,
    model_id: str,
    max_criteria: int = 20,
    speculative_k: int = 4,
    criteria_rows: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Run all three greedy decoding modes on one current, user-provided prompt."""
    outputs = {}
    for mode in ("no_cache", "greedy_kv", "speculative"):
        start = perf_counter()
        outputs[mode] = screen_patient(
            note,
            criteria_text,
            model_id=model_id,
            decoding=mode,
            max_criteria=max_criteria,
            speculative_k=speculative_k,
            criteria_rows=criteria_rows,
            write_reports=False,
        )
        outputs[mode]["elapsed_s"] = perf_counter() - start

    reference = outputs["no_cache"]["rows"]
    comparison = []
    for mode, output in outputs.items():
        decision_rows = [[row[3], row[4]] for row in output["rows"]]
        reference_rows = [[row[3], row[4]] for row in reference]
        stats = output["decoder_stats"]
        proposed = sum(int(item.get("proposed", 0)) for item in stats)
        accepted = sum(int(item.get("accepted", 0)) for item in stats)
        comparison.append(
            {
                "mode": mode,
                "elapsed_s": output["elapsed_s"],
                "decision_and_evidence_equal_to_no_cache": decision_rows == reference_rows,
                "acceptance_rate": accepted / proposed if proposed else None,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
            }
        )
    return {
        "runs": comparison,
        "note": "User-input comparison run; results are not a benchmark measurement.",
    }


def attention_lab(
    *,
    sequence_length: int = 64,
    head_dim: int = 32,
    heads: int = 4,
    block_size: int = 16,
    causal: bool = True,
) -> dict[str, Any]:
    """Compare reference and tiled attention on deterministic synthetic inputs."""
    from criterion_lens.attention.reference import sdpa_reference
    from criterion_lens.attention.tiled import tiled_attention
    from criterion_lens.decoding.timing import Timer

    values = (sequence_length, head_dim, heads, block_size)
    if any(value <= 0 for value in values):
        raise ValueError("Attention dimensions and block size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(42)
    q, k, v = (
        torch.randn((1, heads, sequence_length, head_dim), generator=generator)
        for _ in range(3)
    )
    with Timer("cpu") as reference_timer:
        reference = sdpa_reference(q, k, v, causal=causal)
    with Timer("cpu") as tiled_timer:
        tiled = tiled_attention(
            q,
            k,
            v,
            causal=causal,
            block_q=block_size,
            block_k=block_size,
        )
    return {
        "device": "cpu",
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "heads": heads,
        "block_size": block_size,
        "causal": causal,
        "max_abs_error": float((reference - tiled).abs().max().item()),
        "mean_abs_error": float((reference - tiled).abs().mean().item()),
        "reference_time_s": reference_timer.elapsed_s,
        "tiled_time_s": tiled_timer.elapsed_s,
        "note": "Synthetic CPU lab input; not a project benchmark result.",
    }

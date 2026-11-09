"""Benchmark speculative decoding against target greedy KV-cache decoding."""

import argparse
import csv
import json
import logging
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from criterion_lens.data.prompts import build_messages
from criterion_lens.data.sentences import parse_numbered_note
from criterion_lens.decoding.kv_cache import GenResult, greedy_kv_cache
from criterion_lens.decoding.speculative import (
    SpecResult,
    speculative_greedy,
    validate_tokenizer_compatibility,
)
from criterion_lens.decoding.timing import Timer

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
DATA_AUDIT = PROJECT_ROOT / "results" / "data_audit.json"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "raw" / "annotations.parquet"
CSV_FIELDS = [
    "prompt_index",
    "input_length",
    "max_new",
    "k",
    "target_model",
    "draft_model",
    "dtype",
    "target_latency_s",
    "speculative_latency_s",
    "speedup",
    "target_tokens_per_second",
    "speculative_tokens_per_second",
    "sequence_equality_rate",
    "token_equality_rate",
    "acceptance_rate",
    "mean_accepted_per_iteration",
    "target_forwards_per_output_token",
    "proposed",
    "accepted",
    "target_forwards",
    "draft_forwards",
    "target_peak_memory_bytes",
    "speculative_peak_memory_bytes",
]


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(samples),
        "std": statistics.pstdev(samples),
        "median": statistics.median(samples),
    }


def _is_eos_token(tokenizer: Any) -> int | None:
    token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if token_id == unknown_id:
            token_id = None
    return token_id if token_id is not None else getattr(tokenizer, "eos_token_id", None)


def load_validation_prompts(tokenizer: Any, count: int) -> list[torch.Tensor]:
    """Build prompts only from the fixed validation patient IDs; never read test rows."""
    if count <= 0:
        raise ValueError("count must be a positive integer")
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
    if audit.get("dataset_revision") != manifest.get("dataset_revision"):
        raise ValueError("Cached annotation audit and split manifest revisions do not match")
    if not ANNOTATIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Cached sanitized annotations are required: {ANNOTATIONS_PATH}"
        )
    required = {"annotation_id", "patient_id", "note", "criterion_type", "criterion_text"}
    frame = pd.read_parquet(ANNOTATIONS_PATH, columns=sorted(required))
    val_ids = set(manifest["val"]["patient_ids"])
    validation = frame.loc[frame["patient_id"].isin(val_ids)].copy()
    expected_count = int(manifest["val"]["n_annotations"])
    if len(validation) != expected_count:
        raise ValueError(
            f"Validation cache has {len(validation)} rows; manifest expects {expected_count}"
        )
    if count > len(validation):
        raise ValueError(f"Requested {count} validation prompts; only {len(validation)} exist")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer must define a chat template")

    prompts = []
    selected = validation.sort_values("annotation_id").head(count)
    for _, row in selected.iterrows():
        messages = build_messages(
            parse_numbered_note(str(row["note"])),
            str(row["criterion_type"]),
            str(row["criterion_text"]),
        )
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Tokenizer must return one prompt with shape [1, sequence_length]")
        prompts.append(input_ids.to(dtype=torch.long))
    return prompts


def synthetic_prompts(
    *,
    count: int,
    length: int,
    vocab_size: int,
) -> list[torch.Tensor]:
    if count <= 0 or length <= 0:
        raise ValueError("Synthetic prompt count and length must be positive")
    if vocab_size <= 3:
        raise ValueError("Model vocabulary is too small for synthetic prompts")
    return [
        ((torch.arange(length, dtype=torch.long) + index) % (vocab_size - 3) + 3)[None, :]
        for index in range(count)
    ]


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _time_generation(
    function: Callable[[], GenResult | SpecResult],
    device: torch.device,
) -> tuple[GenResult | SpecResult, float, int | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    with Timer(device) as timer:
        result = function()
    if timer.elapsed_s is None:
        raise RuntimeError("Timer did not record an elapsed duration")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return result, timer.elapsed_s, peak_memory


def _first_divergence(
    target_tokens: torch.Tensor,
    speculative_tokens: torch.Tensor,
) -> dict[str, int] | None:
    target_ids = target_tokens[0].detach().cpu().tolist()
    speculative_ids = speculative_tokens[0].detach().cpu().tolist()
    for index, (target_id, speculative_id) in enumerate(
        zip(target_ids, speculative_ids, strict=False)
    ):
        if target_id != speculative_id:
            return {
                "step": index + 1,
                "target_token_id": target_id,
                "speculative_token_id": speculative_id,
            }
    if len(target_ids) != len(speculative_ids):
        return {
            "step": min(len(target_ids), len(speculative_ids)) + 1,
            "target_token_id": target_ids[len(speculative_ids)]
            if len(target_ids) > len(speculative_ids)
            else -1,
            "speculative_token_id": speculative_ids[len(target_ids)]
            if len(speculative_ids) > len(target_ids)
            else -1,
        }
    return None


def benchmark_case(
    target: torch.nn.Module,
    draft: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    prompt_index: int,
    max_new: int,
    k: int,
    eos_id: int | None,
    vocab_size: int,
    warmup: int,
    repeats: int,
    target_label: str,
    draft_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if max_new <= 0 or k <= 0 or warmup < 0 or repeats <= 0:
        raise ValueError("max_new/k/repeats must be positive and warmup non-negative")
    device = _model_device(target)

    def run_target() -> GenResult:
        return greedy_kv_cache(target, input_ids, max_new, eos_id)

    def run_speculative() -> SpecResult:
        return speculative_greedy(
            target,
            draft,
            input_ids,
            max_new,
            k,
            eos_id,
            vocab_size=vocab_size,
        )

    for _ in range(warmup):
        run_target()
        run_speculative()
    target_runs = [_time_generation(run_target, device) for _ in range(repeats)]
    speculative_runs = [_time_generation(run_speculative, device) for _ in range(repeats)]

    target_results = [result for result, _, _ in target_runs]
    speculative_results = [result for result, _, _ in speculative_runs]
    target_latencies = [elapsed for _, elapsed, _ in target_runs]
    speculative_latencies = [elapsed for _, elapsed, _ in speculative_runs]
    target_lengths = [int(result.tokens.shape[1]) for result in target_results]
    sequence_equalities = [
        torch.equal(target_result.tokens, speculative_result.tokens)
        for target_result, speculative_result in zip(
            target_results,
            speculative_results,
            strict=True,
        )
    ]
    equal_tokens = 0
    compared_tokens = 0
    mismatch_details: list[dict[str, Any]] = []
    for repeat, (target_result, speculative_result) in enumerate(
        zip(target_results, speculative_results, strict=True)
    ):
        target_ids = target_result.tokens[0].detach().cpu().tolist()
        speculative_ids = speculative_result.tokens[0].detach().cpu().tolist()
        equal_tokens += sum(
            target_id == speculative_id
            for target_id, speculative_id in zip(
                target_ids,
                speculative_ids,
                strict=False,
            )
        )
        compared_tokens += max(len(target_ids), len(speculative_ids))
        divergence = _first_divergence(target_result.tokens, speculative_result.tokens)
        if divergence is not None:
            mismatch_details.append(
                {
                    "repeat": repeat,
                    "first_divergence": divergence,
                    "target_tokens": target_ids,
                    "speculative_tokens": speculative_ids,
                    "cache_lengths": speculative_result.cache_lengths,
                }
            )

    proposed = sum(result.proposed for result in speculative_results)
    accepted = sum(result.accepted for result in speculative_results)
    accepted_counts = [
        count
        for result in speculative_results
        for count in result.accepted_per_iteration
    ]
    target_mean = statistics.mean(target_latencies)
    speculative_mean = statistics.mean(speculative_latencies)
    target_peak = [peak for _, _, peak in target_runs if peak is not None]
    speculative_peak = [peak for _, _, peak in speculative_runs if peak is not None]
    dtype = str(next(target.parameters()).dtype).replace("torch.", "")
    row = {
        "prompt_index": prompt_index,
        "input_length": int(input_ids.shape[1]),
        "max_new": max_new,
        "k": k,
        "target_model": target_label,
        "draft_model": draft_label,
        "dtype": dtype,
        "target_latency_s": _summary(target_latencies),
        "speculative_latency_s": _summary(speculative_latencies),
        "speedup": target_mean / speculative_mean if speculative_mean > 0 else None,
        "target_tokens_per_second": (
            statistics.mean(
                length / elapsed
                for length, elapsed in zip(target_lengths, target_latencies, strict=True)
                if elapsed > 0
            )
            if any(elapsed > 0 for elapsed in target_latencies)
            else None
        ),
        "speculative_tokens_per_second": (
            statistics.mean(
                int(result.tokens.shape[1]) / elapsed
                for result, elapsed in zip(
                    speculative_results,
                    speculative_latencies,
                    strict=True,
                )
                if elapsed > 0
            )
            if any(elapsed > 0 for elapsed in speculative_latencies)
            else None
        ),
        "sequence_equality_rate": sum(sequence_equalities) / len(sequence_equalities),
        "token_equality_rate": equal_tokens / compared_tokens if compared_tokens else 1.0,
        "acceptance_rate": accepted / proposed if proposed else None,
        "mean_accepted_per_iteration": (
            statistics.mean(accepted_counts) if accepted_counts else 0.0
        ),
        "target_forwards_per_output_token": statistics.mean(
            result.target_forwards / max(1, result.tokens.shape[1])
            for result in speculative_results
        ),
        "target_peak_memory_bytes": max(target_peak) if target_peak else None,
        "speculative_peak_memory_bytes": (
            max(speculative_peak) if speculative_peak else None
        ),
        "proposed": proposed,
        "accepted": accepted,
        "target_forwards": sum(result.target_forwards for result in speculative_results),
        "draft_forwards": sum(result.draft_forwards for result in speculative_results),
    }
    return row, mismatch_details


def run_benchmark(
    target: torch.nn.Module,
    draft: torch.nn.Module,
    prompts: list[torch.Tensor],
    *,
    target_label: str,
    draft_label: str,
    output_lengths: list[int],
    k_values: list[int],
    eos_id: int | None,
    vocab_size: int,
    input_source: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("At least one prompt is required")
    if not output_lengths or any(length <= 0 for length in output_lengths):
        raise ValueError("output_lengths must contain positive integers")
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("k_values must contain positive integers")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    device = _model_device(target)
    rows = []
    mismatches = []
    for prompt_index, prompt in enumerate(prompts):
        input_ids = prompt.to(device=device, dtype=torch.long)
        for output_length in output_lengths:
            for k in k_values:
                row, case_mismatches = benchmark_case(
                    target,
                    draft,
                    input_ids,
                    prompt_index=prompt_index,
                    max_new=output_length,
                    k=k,
                    eos_id=eos_id,
                    vocab_size=vocab_size,
                    warmup=warmup,
                    repeats=repeats,
                    target_label=target_label,
                    draft_label=draft_label,
                )
                rows.append(row)
                if case_mismatches:
                    mismatches.append(
                        {
                            "prompt_index": prompt_index,
                            "max_new": output_length,
                            "k": k,
                            "details": case_mismatches,
                        }
                    )
    return {
        "target_model": target_label,
        "draft_model": draft_label,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine() or "CPU"
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "warmup": warmup,
        "repeats": repeats,
        "prompt_count": len(prompts),
        "input_source": input_source,
        "results": rows,
        "mismatches": mismatches,
    }


def _load_model(
    spec: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if spec.get("type") == "tiny":
        torch.manual_seed(int(spec.get("seed", 42)))
        model = Qwen2ForCausalLM(Qwen2Config(**spec["tiny_config"]))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec["model_name_or_path"]),
            dtype=dtype,
        )
    return model.to(device=device, dtype=dtype).eval()


def _load_json(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config must be a YAML mapping: {path}")
    return config


def _write_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result,
    }
    (output_dir / "speculative.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "speculative.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in result["results"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in CSV_FIELDS},
                    "target_latency_s": json.dumps(row["target_latency_s"], sort_keys=True),
                    "speculative_latency_s": json.dumps(
                        row["speculative_latency_s"],
                        sort_keys=True,
                    ),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "speculative_bench.yaml",
    )
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--device")
    parser.add_argument("--prompt-source", choices=("val", "synthetic"), default="val")
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--input-length", type=int)
    parser.add_argument("--output-lengths", type=int, nargs="+")
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "bench")
    args = parser.parse_args()
    config = _load_json(args.config)
    target_label = args.model or str(config["target_model"])
    draft_label = args.draft or str(config["draft_model"])
    if target_label not in config["models"]:
        raise ValueError(f"Unknown target model {target_label!r}")
    if draft_label not in config["drafts"]:
        raise ValueError(f"Unknown draft model {draft_label!r}")
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for speculative benchmarking but is unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    target_spec = config["models"][target_label]
    draft_spec = config["drafts"][draft_label]
    target = _load_model(target_spec, device, dtype)
    draft = _load_model(draft_spec, device, dtype)
    eos_id = None
    shared_vocab_size = min(
        int(target.config.vocab_size),
        int(draft.config.vocab_size),
    )
    if args.prompt_source == "val":
        if target_spec.get("type") == "tiny" or draft_spec.get("type") == "tiny":
            raise ValueError("Validation prompts require real target and draft models")
        target_tokenizer = AutoTokenizer.from_pretrained(
            str(target_spec["tokenizer_name_or_path"]),
            use_fast=True,
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(draft_spec["tokenizer_name_or_path"]),
            use_fast=True,
        )
        tokenizer_vocab_size = validate_tokenizer_compatibility(
            target_tokenizer,
            draft_tokenizer,
        )
        if tokenizer_vocab_size > shared_vocab_size:
            raise ValueError(
                "The shared tokenizer vocabulary exceeds the target/draft model vocabulary"
            )
        shared_vocab_size = tokenizer_vocab_size
        prompts = load_validation_prompts(
            target_tokenizer,
            args.num_prompts or int(config["num_val_prompts"]),
        )
        eos_id = _is_eos_token(target_tokenizer)
    else:
        count = args.num_prompts or 1
        prompts = synthetic_prompts(
            count=count,
            length=args.input_length or int(config["synthetic_prompt_length"]),
            vocab_size=shared_vocab_size,
        )
    result = run_benchmark(
        target,
        draft,
        prompts,
        target_label=target_label,
        draft_label=draft_label,
        output_lengths=[
            int(length) for length in (args.output_lengths or config["output_lengths"])
        ],
        k_values=[int(value) for value in (args.k or config["k_values"])],
        eos_id=eos_id,
        vocab_size=shared_vocab_size,
        input_source="validation_split" if args.prompt_source == "val" else "synthetic",
        warmup=args.warmup if args.warmup is not None else int(config["warmup"]),
        repeats=args.repeats if args.repeats is not None else int(config["repeats"]),
    )
    _write_results(result, args.output_dir)
    LOGGER.info(
        "Wrote %d benchmark rows and %d mismatch cases to %s",
        len(result["results"]),
        len(result["mismatches"]),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

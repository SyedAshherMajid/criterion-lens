"""Benchmark greedy KV-cache decoding against full-prefix recomputation."""

import argparse
import csv
import json
import logging
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from criterion_lens.decoding.kv_cache import GenResult, greedy_kv_cache, greedy_no_cache
from criterion_lens.decoding.timing import Timer

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECODERS: dict[str, Callable[..., GenResult]] = {
    "no_cache": greedy_no_cache,
    "kv_cache": greedy_kv_cache,
}
DTYPES = {"float16": torch.float16, "float32": torch.float32}


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(samples),
        "std": statistics.pstdev(samples),
        "median": statistics.median(samples),
    }


def _token_ids(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    if vocab_size <= 3:
        raise ValueError(f"Model vocabulary is too small for synthetic inputs: {vocab_size}")
    return (torch.arange(length, device=device, dtype=torch.long) % (vocab_size - 3) + 3)[
        None, :
    ]


def _first_divergence(
    reference: GenResult,
    cached: GenResult,
) -> dict[str, Any] | None:
    reference_ids = reference.tokens[0].detach().cpu().tolist()
    cached_ids = cached.tokens[0].detach().cpu().tolist()
    for index, (reference_id, cached_id) in enumerate(
        zip(reference_ids, cached_ids, strict=True)
    ):
        if reference_id != cached_id:
            return {
                "step": index + 1,
                "no_cache_token_id": reference_id,
                "kv_cache_token_id": cached_id,
                "no_cache_logits_margin": reference.per_step[index]["logits_margin"],
                "kv_cache_logits_margin": cached.per_step[index]["logits_margin"],
            }
    return None


def _timed_generation(
    decoder: Callable[..., GenResult],
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    output_length: int,
    device: torch.device,
) -> tuple[GenResult, float, int | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    with Timer(device) as timer:
        result = decoder(model, input_ids, output_length, eos_id=None)
    if timer.elapsed_s is None:
        raise RuntimeError("Timer did not record an elapsed duration")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return result, timer.elapsed_s, peak_memory


def _path_metrics(
    results: list[tuple[GenResult, float, int | None]],
    output_length: int,
) -> dict[str, Any]:
    prefill_samples = [result.prefill_s for result, _, _ in results]
    decode_samples = [result.decode_s for result, _, _ in results]
    total_samples = [total_s for _, total_s, _ in results]
    output_tokens_per_second = [
        (output_length - 1) / result.decode_s
        for result, _, _ in results
        if result.decode_s > 0 and output_length > 1
    ]
    peak_samples = [peak for _, _, peak in results if peak is not None]
    return {
        "prefill_s": _summary(prefill_samples),
        "decode_s": _summary(decode_samples),
        "total_latency_s": _summary(total_samples),
        "output_tokens_per_second": (
            _summary(output_tokens_per_second) if output_tokens_per_second else None
        ),
        "time_per_output_token_s": _summary(
            [total_s / output_length for total_s in total_samples]
        ),
        "peak_memory_bytes": max(peak_samples) if peak_samples else None,
    }


def benchmark_model(
    model: torch.nn.Module,
    *,
    model_label: str,
    model_id: str,
    device: torch.device,
    dtype_name: str,
    input_lengths: list[int],
    output_lengths: list[int],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run precision parity checks and repeated timing on deterministic token IDs."""
    if not input_lengths or any(length <= 0 for length in input_lengths):
        raise ValueError("input_lengths must contain positive integers")
    if not output_lengths or any(length <= 0 for length in output_lengths):
        raise ValueError("output_lengths must contain positive integers")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    model.to(device=device, dtype=DTYPES[dtype_name])
    model.eval()
    vocab_size = int(model.config.vocab_size)
    parity_dtypes = ["float16", "float32"] if device.type == "cuda" else ["float32"]
    parity_checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for precision in parity_dtypes:
        model.to(dtype=DTYPES[precision])
        for input_length in input_lengths:
            input_ids = _token_ids(input_length, vocab_size, device)
            for output_length in output_lengths:
                reference = greedy_no_cache(model, input_ids, output_length)
                cached = greedy_kv_cache(model, input_ids, output_length)
                divergence = _first_divergence(reference, cached)
                check = {
                    "precision": precision,
                    "input_length": input_length,
                    "output_length": output_length,
                    "tokens_equal": divergence is None,
                }
                parity_checks.append(check)
                if divergence is not None:
                    mismatches.append(
                        {
                            **check,
                            "first_divergence": divergence,
                            "no_cache_tokens": reference.tokens[0].cpu().tolist(),
                            "kv_cache_tokens": cached.tokens[0].cpu().tolist(),
                            "no_cache_steps": reference.per_step,
                            "kv_cache_steps": cached.per_step,
                        }
                    )

    model.to(dtype=DTYPES[dtype_name])
    timed_rows: list[dict[str, Any]] = []
    for input_length in input_lengths:
        input_ids = _token_ids(input_length, vocab_size, device)
        for output_length in output_lengths:
            for decoder in DECODERS.values():
                for _ in range(warmup):
                    decoder(model, input_ids, output_length, eos_id=None)
            measurements: dict[str, list[tuple[GenResult, float, int | None]]] = {
                name: [] for name in DECODERS
            }
            for _ in range(repeats):
                for name, decoder in DECODERS.items():
                    measurements[name].append(
                        _timed_generation(
                            decoder,
                            model,
                            input_ids,
                            output_length,
                            device,
                        )
                    )
            no_cache_metrics = _path_metrics(measurements["no_cache"], output_length)
            kv_metrics = _path_metrics(measurements["kv_cache"], output_length)
            no_cache_total = no_cache_metrics["total_latency_s"]["mean"]
            kv_total = kv_metrics["total_latency_s"]["mean"]
            timed_rows.append(
                {
                    "input_length": input_length,
                    "output_length": output_length,
                    "dtype": dtype_name,
                    "no_cache": no_cache_metrics,
                    "kv_cache": kv_metrics,
                    "total_latency_speedup": (
                        no_cache_total / kv_total if kv_total > 0 else None
                    ),
                }
            )

    hardware = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine() or "CPU"
    )
    return {
        "model": model_label,
        "model_id": model_id,
        "device": str(device),
        "hardware": hardware,
        "dtype": dtype_name,
        "input_source": "deterministic_synthetic_token_ids",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "precision_checks": parity_checks,
        "mismatches": mismatches,
        "benchmarks": timed_rows,
        "warmup": warmup,
        "repeats": repeats,
    }


def _load_model(
    model_spec: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if model_spec.get("type") == "tiny":
        config = Qwen2Config(**model_spec["tiny_config"])
        model = Qwen2ForCausalLM(config)
        model_id = "tiny-random-qwen2"
    else:
        model_id = str(model_spec["model_name_or_path"])
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device=device, dtype=dtype), model_id


def _write_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result,
    }
    (output_dir / "kv_cache.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "kv_cache.csv"
    rows = result["benchmarks"]
    columns = [
        "input_length",
        "output_length",
        "dtype",
        "no_cache",
        "kv_cache",
        "total_latency_speedup",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "no_cache": json.dumps(row["no_cache"], sort_keys=True),
                    "kv_cache": json.dumps(row["kv_cache"], sort_keys=True),
                }
            )


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config must be a YAML mapping: {path}")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "bench.yaml")
    parser.add_argument("--model", default="1.5B", help="Model key from the config")
    parser.add_argument("--device", help="Device override, such as cuda or cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), help="Model dtype override")
    parser.add_argument("--input-lengths", type=int, nargs="+")
    parser.add_argument("--output-lengths", type=int, nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "bench")
    args = parser.parse_args()

    config = _load_config(args.config)
    models = config.get("models")
    if not isinstance(models, dict) or args.model not in models:
        raise ValueError(f"Model {args.model!r} is not defined in {args.config}")
    spec = models[args.model]
    if not isinstance(spec, dict):
        raise ValueError(f"Model config for {args.model!r} must be a mapping")
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for KV-cache benchmarking but is unavailable")
    configured_dtype = args.dtype or str(spec.get("dtype", "float32"))
    dtype_name = configured_dtype
    if device.type == "cpu" and args.dtype is None:
        dtype_name = "float32"
        if configured_dtype != "float32":
            LOGGER.info(
                "Using float32 for CPU benchmarking instead of configured %s",
                configured_dtype,
            )
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")

    model, model_id = _load_model(
        spec,
        dtype=DTYPES[dtype_name],
        device=device,
    )
    result = benchmark_model(
        model,
        model_label=args.model,
        model_id=model_id,
        device=device,
        dtype_name=dtype_name,
        input_lengths=[
            int(length) for length in (args.input_lengths or config["input_lengths"])
        ],
        output_lengths=[
            int(length) for length in (args.output_lengths or config["output_lengths"])
        ],
        warmup=args.warmup if args.warmup is not None else int(config["warmup"]),
        repeats=args.repeats if args.repeats is not None else int(config["repeats"]),
    )
    result["model_source"] = spec.get("model_name_or_path", "random tiny Qwen2")
    result["config_path"] = str(args.config)
    result["parity_dtypes"] = (
        ["float16", "float32"] if device.type == "cuda" else ["float32"]
    )
    _write_results(result, args.output_dir)
    LOGGER.info(
        "Wrote benchmark JSON and CSV to %s; %d precision checks, %d mismatches",
        args.output_dir,
        len(result["precision_checks"]),
        len(result["mismatches"]),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

"""Measure tiled attention accuracy, latency, and CUDA peak memory."""

import argparse
import csv
import logging
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as functional
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from criterion_lens.attention.reference import sdpa_reference
from criterion_lens.attention.tiled import tiled_attention
from criterion_lens.decoding.timing import Timer

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float16": torch.float16, "float32": torch.float32}
SDPA_BACKENDS = {
    "sdpa_math": SDPBackend.MATH,
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
}
ACCURACY_FIELDS = [
    "batch",
    "heads",
    "kv_heads",
    "query_length",
    "key_length",
    "head_dim",
    "causal",
    "dtype",
    "block_q",
    "block_k",
    "status",
    "max_abs_error",
    "mean_abs_error",
    "within_tolerance",
    "error",
]
PERFORMANCE_FIELDS = [
    "method",
    "sequence_length",
    "head_dim",
    "heads",
    "dtype",
    "causal",
    "status",
    "mean_s",
    "std_s",
    "median_s",
    "peak_memory_bytes",
    "error",
]


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or (
        "out of memory" in message or "can't allocate memory" in message
    )


def _is_backend_unavailable(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in ("no available kernel", "no viable backend", "no available backend")
    )


def _record_oom(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _random_qkv(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, key_length, head_dim = shape
    kv_heads = max(heads // 2, 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            batch, heads, query_length, head_dim, dtype=dtype, device=device, generator=generator
        ),
        torch.randn(
            batch, kv_heads, key_length, head_dim, dtype=dtype, device=device, generator=generator
        ),
        torch.randn(
            batch, kv_heads, key_length, head_dim, dtype=dtype, device=device, generator=generator
        ),
    )


def accuracy_rows(
    *,
    device: torch.device,
    shapes: list[tuple[int, int, int, int, int]],
    causal_values: list[bool],
    dtype_names: list[str],
    block_sizes: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for shape in shapes:
        batch, heads, query_length, key_length, head_dim = shape
        kv_heads = max(heads // 2, 1)
        for causal in causal_values:
            for dtype_name in dtype_names:
                dtype = DTYPES[dtype_name]
                q, k, v = _random_qkv(
                    shape,
                    dtype,
                    device,
                    seed=sum(shape) + int(causal) + len(dtype_name),
                )
                for block_size in block_sizes:
                    row: dict[str, Any] = {
                        "batch": batch,
                        "heads": heads,
                        "kv_heads": kv_heads,
                        "query_length": query_length,
                        "key_length": key_length,
                        "head_dim": head_dim,
                        "causal": causal,
                        "dtype": dtype_name,
                        "block_q": block_size,
                        "block_k": block_size,
                        "status": "OK",
                        "max_abs_error": None,
                        "mean_abs_error": None,
                        "within_tolerance": None,
                        "error": "",
                    }
                    try:
                        reference = sdpa_reference(
                            q.double(),
                            k.double(),
                            v.double(),
                            causal=causal,
                        )
                        actual = tiled_attention(
                            q,
                            k,
                            v,
                            causal=causal,
                            block_q=block_size,
                            block_k=block_size,
                        )
                        differences = (actual.double() - reference).abs()
                        tolerance = 1e-5 if dtype == torch.float32 else 2e-3
                        row["max_abs_error"] = float(differences.max().item())
                        row["mean_abs_error"] = float(differences.mean().item())
                        row["within_tolerance"] = bool(
                            torch.allclose(
                                actual.double(),
                                reference,
                                atol=tolerance,
                                rtol=tolerance,
                            )
                        )
                    except (MemoryError, RuntimeError) as error:
                        if not _is_oom(error):
                            raise
                        row["status"] = "OOM"
                        row["error"] = str(error)
                        _record_oom(device)
                    rows.append(row)
    return rows


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            output = function()
            del output
        samples: list[float] = []
        peak_memory = None
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            with Timer(device) as timer:
                output = function()
            if timer.elapsed_s is None:
                raise RuntimeError("Timer did not record an elapsed duration")
            samples.append(timer.elapsed_s)
            if device.type == "cuda":
                peak_memory = max(
                    peak_memory or 0,
                    int(torch.cuda.max_memory_allocated(device)),
                )
            del output
        return {
            "status": "OK",
            "mean_s": statistics.mean(samples),
            "std_s": statistics.pstdev(samples),
            "median_s": statistics.median(samples),
            "peak_memory_bytes": peak_memory,
            "error": "",
        }
    except (MemoryError, RuntimeError) as error:
        if _is_oom(error):
            _record_oom(device)
            return {
                "status": "OOM",
                "mean_s": None,
                "std_s": None,
                "median_s": None,
                "peak_memory_bytes": None,
                "error": str(error),
            }
        if allow_unavailable and _is_backend_unavailable(error):
            return {
                "status": "UNAVAILABLE",
                "mean_s": None,
                "std_s": None,
                "median_s": None,
                "peak_memory_bytes": None,
                "error": str(error),
            }
        raise


def performance_rows(
    *,
    device: torch.device,
    lengths: list[int],
    head_dims: list[int],
    heads: int,
    dtype_name: str,
    causal_values: list[bool],
    warmup: int,
    repeats: int,
    block_size: int = 64,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dtype = DTYPES[dtype_name]
    for length in lengths:
        for head_dim in head_dims:
            shape = (1, heads, length, length, head_dim)
            q, k, v = _random_qkv(
                shape,
                dtype,
                device,
                seed=length + head_dim + heads,
            )
            k = k.repeat_interleave(heads // k.shape[1], dim=1)
            v = v.repeat_interleave(heads // v.shape[1], dim=1)
            for causal in causal_values:
                methods: dict[str, tuple[Callable[[], torch.Tensor], bool]] = {
                    "reference": (
                        lambda q=q, k=k, v=v, causal=causal: sdpa_reference(
                            q, k, v, causal=causal
                        ),
                        False,
                    ),
                    "tiled": (
                        lambda q=q, k=k, v=v, causal=causal: tiled_attention(
                            q,
                            k,
                            v,
                            causal=causal,
                            block_q=block_size,
                            block_k=block_size,
                        ),
                        False,
                    ),
                }
                for method_name, backend in SDPA_BACKENDS.items():
                    def run_sdpa(
                        q: torch.Tensor = q,
                        k: torch.Tensor = k,
                        v: torch.Tensor = v,
                        causal: bool = causal,
                        backend: SDPBackend = backend,
                    ) -> torch.Tensor:
                        with sdpa_kernel(backends=[backend]):
                            return functional.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                is_causal=causal,
                            )

                    methods[method_name] = (run_sdpa, True)

                for method_name, (function, allow_unavailable) in methods.items():
                    measurement = _measure(
                        function,
                        device=device,
                        warmup=warmup,
                        repeats=repeats,
                        allow_unavailable=allow_unavailable,
                    )
                    rows.append(
                        {
                            "method": method_name,
                            "sequence_length": length,
                            "head_dim": head_dim,
                            "heads": heads,
                            "dtype": dtype_name,
                            "causal": causal,
                            **measurement,
                        }
                    )
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Attention benchmark config must be a YAML mapping: {path}")
    return config


def run_benchmark(
    *,
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
    warmup: int,
    repeats: int,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if quick:
        shapes = [(1, 1, 8, 8, 4), (1, 2, 1, 77, 64)]
        causal_values = [False, True]
        accuracy_dtypes = ["float32"]
        block_sizes = [16, 64]
        lengths, head_dims, heads = [32, 64], [16], 4
        performance_dtype = "float32"
    else:
        shapes = [tuple(int(value) for value in shape) for shape in config["accuracy_shapes"]]
        causal_values = [bool(value) for value in config["causal"]]
        accuracy_dtypes = list(config["accuracy_dtypes"])
        block_sizes = [int(value) for value in config["block_sizes"]]
        lengths = [int(value) for value in config["performance_lengths"]]
        head_dims = [int(value) for value in config["performance_head_dims"]]
        heads = int(config["performance_heads"])
        performance_dtype = "float16" if device.type == "cuda" else "float32"

    accuracy = accuracy_rows(
        device=device,
        shapes=shapes,
        causal_values=causal_values,
        dtype_names=accuracy_dtypes,
        block_sizes=block_sizes,
    )
    performance = performance_rows(
        device=device,
        lengths=lengths,
        head_dims=head_dims,
        heads=heads,
        dtype_name=performance_dtype,
        causal_values=causal_values if not quick else [False],
        warmup=warmup,
        repeats=repeats,
    )
    _write_csv(output_dir / "attention_accuracy.csv", ACCURACY_FIELDS, accuracy)
    _write_csv(output_dir / "attention_perf.csv", PERFORMANCE_FIELDS, performance)
    return accuracy, performance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "attention_bench.yaml",
    )
    parser.add_argument("--device", help="Device override, such as cuda or cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "bench")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--quick", action="store_true", help="Run a small non-reporting smoke grid")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for attention benchmarking but is unavailable")
    warmup = args.warmup if args.warmup is not None else int(config["warmup"])
    repeats = args.repeats if args.repeats is not None else int(config["repeats"])
    accuracy, performance = run_benchmark(
        device=device,
        config=config,
        output_dir=args.output_dir,
        warmup=warmup,
        repeats=repeats,
        quick=args.quick,
    )
    LOGGER.info(
        "Wrote accuracy rows=%d and performance rows=%d to %s",
        len(accuracy),
        len(performance),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

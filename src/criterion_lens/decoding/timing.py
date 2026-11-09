"""Synchronized timing and peak-memory helpers for decoding benchmarks."""

import time
from collections.abc import Callable
from statistics import mean, median, pstdev
from typing import Any

import torch


def _cuda_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else None
    resolved = torch.device(device)
    return resolved if resolved.type == "cuda" else None


def _synchronize(device: torch.device | str | None) -> torch.device | None:
    cuda_device = _cuda_device(device)
    if cuda_device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
        return cuda_device
    return None


class Timer:
    """Measure wall-clock duration, synchronizing CUDA work when applicable."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device
        self.elapsed_s: float | None = None
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        _synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        _synchronize(self.device)
        if self._start is None:
            raise RuntimeError("Timer was not started")
        self.elapsed_s = time.perf_counter() - self._start
        return False


def measure_peak_memory(
    function: Callable[[], Any],
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run a callable and report its result plus CUDA peak allocated bytes."""
    cuda_device = _cuda_device(device)
    if cuda_device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    result = function()
    peak_bytes = None
    if cuda_device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
        peak_bytes = int(torch.cuda.max_memory_allocated(cuda_device))
    return {"result": result, "peak_memory_bytes": peak_bytes}


def measure_repeated(
    function: Callable[[], Any],
    *,
    warmup: int = 2,
    repeats: int = 5,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Warm up, then return timing samples and their mean, population std, median."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    for _ in range(warmup):
        function()

    samples: list[float] = []
    for _ in range(repeats):
        with Timer(device) as timer:
            function()
        if timer.elapsed_s is None:
            raise RuntimeError("Timer did not record an elapsed duration")
        samples.append(timer.elapsed_s)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "mean_s": mean(samples),
        "std_s": pstdev(samples),
        "median_s": median(samples),
        "samples_s": samples,
    }

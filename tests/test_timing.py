import pytest
import torch

from criterion_lens.decoding.timing import Timer, measure_peak_memory, measure_repeated


def test_timer_records_duration_and_propagates_errors() -> None:
    with Timer("cpu") as timer:
        sum(range(100))
    assert timer.elapsed_s is not None
    assert timer.elapsed_s >= 0

    with pytest.raises(RuntimeError, match="expected"):
        with Timer("cpu"):
            raise RuntimeError("expected")


def test_measure_peak_memory_reports_result_without_cuda_for_cpu() -> None:
    measured = measure_peak_memory(lambda: 42, "cpu")

    assert measured == {"result": 42, "peak_memory_bytes": None}


def test_measure_repeated_runs_warmups_and_reports_summary() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    measured = measure_repeated(operation, warmup=2, repeats=3, device="cpu")

    assert calls == 5
    assert measured["warmup"] == 2
    assert measured["repeats"] == 3
    assert len(measured["samples_s"]) == 3
    assert measured["mean_s"] == pytest.approx(sum(measured["samples_s"]) / 3)
    assert measured["median_s"] >= 0
    assert measured["std_s"] >= 0


def test_measure_repeated_validates_run_counts() -> None:
    with pytest.raises(ValueError, match="warmup"):
        measure_repeated(lambda: None, warmup=-1)
    with pytest.raises(ValueError, match="repeats"):
        measure_repeated(lambda: None, repeats=0)


def test_timer_accepts_torch_device() -> None:
    with Timer(torch.device("cpu")) as timer:
        _ = torch.ones(1)
    assert timer.elapsed_s is not None

import csv

import torch

from scripts.bench_attention import run_benchmark


def test_quick_attention_benchmark_writes_accuracy_and_performance_csv(tmp_path) -> None:
    config = {"warmup": 0, "repeats": 1}

    accuracy, performance = run_benchmark(
        device=torch.device("cpu"),
        config=config,
        output_dir=tmp_path,
        warmup=0,
        repeats=1,
        quick=True,
    )

    with (tmp_path / "attention_accuracy.csv").open(newline="", encoding="utf-8") as file:
        accuracy_rows = list(csv.DictReader(file))
    with (tmp_path / "attention_perf.csv").open(newline="", encoding="utf-8") as file:
        performance_rows = list(csv.DictReader(file))
    assert len(accuracy_rows) == len(accuracy)
    assert len(performance_rows) == len(performance)
    assert all(row["status"] == "OK" for row in accuracy)
    assert {row["method"] for row in performance} == {
        "reference",
        "tiled",
        "sdpa_math",
        "sdpa_flash",
        "sdpa_efficient",
    }

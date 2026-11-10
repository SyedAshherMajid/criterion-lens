import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[1]
    / "kaggle"
    / "jobs"
    / "benchmarks"
    / "01_kaggle_benchmarks.ipynb"
)


def test_kaggle_benchmark_notebook_does_not_rerun_evaluation() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "scripts/evaluate.py" not in code
    assert "--split" not in code
    assert "--final" not in code
    assert "scripts/bench_kv.py" in code
    assert "scripts/bench_attention.py" in code
    assert "scripts/bench_speculative.py" in code
    assert "'--prompt-source', 'synthetic'" in code
    assert "'--prompt-source', 'val'" not in code
    for output_name in (
        "kv_1p5b",
        "kv_0p5b",
        "kv_tiny",
        "attention",
        "speculative_base_0p5b",
        "speculative_finetuned_0p5b",
    ):
        assert output_name in code

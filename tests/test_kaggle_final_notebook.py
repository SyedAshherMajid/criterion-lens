import ast
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[1]
    / "kaggle"
    / "jobs"
    / "eval_test_final"
    / "01_kaggle_eval_test_final.ipynb"
)


def test_final_kaggle_notebook_runs_only_the_four_sealed_test_models() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert code.count("'--split', 'test'") == 1
    assert code.count("'--final'") == 1
    assert code.count("scripts/evaluate.py") == 1
    assert "freeze_manifest.json" in code
    assert "Qwen/Qwen2.5-1.5B-Instruct" in code
    assert "Qwen/Qwen2.5-0.5B-Instruct" in code
    assert "criterion-lens-qwen2.5-1.5b-lora" in code
    assert "criterion-lens-qwen2.5-0.5b-lora" in code
    tree = ast.parse(code)
    runs_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "evaluation_runs"
            for target in node.targets
        )
    )
    runs = ast.literal_eval(runs_assignment.value)
    assert {name for name, _, _ in runs} == {
        "base_1p5b",
        "ft_1p5b",
        "base_0p5b",
        "ft_0p5b",
    }
    assert len(runs) == 4
    assert len({(model, adapter) for _, model, adapter in runs}) == 4

import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kaggle import sync_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_notebooks_are_valid_identical_and_secret_free() -> None:
    notebook_paths = (
        PROJECT_ROOT / "notebooks" / "01_kaggle_train.ipynb",
        PROJECT_ROOT / "kaggle" / "jobs" / "train" / "01_kaggle_train.ipynb",
    )
    notebook_texts = [path.read_text(encoding="utf-8") for path in notebook_paths]
    assert notebook_texts[0] == notebook_texts[1]

    notebook = json.loads(notebook_texts[0])
    code_cells = [
        cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert code_cells
    for source in code_cells:
        ast.parse("".join(source))
    combined_source = "\n".join("".join(source) for source in code_cells)
    assert "CUDA_VISIBLE_DEVICES" in combined_source
    assert "requirements_without_torch" in combined_source
    assert 'input_root.glob("**/requirements.txt")' in combined_source
    assert "/kaggle/input/criterion-lens-code" not in combined_source
    assert "HF_TOKEN" not in combined_source
    assert "kaggle_secrets" not in combined_source


def test_training_notebook_finds_dataset_at_nested_mount_path(tmp_path: Path) -> None:
    notebook = json.loads(
        (
            PROJECT_ROOT
            / "notebooks"
            / "01_kaggle_train.ipynb"
        ).read_text(encoding="utf-8")
    )
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "install-code")
    tree = ast.parse("".join(cell["source"]))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "locate_code_dataset"
    )
    namespace = {"Path": Path}
    compiled = compile(
        ast.Module(body=[function], type_ignores=[]),
        "<notebook-test>",
        "exec",
    )
    exec(compiled, namespace)

    dataset_root = tmp_path / "mount" / "owner" / "criterion-lens-code"
    dataset_root.mkdir(parents=True)
    (dataset_root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (dataset_root / "pyproject.toml").write_text("[project]\nname='code'\n", encoding="utf-8")
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        found = namespace["locate_code_dataset"](tmp_path / "mount")

    assert found == dataset_root
    assert str(dataset_root / "requirements.txt") in output.getvalue()


def test_training_notebook_requires_unique_code_dataset_root(tmp_path: Path) -> None:
    notebook = json.loads(
        (
            PROJECT_ROOT
            / "notebooks"
            / "01_kaggle_train.ipynb"
        ).read_text(encoding="utf-8")
    )
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "install-code")
    tree = ast.parse("".join(cell["source"]))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "locate_code_dataset"
    )
    namespace = {"Path": Path}
    compiled = compile(
        ast.Module(body=[function], type_ignores=[]),
        "<notebook-test>",
        "exec",
    )
    exec(compiled, namespace)

    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        (root / "requirements.txt").write_text("", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\nname='code'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Multiple Kaggle code datasets"):
        namespace["locate_code_dataset"](tmp_path)


def test_training_kernel_metadata_uses_private_code_dataset_and_t4_gpu() -> None:
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "kaggle"
            / "jobs"
            / "train"
            / "kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert metadata["id"] == "syedashhermajid/criterion-lens-train"
    assert metadata["is_private"]
    assert metadata["enable_gpu"]
    assert metadata["enable_internet"]
    assert metadata["dataset_sources"] == ["syedashhermajid/criterion-lens-code"]
    assert metadata["code_file"] == "01_kaggle_train.ipynb"


def test_kaggle_dataset_metadata_uses_cli_filename_and_private_id(tmp_path: Path) -> None:
    sync_code._write_metadata(tmp_path)

    metadata_path = tmp_path / "dataset-metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == sync_code.DATASET_ID
    assert metadata["isPrivate"]


def test_kaggle_dataset_existence_checks_exact_owned_ref(monkeypatch) -> None:
    monkeypatch.setattr(
        sync_code,
        "run_kaggle",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                [
                    {"ref": "syedashhermajid/unrelated-dataset"},
                    {"ref": sync_code.DATASET_ID},
                ]
            )
        ),
    )

    assert sync_code._dataset_exists()

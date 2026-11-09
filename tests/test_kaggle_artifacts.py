import ast
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert "HF_TOKEN" not in combined_source
    assert "kaggle_secrets" not in combined_source


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

    assert metadata["id"] == "syedashher/criterion-lens-train"
    assert metadata["is_private"]
    assert metadata["enable_gpu"]
    assert metadata["enable_internet"]
    assert metadata["dataset_sources"] == ["syedashher/criterion-lens-code"]
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
                    {"ref": "syedashher/unrelated-dataset"},
                    {"ref": sync_code.DATASET_ID},
                ]
            )
        ),
    )

    assert sync_code._dataset_exists()

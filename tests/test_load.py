from types import SimpleNamespace

import pandas as pd
import pytest

import criterion_lens.data.load as annotation_loader


class DatasetStub:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.column_names = list(frame.columns)

    def remove_columns(self, columns: list[str]) -> "DatasetStub":
        return DatasetStub(self.frame.drop(columns=columns))

    def to_pandas(self) -> pd.DataFrame:
        assert not annotation_loader.GPT4_COLUMNS.intersection(self.column_names)
        return self.frame.copy()


def _install_dataset_stub(monkeypatch, tmp_path, frame: pd.DataFrame) -> None:
    requested_revisions = []

    class ApiStub:
        def dataset_info(self, dataset_id: str, revision: str | None = None):
            assert dataset_id == annotation_loader.DATASET_ID
            return SimpleNamespace(sha="dataset-sha")

    def dataset_loader(dataset_id: str, split: str, revision: str):
        assert dataset_id == annotation_loader.DATASET_ID
        assert split == "train"
        requested_revisions.append(revision)
        return DatasetStub(frame)

    monkeypatch.setattr(annotation_loader, "HfApi", ApiStub)
    monkeypatch.setattr(annotation_loader, "load_dataset", dataset_loader)
    monkeypatch.setattr(annotation_loader, "CACHE_PATH", tmp_path / "annotations.parquet")
    return requested_revisions


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annotation_id": [1, 2],
            "patient_id": ["p1", "p2"],
            "note": ["0. First note.", "0. Second note."],
            "trial_id": ["NCT00000001", "NCT00000002"],
            "trial_title": ["Trial one", "Trial two"],
            "criterion_type": [" Inclusion ", "EXCLUSION"],
            "criterion_text": [" Has diagnosis ", "   "],
            "expert_eligibility": [" Included ", "EXCLUDED"],
            "expert_sentences": ["[0]", "[]"],
            "gpt4_explanation": ["not used", "not used"],
            "gpt4_sentences": ["[0]", "[]"],
            "gpt4_eligibility": ["included", "excluded"],
            "explanation_correctness": [True, True],
            "training": [True, False],
        }
    )


def test_load_annotations_drops_leakage_fields_before_conversion(
    monkeypatch, tmp_path
) -> None:
    revisions = _install_dataset_stub(monkeypatch, tmp_path, _sample_frame())

    frame = annotation_loader.load_annotations()

    assert revisions == ["dataset-sha"]
    assert frame.attrs["dataset_revision"] == "dataset-sha"
    assert len(frame) == 1
    assert frame.loc[0, "criterion_text"] == " Has diagnosis "
    assert frame.loc[0, "criterion_type"] == "inclusion"
    assert frame.loc[0, "expert_eligibility"] == "included"
    assert frame.loc[0, "expert_sentences"] == [0]
    assert not annotation_loader.GPT4_COLUMNS.intersection(frame.columns)
    assert (tmp_path / "annotations.parquet").is_file()


def test_load_annotations_rejects_malformed_evidence_ids(monkeypatch, tmp_path) -> None:
    frame = _sample_frame().iloc[:1].copy()
    frame.loc[0, "criterion_text"] = "Valid criterion"
    frame.loc[0, "expert_sentences"] = "[0, \"bad\"]"
    _install_dataset_stub(monkeypatch, tmp_path, frame)

    with pytest.raises(ValueError, match="Invalid expert_sentences"):
        annotation_loader.load_annotations()

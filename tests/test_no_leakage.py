from types import SimpleNamespace

import pandas as pd

import criterion_lens.data.load as annotation_loader
from criterion_lens.data.prompts import build_messages, format_target
from criterion_lens.data.sentences import parse_numbered_note

GPT4_SENTINEL = "GPT4_ONLY_LEAKAGE_SENTINEL"


class DatasetStub:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.column_names = list(frame.columns)

    def remove_columns(self, columns: list[str]) -> "DatasetStub":
        return DatasetStub(self.frame.drop(columns=columns))

    def to_pandas(self) -> pd.DataFrame:
        assert not annotation_loader.GPT4_COLUMNS.intersection(self.column_names)
        return self.frame.copy()


def test_gpt4_fields_and_training_flag_never_reach_prompt_or_target(
    monkeypatch, tmp_path
) -> None:
    source = pd.DataFrame(
        {
            "annotation_id": [1],
            "patient_id": ["synthetic-patient"],
            "note": ["0. Patient is 58 years old."],
            "trial_id": ["NCT00000001"],
            "trial_title": ["Synthetic trial"],
            "criterion_type": ["inclusion"],
            "criterion_text": ["Age over 18"],
            "expert_eligibility": ["included"],
            "expert_sentences": ["[0]"],
            "gpt4_explanation": [GPT4_SENTINEL],
            "gpt4_sentences": [GPT4_SENTINEL],
            "gpt4_eligibility": [GPT4_SENTINEL],
            "explanation_correctness": [GPT4_SENTINEL],
            "training": [False],
        }
    )
    monkeypatch.setattr(
        annotation_loader,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda dataset_id, revision=None: SimpleNamespace(sha="test-sha")
        ),
    )
    monkeypatch.setattr(
        annotation_loader,
        "load_dataset",
        lambda dataset_id, split, revision: DatasetStub(source),
    )
    monkeypatch.setattr(annotation_loader, "CACHE_PATH", tmp_path / "annotations.parquet")

    frame = annotation_loader.load_annotations()
    row = frame.iloc[0]
    messages = build_messages(
        parse_numbered_note(row["note"]),
        row["criterion_type"],
        row["criterion_text"],
    )
    target = format_target(row["expert_sentences"], row["expert_eligibility"])
    prompt_and_target = "\n".join(message["content"] for message in messages) + target

    assert not annotation_loader.GPT4_COLUMNS.intersection(frame.columns)
    assert GPT4_SENTINEL not in prompt_and_target
    assert row["expert_sentences"] == [0]

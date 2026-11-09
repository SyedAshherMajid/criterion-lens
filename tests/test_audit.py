import pandas as pd

from criterion_lens.data.audit import audit_annotations


class TokenizerStub:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        assert add_generation_prompt
        length = sum(len(message["content"].split()) for message in messages) + 1
        return list(range(length))

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return 2

    unk_token_id = 0


def _audit_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "annotation_id": [1, 2, 3],
            "patient_id": ["p1", "p1", "p2"],
            "note": [
                "0. Age is 58. 1. Diagnosis documented.",
                "0. Age is 58. 1. Diagnosis documented.",
                "0. No history is reported.",
            ],
            "trial_id": ["t1", "t1", "t2"],
            "criterion_type": ["inclusion", "exclusion", "exclusion"],
            "criterion_text": ["Age over 18", "No prior surgery", "No history"],
            "expert_eligibility": ["included", "included", "not enough information"],
            "expert_sentences": [[0], [8], []],
        }
    )
    frame.attrs["dataset_revision"] = "test-sha"
    frame.attrs["dropped_empty_criterion_count"] = 1
    return frame


def test_audit_reports_inconsistencies_evidence_ranges_and_lengths() -> None:
    manifest = {
        "train": {"patient_ids": ["p1"]},
        "val": {"patient_ids": ["p2"]},
        "test": {"patient_ids": []},
    }

    audit = audit_annotations(_audit_frame(), manifest, TokenizerStub())

    assert audit["dataset_revision"] == "test-sha"
    assert audit["n_annotations"] == 3
    assert audit["dropped_empty_criterion_count"] == 1
    assert audit["label_type_inconsistencies"]["count"] == 1
    assert audit["label_type_inconsistencies"]["annotation_ids"] == [2]
    assert audit["evidence"]["out_of_range_annotation_ids"] == [2]
    assert audit["evidence"]["out_of_range_sentence_id_count"] == 1
    assert audit["evidence"]["empty_evidence_by_label"]["not enough information"] == {
        "n_annotations": 1,
        "n_empty": 1,
        "empty_rate": 1.0,
    }
    assert audit["lengths"]["notes"]["sentences"]["count"] == 2
    assert audit["lengths"]["criterion_tokens"]["count"] == 3
    assert audit["lengths"]["prompt_and_target_tokens"]["overall"]["count"] == 3
    assert audit["lengths"]["prompt_and_target_tokens"]["by_split"]["train"]["count"] == 2
    assert audit["lengths"]["prompt_and_target_tokens"]["by_split"]["val"]["count"] == 1


def test_audit_marks_missing_label_or_type_as_inconsistent() -> None:
    frame = _audit_frame()
    frame.loc[0, "criterion_type"] = pd.NA
    manifest = {
        "train": {"patient_ids": ["p1"]},
        "val": {"patient_ids": ["p2"]},
        "test": {"patient_ids": []},
    }

    audit = audit_annotations(frame, manifest, TokenizerStub())

    assert audit["label_type_inconsistencies"]["count"] == 2

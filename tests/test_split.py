import numpy as np
import pandas as pd
import pytest

from criterion_lens.data.split import (
    build_split_manifest,
    make_split,
    split_manifest_rows,
)


def _annotations() -> pd.DataFrame:
    rows = []
    for patient_number in range(53):
        patient_id = f"patient-{patient_number:02d}"
        for annotation_number in range(2):
            rows.append(
                {
                    "annotation_id": patient_number * 2 + annotation_number,
                    "patient_id": patient_id,
                    "note": f"Unique note for {patient_id}.",
                    "criterion_type": "inclusion" if annotation_number == 0 else "exclusion",
                    "expert_eligibility": "included" if annotation_number == 0 else "excluded",
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["dataset_revision"] = "test-sha"
    return frame


def test_make_split_matches_fixed_seed_42_recipe() -> None:
    frame = _annotations()
    expected_ids = np.array(sorted(frame["patient_id"].unique()))
    np.random.default_rng(42).shuffle(expected_ids)

    split_ids = make_split(frame)

    assert split_ids["train"] == expected_ids[:37].tolist()
    assert split_ids["val"] == expected_ids[37:45].tolist()
    assert split_ids["test"] == expected_ids[45:53].tolist()
    assert [len(split_ids[name]) for name in ("train", "val", "test")] == [37, 8, 8]


def test_each_patient_annotation_and_note_stays_in_one_split() -> None:
    frame = _annotations()
    split_ids = make_split(frame)
    row_splits = pd.Series(index=frame.index, dtype="object")
    for split_name, patient_ids in split_ids.items():
        row_splits.loc[frame["patient_id"].isin(patient_ids)] = split_name

    assert row_splits.notna().all()
    assert frame.assign(split=row_splits).groupby("patient_id")["split"].nunique().eq(1).all()
    assert frame.assign(split=row_splits).groupby("note")["split"].nunique().eq(1).all()


def test_make_split_is_deterministic_and_requires_53_patients() -> None:
    frame = _annotations()

    assert make_split(frame) == make_split(frame.copy())
    with pytest.raises(ValueError, match="Expected 53 unique patients"):
        make_split(frame.loc[frame["patient_id"] != "patient-52"])


def test_manifest_and_csv_rows_recompute_exactly() -> None:
    frame = _annotations()
    split_ids = make_split(frame)

    manifest = build_split_manifest(frame, split_ids)
    repeated_manifest = build_split_manifest(frame, make_split(frame))
    rows = split_manifest_rows(frame, split_ids)

    assert manifest == repeated_manifest
    assert manifest["dataset_revision"] == "test-sha"
    assert [manifest[name]["n_patients"] for name in ("train", "val", "test")] == [37, 8, 8]
    assert sum(manifest[name]["n_annotations"] for name in ("train", "val", "test")) == len(
        frame
    )
    assert manifest["train"]["label_counts"] == {"excluded": 37, "included": 37}
    assert len(rows) == 53
    assert all(row["n_annotations"] == 2 for row in rows)


def test_manifest_rejects_overlapping_or_incomplete_patient_ids() -> None:
    frame = _annotations()
    split_ids = make_split(frame)
    split_ids["val"][0] = split_ids["train"][0]

    with pytest.raises(ValueError, match="cover the dataset exactly"):
        build_split_manifest(frame, split_ids)

"""Create and describe the fixed patient-grouped data split."""

from typing import Any

import numpy as np
import pandas as pd

SEED = 42
SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}
SPLIT_METHOD = (
    'ids = np.array(sorted(df["patient_id"].unique())); '
    "rng = np.random.default_rng(42); rng.shuffle(ids); "
    "first 37 -> train, next 8 -> val, last 8 -> test"
)


def make_split(df: pd.DataFrame) -> dict[str, list[str]]:
    """Assign the 53 patient IDs using the fixed seed-42 recipe.

    Splitting patient IDs before selecting rows prevents annotations from the
    same note-holder from crossing train, validation, and test.
    """
    if df["patient_id"].isna().any():
        raise ValueError("patient_id contains null values")

    ids = np.array(sorted(df["patient_id"].unique()))
    if len(ids) != sum(SPLIT_SIZES.values()):
        raise ValueError(
            f"Expected 53 unique patients for the fixed split, found {len(ids)}"
        )

    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    train_ids = ids[:37].tolist()
    val_ids = ids[37:45].tolist()
    test_ids = ids[45:53].tolist()

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_split_manifest(
    df: pd.DataFrame, split_ids: dict[str, list[str]]
) -> dict[str, Any]:
    """Build the published split manifest from IDs and expert annotations."""
    if set(split_ids) != set(SPLIT_SIZES):
        raise ValueError(f"Split names must be exactly {sorted(SPLIT_SIZES)}")

    all_patients = set(df["patient_id"].unique())
    split_patient_sets = {name: set(ids) for name, ids in split_ids.items()}
    if any(len(split_patient_sets[name]) != expected for name, expected in SPLIT_SIZES.items()):
        raise ValueError("Split patient counts do not match the fixed 37/8/8 recipe")
    if set.union(*split_patient_sets.values()) != all_patients:
        raise ValueError("Split patient IDs do not cover the dataset exactly")
    if sum(map(len, split_patient_sets.values())) != len(all_patients):
        raise ValueError("Patient IDs overlap across splits")

    dataset_revision = df.attrs.get("dataset_revision")
    if not dataset_revision:
        raise ValueError("DataFrame is missing dataset_revision metadata")

    manifest: dict[str, Any] = {
        "seed": SEED,
        "method": SPLIT_METHOD,
        "dataset_revision": dataset_revision,
    }
    for split_name, patient_ids in split_ids.items():
        subset = df.loc[df["patient_id"].isin(patient_ids)]
        label_counts = (
            subset["expert_eligibility"]
            .astype("string")
            .fillna("<missing>")
            .value_counts()
            .sort_index()
        )
        manifest[split_name] = {
            "patient_ids": patient_ids,
            "n_patients": len(patient_ids),
            "n_annotations": len(subset),
            "n_inclusion": int((subset["criterion_type"] == "inclusion").sum()),
            "n_exclusion": int((subset["criterion_type"] == "exclusion").sum()),
            "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        }
    return manifest


def split_manifest_rows(
    df: pd.DataFrame, split_ids: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Return one CSV-ready row for each patient without exposing note text."""
    annotation_counts = df["patient_id"].value_counts()
    return [
        {
            "patient_id": patient_id,
            "split": split_name,
            "n_annotations": int(annotation_counts[patient_id]),
        }
        for split_name, patient_ids in split_ids.items()
        for patient_id in patient_ids
    ]

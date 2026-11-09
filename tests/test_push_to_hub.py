from typing import Any

from scripts.push_to_hub import build_model_card


def test_model_card_has_training_provenance_and_no_invented_metrics() -> None:
    config: dict[str, Any] = {
        "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_revision": "fixed-sha",
        "max_seq_len": 640,
        "num_train_epochs": 3,
        "lora": {"r": 16, "lora_alpha": 32},
    }
    trainable = {"trainable_parameters": 18_000_000, "total_parameters": 1_500_000_000}
    split_manifest = {
        "train": {"n_patients": 37, "n_annotations": 602},
        "val": {"n_patients": 8, "n_annotations": 206},
        "test": {"n_patients": 8, "n_annotations": 206},
    }

    card = build_model_card(config, trainable, split_manifest)

    assert "Validation macro-F1: **TBD**" in card
    assert "No test-set metrics are included" in card
    assert "fixed-sha" in card
    assert "| train | 37 | 602 |" in card

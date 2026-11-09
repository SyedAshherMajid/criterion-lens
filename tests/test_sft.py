from pathlib import Path

import pytest

from criterion_lens.training.sft import build_training_arguments, load_training_config


def _config() -> dict[str, object]:
    return {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "num_train_epochs": 3,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "seed": 42,
        "group_by_length": True,
        "gradient_checkpointing": True,
        "fp16": True,
    }


def test_training_arguments_match_recipe_and_cpu_smoke() -> None:
    args = build_training_arguments(
        _config(),
        Path("outputs/smoke"),
        max_steps=5,
        cuda_available=False,
    )

    assert args.do_train
    assert args.do_eval
    assert args.max_steps == 5
    assert args.eval_strategy == "steps"
    assert args.eval_steps == 5
    assert args.save_strategy == "steps"
    assert args.save_steps == 5
    assert args.logging_steps == 5
    assert args.use_cpu
    assert not args.fp16
    assert args.report_to == []


def test_training_arguments_reject_nonpositive_smoke_steps() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_training_arguments(_config(), Path("outputs/smoke"), max_steps=0)


def test_full_training_arguments_evaluate_and_save_each_epoch() -> None:
    args = build_training_arguments(
        _config(),
        Path("outputs/full-run"),
        cuda_available=False,
    )

    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
    assert args.max_grad_norm == 1.0
    assert args.learning_rate == 2e-4
    assert args.gradient_checkpointing


def test_load_training_config_requires_a_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_training_config(config_file)

"""Supervised fine-tuning with completion-only loss and LoRA."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from criterion_lens.data.load import load_annotations
from criterion_lens.data.tokenize import encode_example
from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_training_config(path: Path) -> dict[str, Any]:
    """Read a YAML training config and require a mapping at its root."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a YAML mapping: {path}")
    return config


def build_training_arguments(
    config: Mapping[str, Any],
    output_dir: Path,
    max_steps: int | None = None,
    cuda_available: bool | None = None,
) -> TrainingArguments:
    """Create version-compatible Trainer arguments for a full run or short smoke test."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps override must be positive")
    has_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    smoke_test = max_steps is not None
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps" if smoke_test else config.get("eval_strategy", "epoch"),
        eval_steps=max_steps if smoke_test else None,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=max_steps if smoke_test else -1,
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        save_strategy="steps" if smoke_test else config.get("save_strategy", "epoch"),
        save_steps=max_steps if smoke_test else 500,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        group_by_length=bool(config["group_by_length"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=bool(config["fp16"]) and has_cuda,
        use_cpu=not has_cuda,
        report_to="none",
        save_safetensors=True,
        load_best_model_at_end=False,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _encode_split(
    frame: Any,
    tokenizer: Any,
    max_seq_len: int,
    allow_truncation: bool,
    truncation_log: list[dict[str, Any]] | None = None,
) -> Dataset:
    encoded = [
        encode_example(
            tokenizer,
            row,
            max_seq_len,
            truncation_log=truncation_log,
            allow_truncation=allow_truncation,
        )
        for row in frame.to_dict("records")
    ]
    return Dataset.from_list(encoded)


def run_training(
    config_path: Path,
    model_override: str | None = None,
    max_steps: int | None = None,
    run_name_override: str | None = None,
    force_cpu: bool = False,
) -> Path:
    """Train a LoRA adapter on train patients and evaluate loss on validation only."""
    load_env()
    if not torch.cuda.is_available() and max_steps is None:
        raise RuntimeError("A CUDA GPU is required for full training; CPU is for smoke tests only")
    config = load_training_config(config_path)
    set_seed(int(config["seed"]))
    model_name = model_override or str(config["model_name_or_path"])
    tokenizer_name = str(config.get("tokenizer_name_or_path") or model_name)
    run_name = run_name_override or str(config["run_name"])
    if Path(run_name).name != run_name:
        raise ValueError("run_name must be a single path component")

    manifest_path = PROJECT_ROOT / str(config["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = load_annotations(revision=manifest["dataset_revision"])
    train_patient_ids = set(manifest["train"]["patient_ids"])
    val_patient_ids = set(manifest["val"]["patient_ids"])
    if train_patient_ids & val_patient_ids:
        raise ValueError("Training and validation patients overlap")
    train_frame = frame.loc[frame["patient_id"].isin(train_patient_ids)]
    val_frame = frame.loc[frame["patient_id"].isin(val_patient_ids)]
    for split_name, split_frame in (("train", train_frame), ("val", val_frame)):
        expected = int(manifest[split_name]["n_annotations"])
        if len(split_frame) != expected:
            raise ValueError(
                f"{split_name} split has {len(split_frame)} rows; manifest expects {expected}"
            )
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and validation splits must contain annotations")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer {tokenizer_name} has no chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer {tokenizer_name} has no padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = int(config["max_seq_len"])
    train_truncations: list[dict[str, Any]] = []
    train_dataset = _encode_split(
        train_frame, tokenizer, max_seq_len, allow_truncation=True,
        truncation_log=train_truncations,
    )
    eval_dataset = _encode_split(
        val_frame, tokenizer, max_seq_len, allow_truncation=False
    )

    output_root = PROJECT_ROOT / str(config["output_root"])
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {run_dir}; choose a different run name"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available() and not force_cpu
    configured_dtype = str(config["torch_dtype"])
    if configured_dtype not in _TORCH_DTYPES:
        raise ValueError(f"Unsupported torch_dtype: {configured_dtype}")
    dtype_name = configured_dtype if has_cuda else "float32"
    if not has_cuda and configured_dtype != "float32":
        LOGGER.warning("CUDA is unavailable; loading the model in float32 for CPU training")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_TORCH_DTYPES[dtype_name],
    )
    model_vocab_size = model.get_input_embeddings().num_embeddings
    max_token_id = max(tokenizer.get_vocab().values())
    if model_vocab_size <= max_token_id:
        raise ValueError(
            f"Model vocabulary size {model_vocab_size} cannot represent tokenizer "
            f"token ID {max_token_id}"
        )

    lora_config = config["lora"]
    target_modules = list(lora_config["target_modules"])
    available_modules = [name for name, _ in model.named_modules()]
    missing_modules = [
        target
        for target in target_modules
        if not any(name.endswith(target) for name in available_modules)
    ]
    if missing_modules:
        raise ValueError(
            "Model is missing configured LoRA target modules: "
            + ", ".join(missing_modules)
        )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["lora_alpha"]),
            lora_dropout=float(lora_config["lora_dropout"]),
            target_modules=target_modules,
            bias=str(lora_config["bias"]),
        ),
    )
    if bool(config["gradient_checkpointing"]):
        model.enable_input_require_grads()

    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    _write_json(
        run_dir / "trainable_params.json",
        {
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
    )
    resolved_config = {
        **config,
        "model_name_or_path": model_name,
        "tokenizer_name_or_path": tokenizer_name,
        "dataset_revision": manifest["dataset_revision"],
        "train_annotations": len(train_frame),
        "validation_annotations": len(val_frame),
        "torch_dtype_used": dtype_name,
        "max_steps_override": max_steps,
    }
    _write_json(run_dir / "train_config.json", resolved_config)

    training_args = build_training_arguments(
        config, run_dir, max_steps=max_steps, cuda_available=has_cuda
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    model.config.use_cache = True
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(run_dir)
    trainer.save_metrics("train", train_result.metrics)
    _write_json(run_dir / "log_history.json", trainer.state.log_history)
    _write_json(run_dir / "truncation_log.json", train_truncations)
    if train_truncations:
        _write_json(PROJECT_ROOT / "results" / "truncation_log.json", train_truncations)

    LOGGER.info("Training complete; adapter and logs saved to %s", run_dir)
    return run_dir

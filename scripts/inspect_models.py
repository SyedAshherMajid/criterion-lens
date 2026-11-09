"""Inspect model configs and lightweight Hub metadata without downloading weights."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoConfig, PretrainedConfig

from criterion_lens.utils.env import load_env

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen3-1.7B",
]
LOGGER = logging.getLogger(__name__)


def config_fields(model_id: str, config: PretrainedConfig) -> dict[str, Any]:
    """Extract architecture facts needed to compare models without loading weights."""
    query_heads = getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_size is not None and query_heads:
        head_dim = hidden_size // query_heads

    return {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "parameters": None,
        "parameter_count_source": None,
        "license": None,
        "status": "config_loaded",
        "layers": getattr(config, "num_hidden_layers", None),
        "query_heads": query_heads,
        "kv_heads": getattr(config, "num_key_value_heads", query_heads),
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "context_length": getattr(
            config,
            "max_position_embeddings",
            getattr(config, "max_sequence_length", None),
        ),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
    }


def inspect_model(model_id: str, api: HfApi) -> dict[str, Any]:
    """Read one model config and non-weight metadata, recording access failures explicitly."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except (HfHubHTTPError, OSError, ValueError) as error:
        LOGGER.warning("Could not read config for %s (%s)", model_id, type(error).__name__)
        return {
            "model_id": model_id,
            "status": "unavailable",
            "error_type": type(error).__name__,
        }

    spec = config_fields(model_id, config)
    try:
        info = api.model_info(model_id, files_metadata=False)
    except (HfHubHTTPError, OSError, ValueError) as error:
        spec["status"] = "config_only"
        spec["metadata_error_type"] = type(error).__name__
        LOGGER.warning("Hub metadata unavailable for %s (%s)", model_id, type(error).__name__)
        return spec

    parameter_count = getattr(getattr(info, "safetensors", None), "total", None)
    card_data = getattr(info, "card_data", None)
    license_name = getattr(card_data, "license", None)
    spec["parameters"] = parameter_count
    spec["parameter_count_source"] = (
        "Hugging Face safetensors metadata" if parameter_count is not None else None
    )
    spec["license"] = license_name
    spec["status"] = "complete" if parameter_count is not None else "partial"
    return spec


def parse_args() -> argparse.Namespace:
    """Parse model IDs and the destination for the generated specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=Path("results/model_specs.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Inspect candidate configs, persist JSON, and return nonzero if none were readable."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(args.seed)
    load_env()

    api = HfApi()
    specs = [inspect_model(model_id, api) for model_id in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")

    for spec in specs:
        LOGGER.info(
            "%s: status=%s layers=%s q_heads=%s kv_heads=%s head_dim=%s "
            "vocab=%s context=%s parameters=%s",
            spec["model_id"],
            spec["status"],
            spec.get("layers"),
            spec.get("query_heads"),
            spec.get("kv_heads"),
            spec.get("head_dim"),
            spec.get("vocab_size"),
            spec.get("context_length"),
            spec.get("parameters"),
        )
    LOGGER.info("Wrote model specs to %s", args.output)
    return 0 if any(spec["status"] != "unavailable" for spec in specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

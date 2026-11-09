"""Fine-tune a causal language model with LoRA."""

import argparse
import logging
from pathlib import Path

from criterion_lens.training.sft import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_qwen25_1p5b.yaml"),
    )
    parser.add_argument("--model", help="Override the model ID or local model path")
    parser.add_argument("--max_steps", type=int, help="Short smoke run; must be positive")
    parser.add_argument("--run_name", help="Override the configured output run name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training for smoke tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training(
        args.config,
        model_override=args.model,
        max_steps=args.max_steps,
        run_name_override=args.run_name,
        force_cpu=args.cpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

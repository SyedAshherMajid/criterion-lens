"""Hugging Face Spaces entry point for CriterionLens."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from app.ui import build_app

    build_app().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

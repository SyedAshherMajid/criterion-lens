"""Start the CriterionLens Gradio application."""

import argparse

from app.ui import build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local CriterionLens app")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link for remote access",
    )
    args = parser.parse_args()
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=args.share,
    )


if __name__ == "__main__":
    main()

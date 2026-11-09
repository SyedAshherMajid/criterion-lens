"""Start the CriterionLens Gradio application."""

from app.ui import build_app


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

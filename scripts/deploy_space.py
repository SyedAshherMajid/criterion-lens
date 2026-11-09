"""Deploy the CriterionLens Gradio app to its Hugging Face Space."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "Ashher123/criterion-lens"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".env", "*.bin", "*.safetensors")


def prepare_space_payload(payload_dir: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Assemble only the app and its runtime package into an isolated upload folder."""
    if payload_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Space staging directory: {payload_dir}")
    space_dir = project_root / "space"
    payload_dir.mkdir(parents=True)
    for filename in ("README.md", "requirements.txt", "app.py"):
        source = space_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required Space file is missing: {source}")
        shutil.copy2(source, payload_dir / filename)

    app_source = project_root / "app"
    package_source = project_root / "src" / "criterion_lens"
    if not app_source.is_dir() or not package_source.is_dir():
        raise FileNotFoundError("The app package or criterion_lens source package is missing")
    shutil.copytree(app_source, payload_dir / "app", ignore=_IGNORE)
    shutil.copytree(
        package_source,
        payload_dir / "src" / "criterion_lens",
        ignore=_IGNORE,
    )
    return payload_dir


def deploy_space() -> str:
    """Create/update the public Space and upload a minimal, secret-free payload."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; load a write-enabled token before deployment")

    api = HfApi(token=token)
    identity = api.whoami()
    owner = SPACE_ID.split("/", maxsplit=1)[0]
    if str(identity["name"]).casefold() != owner.casefold():
        raise ValueError(f"Authenticated Hugging Face user does not own Space {SPACE_ID}")

    with tempfile.TemporaryDirectory(prefix="criterion-lens-space-") as temporary:
        payload_dir = prepare_space_payload(Path(temporary) / "payload")
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="cpu-basic",
            private=False,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=SPACE_ID,
            repo_type="space",
            folder_path=payload_dir,
            ignore_patterns=[".env", "**/.env", "__pycache__", "*.pyc"],
            commit_message="Deploy CriterionLens Gradio application",
        )
    LOGGER.info("Deployed Space %s at commit %s", SPACE_ID, commit.oid)
    return commit.oid


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    deploy_space()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

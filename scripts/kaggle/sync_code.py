"""Publish the allowlisted project source as the private Kaggle code dataset."""

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

if __package__:
    from .kaggle_env import run_kaggle
else:
    from kaggle_env import run_kaggle

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "syedashhermajid/criterion-lens-code"
_DIRECTORIES = ("src", "scripts", "configs", "data/splits")
_FILES = ("requirements.txt", "pyproject.toml", "README.md", "LICENSE")
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".venv"
)


def _copy_code(destination: Path) -> None:
    for relative_path in _DIRECTORIES:
        source = PROJECT_ROOT / relative_path
        if not source.is_dir():
            raise FileNotFoundError(f"Required code directory is missing: {source}")
        shutil.copytree(source, destination / relative_path, ignore=_IGNORE)
    for relative_path in _FILES:
        source = PROJECT_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Required project file is missing: {source}")
        shutil.copy2(source, destination / relative_path)


def _write_metadata(destination: Path) -> None:
    metadata = {
        "title": "CriterionLens Code",
        "id": DATASET_ID,
        "subtitle": "Reproducible source for CriterionLens training and evaluation",
        "description": (
            "Allowlisted source and fixed data split manifest; no secrets or model weights."
        ),
        "isPrivate": True,
        "licenses": [{"name": "MIT"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _dataset_exists() -> bool:
    result = run_kaggle(
        ["datasets", "list", "--mine", "--page-size", "200", "--format", "json"],
        capture_output=True,
    )
    return any(entry["ref"] == DATASET_ID for entry in json.loads(result.stdout))


def _wait_until_ready(timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = run_kaggle(
            ["datasets", "status", DATASET_ID],
            check=False,
            capture_output=True,
        )
        output = (status.stdout or "").strip()
        error = (status.stderr or "").strip()
        if status.returncode:
            raise subprocess.CalledProcessError(
                status.returncode,
                ["kaggle", "datasets", "status", DATASET_ID],
                output=output,
                stderr=error,
            )
        LOGGER.info("Kaggle dataset status: %s", output)
        normalized = output.casefold()
        if "not ready" not in normalized and (
            "ready" in normalized or "complete" in normalized
        ):
            return
        if any(state in normalized for state in ("failed", "error")):
            raise RuntimeError("Kaggle dataset creation/versioning failed: " + output)
        if time.monotonic() >= deadline:
            raise TimeoutError("Kaggle code dataset did not become ready in time")
        time.sleep(30)


def sync_code(wait_seconds: int = 600) -> Path:
    """Stage an allowlisted snapshot, then create or version the private dataset."""
    build_dir = PROJECT_ROOT / "build"
    owner = DATASET_ID.split("/", maxsplit=1)[0]
    destination = build_dir / f"kaggle_code_{owner}"
    marker = build_dir / f".criterionlens-kaggle-stage-{owner}"
    build_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8") != DATASET_ID:
            raise FileExistsError(
                f"Refusing to overwrite an unmarked staging directory: {destination}"
            )
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    marker.write_text(DATASET_ID, encoding="utf-8")
    _copy_code(destination)
    _write_metadata(destination)
    if any(path.name == ".env" for path in destination.rglob("*")):
        raise RuntimeError("Refusing to publish a staging directory containing .env")

    if _dataset_exists():
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(destination.relative_to(PROJECT_ROOT)),
                "-m",
                commit,
                "--dir-mode",
                "zip",
            ],
            cwd=PROJECT_ROOT,
        )
    else:
        run_kaggle(
            [
                "datasets",
                "create",
                "-p",
                str(destination.relative_to(PROJECT_ROOT)),
                "--dir-mode",
                "zip",
            ],
            cwd=PROJECT_ROOT,
        )
    _wait_until_ready(wait_seconds)
    LOGGER.info("Private Kaggle code dataset is ready: %s", DATASET_ID)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sync_code(wait_seconds=args.wait_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

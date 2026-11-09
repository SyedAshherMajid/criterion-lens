"""Run Kaggle CLI commands with credentials loaded without displaying them."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from criterion_lens.utils.env import load_env


def kaggle_executable() -> str:
    local_executable = Path.home() / ".local" / "bin" / "kaggle.exe"
    if local_executable.is_file():
        return str(local_executable)
    executable = shutil.which("kaggle")
    if executable:
        return executable
    raise FileNotFoundError("Kaggle CLI executable was not found")


def run_kaggle(
    arguments: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Kaggle CLI command using the local .env token in its environment."""
    load_env(override_keys={"KAGGLE_API_TOKEN", "KAGGLE_USERNAME"})
    missing = [
        key
        for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME")
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError("Missing Kaggle environment variables: " + ", ".join(missing))

    command = [kaggle_executable(), *arguments]
    result = subprocess.run(
        command,
        check=False,
        capture_output=capture_output,
        env=os.environ.copy(),
        cwd=cwd,
        text=True,
    )
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def main() -> int:
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("Usage: kaggle_env.py <kaggle arguments...>")
    return run_kaggle(arguments).returncode


if __name__ == "__main__":
    raise SystemExit(main())

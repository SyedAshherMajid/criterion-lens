"""Load local environment variables without exposing their values."""

import logging
import os
import re
from collections.abc import Collection
from pathlib import Path

logger = logging.getLogger(__name__)
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def load_env(
    env_path: Path | None = None,
    *,
    override_keys: Collection[str] = (),
) -> int:
    """Load unset variables from the project .env file and return the number loaded.

    Existing process variables take precedence so deployment secrets cannot be
    unintentionally replaced by a local file, except for explicitly selected keys.
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if not env_path.is_file():
            logger.info("No local environment file at %s; skipping .env loading", env_path)
            return 0
    elif not env_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    loaded = 0
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid .env entry at {env_path}:{line_number}; expected KEY=VALUE"
            )

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"Invalid .env key at {env_path}:{line_number}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unmatched quote in .env value at {env_path}:{line_number}")
            value = value[1:-1]

        if key not in os.environ or key in override_keys:
            os.environ[key] = value
            loaded += 1

    return loaded

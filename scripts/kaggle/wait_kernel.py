"""Check or wait for a Kaggle kernel without polling more often than requested."""

import argparse
import logging
import subprocess
import sys
import time

from kaggle_env import run_kaggle

LOGGER = logging.getLogger(__name__)
_TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


def kernel_status(kernel_id: str) -> tuple[str, str]:
    result = run_kaggle(
        ["kernels", "status", kernel_id],
        capture_output=True,
    )
    output = (result.stdout or "").strip()
    state = output.splitlines()[-1].strip().split(".")[-1].upper()
    return state, output


def wait_kernel(
    kernel_id: str,
    interval_seconds: int = 300,
    max_wait_minutes: int = 45,
    once: bool = False,
) -> int:
    """Poll kernel status and signal the documented queue fallback on timeout."""
    if interval_seconds < 300:
        raise ValueError("Kaggle status checks must be at least 300 seconds apart")
    if max_wait_minutes <= 0:
        raise ValueError("max_wait_minutes must be positive")

    deadline = time.monotonic() + max_wait_minutes * 60
    while True:
        state, output = kernel_status(kernel_id)
        LOGGER.info("%s", output)
        if state in _TERMINAL_STATES:
            return 0 if state == "COMPLETE" else 2
        if once:
            return 0
        if time.monotonic() >= deadline:
            if state == "QUEUED":
                LOGGER.warning(
                    "Kernel remained QUEUED for %d minutes; continue laptop work and "
                    "use the documented local fallback if appropriate.",
                    max_wait_minutes,
                )
            return 1
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-wait-minutes", type=int, default=45)
    parser.add_argument("--once", action="store_true", help="Report one status and return")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return wait_kernel(
            args.kernel,
            interval_seconds=args.interval_seconds,
            max_wait_minutes=args.max_wait_minutes,
            once=args.once,
        )
    except subprocess.CalledProcessError as error:
        LOGGER.error("Kaggle status command failed with exit code %d", error.returncode)
        return error.returncode or 1


if __name__ == "__main__":
    sys.exit(main())

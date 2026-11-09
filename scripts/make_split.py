"""Write the fixed patient-level split manifest."""

import argparse
import csv
import json
import logging
from pathlib import Path

from criterion_lens.data.load import load_annotations
from criterion_lens.data.split import (
    build_split_manifest,
    make_split,
    split_manifest_rows,
)
from criterion_lens.utils.env import load_env

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the optional dataset revision and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Dataset revision or commit SHA to resolve")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    return parser.parse_args()


def main() -> int:
    """Load sanitized annotations and write reproducible JSON and CSV manifests."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    frame = load_annotations(revision=args.revision)
    split_ids = make_split(frame)
    manifest = build_split_manifest(frame, split_ids)
    rows = split_manifest_rows(frame, split_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "split_manifest.json"
    csv_path = args.output_dir / "split_manifest.csv"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["patient_id", "split", "n_annotations"])
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "Wrote 37/8/8 patient split for revision %s to %s and %s",
        manifest["dataset_revision"],
        json_path,
        csv_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare the pinned BANKING77 dataset and emit a text-free split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from governed_banking.data import acquire_and_prepare, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/banking77-seed-42.json"),
    )
    parser.add_argument("--offline", action="store_true", help="Do not make network requests")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = acquire_and_prepare(
        args.config,
        args.raw_dir,
        args.manifest,
        offline=args.offline,
        force_download=args.force_download,
    )
    validate_manifest(manifest)
    summary = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": manifest["source_commit"],
        "split_counts": {
            name: details["count"] for name, details in manifest["splits"].items()
        },
        "quarantined_train_rows": len(manifest["quarantined_train"]),
        "integrity": manifest["integrity"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

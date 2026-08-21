#!/usr/bin/env python3
"""Build immutable BANKING77 train/validation manifests for seeds 17, 42 and 73."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from governed_banking.data import sha256_file
from governed_banking.multiseed import MultiSeedExperimentConfig, build_multiseed_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multiseed_lora.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    return parser.parse_args()


def implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "prepare_multiseed_manifests.py": script_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    config = MultiSeedExperimentConfig.from_yaml(args.config)
    registry = build_multiseed_manifests(
        config,
        dataset_config_path=args.dataset_config,
        raw_directory=args.raw_dir,
        config_sha256=sha256_file(args.config),
        implementation_sha256=implementation_hashes(Path(__file__)),
    )
    print(
        json.dumps(
            {
                "registry": str(config.registry_path),
                "registry_sha256": registry["registry_sha256"],
                "manifests": registry["manifest_entries"],
                "model_access_boundary": registry["model_access_boundary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

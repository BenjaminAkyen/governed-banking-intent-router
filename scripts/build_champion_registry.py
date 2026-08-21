#!/usr/bin/env python3
"""Validate Module 12 evidence and build the fail-closed model registry snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from governed_banking.champion import (  # noqa: E402
    ChampionChallengerConfig,
    build_registry_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/champion_challenger.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/champion/champion-registry.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = ChampionChallengerConfig.from_yaml(args.config)
        report = build_registry_report(
            config,
            report_path=args.report,
            implementation_paths={
                "build_champion_registry.py": PROJECT_ROOT / "scripts/build_champion_registry.py",
                "champion.py": PROJECT_ROOT / "src/governed_banking/champion.py",
            },
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Champion registry build failed: {error}", file=sys.stderr)
        return 2
    print(f"Champion: {report['current_champion_id']}")
    print(f"Decision: {report['current_decision']['action']}")
    print(f"Report: {args.report}")
    print(f"SHA-256: {report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify one Module 11 runtime profile on real local hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from governed_banking.runtime_evidence import (  # noqa: E402
    RuntimeProfile,
    run_runtime_verification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime/auto.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/runtime/auto-runtime.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        profile = RuntimeProfile.from_yaml(args.profile)
        report = run_runtime_verification(
            profile,
            report_path=args.report,
            implementation_paths={
                "accelerator.py": PROJECT_ROOT / "src/governed_banking/accelerator.py",
                "runtime_evidence.py": PROJECT_ROOT / "src/governed_banking/runtime_evidence.py",
                "verify_accelerator.py": PROJECT_ROOT / "scripts/verify_accelerator.py",
            },
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Runtime verification failed: {error}", file=sys.stderr)
        return 2
    runtime = report["runtime"]
    print(
        f"Verified {profile.profile_name}: {runtime['device']} "
        f"({runtime['accelerator_name']})"
    )
    print(f"Report: {args.report}")
    print(f"SHA-256: {report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

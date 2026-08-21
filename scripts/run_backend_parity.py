#!/usr/bin/env python3
"""Run one Module 11 prediction-parity report on real MPS or CUDA hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from governed_banking.parity import (  # noqa: E402
    PredictionParityConfig,
    run_backend_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("mps", "cuda"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/prediction_parity.yaml",
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = PredictionParityConfig.from_yaml(args.config)
        report = run_backend_evidence(
            config,
            backend=args.backend,
            report_path=args.report,
            implementation_paths={
                "accelerator.py": PROJECT_ROOT / "src/governed_banking/accelerator.py",
                "parity.py": PROJECT_ROOT / "src/governed_banking/parity.py",
                "portable_inference.py": (
                    PROJECT_ROOT / "src/governed_banking/portable_inference.py"
                ),
                "run_backend_parity.py": PROJECT_ROOT / "scripts/run_backend_parity.py",
            },
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Backend parity run failed: {error}", file=sys.stderr)
        return 2
    print(f"Verified real {args.backend.upper()} inference on {report['case_count']} cases")
    print(f"Report: {args.report}")
    print(f"SHA-256: {report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

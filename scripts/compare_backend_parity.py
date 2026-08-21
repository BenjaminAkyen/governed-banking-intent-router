#!/usr/bin/env python3
"""Compare independently generated MPS and CUDA Module 11 evidence reports."""

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
    compare_backend_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/prediction_parity.yaml",
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = PredictionParityConfig.from_yaml(args.config)
        report = compare_backend_reports(
            config,
            reference_report_path=args.reference,
            candidate_report_path=args.candidate,
            comparison_path=args.report,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Parity comparison failed: {error}", file=sys.stderr)
        return 2
    results = report["results"]
    print(
        "MPS/CUDA parity gates: "
        f"{'PASS' if results['all_gates_passed'] else 'FAIL'} "
        f"(max probability delta={results['maximum_absolute_probability_delta']:.8g})"
    )
    print(f"Report: {args.report}")
    print(f"SHA-256: {report['report_sha256']}")
    return 0 if results["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

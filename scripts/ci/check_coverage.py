"""Enforce registered per-module coverage floors from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/ci/coverage-gates.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _normalise(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().removeprefix("./")


def _load_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def evaluate_coverage(coverage: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported coverage contract schema")
    registered = contract.get("modules")
    observed = coverage.get("files")
    if not isinstance(registered, dict) or not isinstance(observed, dict):
        raise ValueError("coverage contract and report must contain module mappings")

    by_path = {_normalise(str(path)): value for path, value in observed.items()}
    results: list[dict[str, Any]] = []
    for raw_path, gate in registered.items():
        path = _normalise(str(raw_path))
        if not isinstance(gate, dict):
            raise ValueError(f"invalid coverage gate for {path}")
        minimum = float(gate["minimum_percent"])
        measured_file = by_path.get(path)
        if not isinstance(measured_file, dict):
            results.append(
                {
                    "module": path,
                    "minimum_percent": minimum,
                    "measured_percent": None,
                    "passed": False,
                    "reason": "module_missing_from_coverage_report",
                }
            )
            continue
        summary = measured_file.get("summary")
        if not isinstance(summary, dict) or "percent_covered" not in summary:
            raise ValueError(f"coverage summary missing for {path}")
        measured = float(summary["percent_covered"])
        results.append(
            {
                "module": path,
                "minimum_percent": minimum,
                "measured_percent": measured,
                "passed": measured >= minimum,
                "reason": "at_or_above_floor" if measured >= minimum else "below_floor",
            }
        )

    return {
        "schema_version": 1,
        "contract_version": contract.get("contract_version"),
        "measurement": contract.get("measurement"),
        "all_passed": all(result["passed"] for result in results),
        "modules": results,
    }


def main() -> int:
    args = _arguments()
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    contract = _load_mapping(args.contract)
    result = evaluate_coverage(coverage, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for module in result["modules"]:
        print(
            f"{module['module']}: measured={module['measured_percent']} "
            f"minimum={module['minimum_percent']} passed={module['passed']}"
        )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

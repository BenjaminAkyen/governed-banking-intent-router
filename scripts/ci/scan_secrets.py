"""Run provider-aware secret detection and emit a value-free result summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DISABLED_NOISY_PLUGINS = (
    "Base64HighEntropyString",
    "HexHighEntropyString",
    "IPPublicDetector",
    "KeywordDetector",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("detect-secrets output did not contain a results mapping")
    findings: list[dict[str, str]] = []
    for filename, matches in results.items():
        if not isinstance(matches, list):
            continue
        for match in matches:
            if isinstance(match, dict):
                findings.append(
                    {
                        "file": str(filename),
                        "type": str(match.get("type", "unknown")),
                    }
                )
    return {
        "schema_version": 1,
        "scanner": "detect-secrets",
        "scanner_version": report.get("version"),
        "network_verification_used": False,
        "disabled_high_false_positive_plugins": list(DISABLED_NOISY_PLUGINS),
        "finding_count": len(findings),
        "findings": findings,
        "contains_secret_values": False,
        "passed": not findings,
    }


def main() -> int:
    args = _arguments()
    command = [sys.executable, "-m", "detect_secrets", "scan", "--no-verify"]
    for plugin in DISABLED_NOISY_PLUGINS:
        command.extend(["--disable-plugin", plugin])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    summary = _summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Provider-aware secret findings: {summary['finding_count']}")
    for finding in summary["findings"]:
        print(f"- {finding['file']}: {finding['type']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

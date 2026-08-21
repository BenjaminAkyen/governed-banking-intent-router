"""Generate a metadata-only benchmark for privacy and deterministic routing controls."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from governed_banking.data import sha256_file
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.privacy import PrivacyConfig, redact_pii


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def main() -> int:
    args = _arguments()
    if not 1 <= args.iterations <= 10_000:
        raise ValueError("iterations must be between 1 and 10,000")
    privacy_path = Path("configs/privacy.yaml")
    policy_path = Path("configs/routing_policy.yaml")
    fixture_path = Path("data/fixtures/pii-redaction-cases.jsonl")
    privacy = PrivacyConfig.from_yaml(privacy_path)
    policy = RoutingPolicyConfig.from_yaml(policy_path)
    messages = [
        json.loads(line)["input"]
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    durations_ms: list[float] = []
    actions: dict[str, int] = {}
    for _ in range(args.iterations):
        for message in messages:
            started = time.perf_counter_ns()
            redaction = redact_pii(privacy, str(message))
            decision = route_request(
                policy,
                RoutingInput(
                    predicted_intent="card_arrival",
                    model_seed=42,
                    uncertainty_signal="max_probability",
                    uncertainty_score=0.95,
                    pii_type_counts=redaction.pii_type_counts,
                    redaction_succeeded=True,
                ),
            )
            durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            actions[decision.action] = actions.get(decision.action, 0) + 1
    report = {
        "schema_version": 1,
        "benchmark": "privacy_redaction_and_deterministic_routing",
        "scope": "synthetic_control_path_only_no_model_inference",
        "iterations_per_fixture": args.iterations,
        "fixture_count": len(messages),
        "operation_count": len(durations_ms),
        "latency_ms": {
            "median": statistics.median(durations_ms),
            "p95": _percentile(durations_ms, 0.95),
            "maximum": max(durations_ms),
        },
        "action_counts": actions,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system().lower(),
            "machine": platform.machine().lower(),
        },
        "input_hashes": {
            "privacy_config_sha256": sha256_file(privacy_path),
            "routing_policy_sha256": sha256_file(policy_path),
            "synthetic_fixture_sha256": sha256_file(fixture_path),
        },
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "performance_gate_applied": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded {len(durations_ms)} metadata-only control-path operations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

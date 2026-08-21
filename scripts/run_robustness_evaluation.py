#!/usr/bin/env python3
"""Run the locked Module 13 synthetic pack on the real MPS LoRA research service."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from governed_banking.api import ServiceConfig
from governed_banking.baseline import assert_text_free_artifact, write_json_artifact
from governed_banking.data import sha256_file, stable_json_sha256, validate_manifest
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.portable_inference import PortableLoRAPredictor
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.robustness import RobustnessEvaluationConfig, load_robustness_cases
from governed_banking.runtime_evidence import RuntimeProfile

REPORT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/robustness_evaluation.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RobustnessEvaluationConfig.from_yaml(args.config)
    pack_manifest = _load_self_hashed_manifest(config.manifest_path)
    _validate_pack_binding(config, pack_manifest)

    service_config = ServiceConfig.from_yaml(config.service_config_path)
    privacy_config = PrivacyConfig.from_yaml(service_config.privacy_config_path)
    routing_config = RoutingPolicyConfig.from_yaml(service_config.routing_config_path)
    runtime_profile = RuntimeProfile.from_yaml(config.runtime_profile_path)
    if runtime_profile.device_preference != "mps":
        raise ValueError("the registered Module 13 local assessment must run on explicit MPS")

    taxonomy_manifest = json.loads(
        config.taxonomy_manifest_path.read_text(encoding="utf-8")
    )
    validate_manifest(taxonomy_manifest)
    cases = load_robustness_cases(
        config.pack_path,
        allowed_intents=taxonomy_manifest["label_names"],
        allowed_pii_types=privacy_config.detector_order,
    )
    predictor = PortableLoRAPredictor(service_config.predictor, runtime_profile)
    results: list[dict[str, Any]] = []
    prohibited_value_matches = 0
    try:
        for case in cases:
            redaction = redact_pii(privacy_config, case.text)
            prediction = predictor.predict(redaction.redacted_text)
            decision = route_request(
                routing_config,
                RoutingInput(
                    predicted_intent=prediction.predicted_intent,
                    model_seed=prediction.model_seed,
                    uncertainty_signal=prediction.uncertainty_signal,
                    uncertainty_score=prediction.uncertainty_score,
                    pii_type_counts=redaction.pii_type_counts,
                    redaction_succeeded=True,
                ),
            )
            threshold = routing_config.thresholds[prediction.model_seed]
            result = {
                "case_id": case.case_id,
                "primary_family": case.primary_family,
                "risk_severity": case.risk_severity,
                "out_of_scope": case.out_of_scope,
                "acceptable_intents": list(case.acceptable_intents),
                "predicted_intent": prediction.predicted_intent,
                "intent_acceptable": (
                    None
                    if case.out_of_scope
                    else prediction.predicted_intent in case.acceptable_intents
                ),
                "expected_routing_action": case.expected_routing_action,
                "observed_routing_action": decision.action,
                "routing_action_match": decision.action == case.expected_routing_action,
                "routing_reason_codes": list(decision.reason_codes),
                "expected_pii_types": list(case.expected_pii_types),
                "observed_pii_types": sorted(redaction.pii_type_counts),
                "pii_expectation_match": tuple(sorted(redaction.pii_type_counts))
                == tuple(sorted(case.expected_pii_types)),
                "uncertainty_signal": prediction.uncertainty_signal,
                "uncertainty_score": prediction.uncertainty_score,
                "experimental_threshold": threshold.threshold,
                "at_or_above_experimental_threshold": (
                    prediction.uncertainty_score >= threshold.threshold
                ),
            }
            serialized = json.dumps(result, sort_keys=True)
            if case.text in serialized or redaction.redacted_text in serialized:
                prohibited_value_matches += 1
            results.append(result)
    finally:
        predictor.release_accelerator_cache()

    report = _build_report(
        config=config,
        pack_manifest=pack_manifest,
        service_config=service_config,
        privacy_config=privacy_config,
        routing_config=routing_config,
        runtime_profile=runtime_profile,
        runtime=predictor.runtime.to_dict(),
        results=results,
        prohibited_value_matches=prohibited_value_matches,
    )
    assert_text_free_artifact(report)
    report["report_sha256"] = stable_json_sha256(report)
    write_json_artifact(report, config.report_path)
    print(
        json.dumps(
            {
                "report": str(config.report_path),
                "report_sha256": report["report_sha256"],
                "case_count": len(results),
                "in_scope_acceptable_intent_rate": report["metrics"][
                    "in_scope_acceptable_intent_rate"
                ],
                "expected_security_routing_recall": report["metrics"][
                    "expected_security_routing_recall"
                ],
                "all_assessment_gates_passed": report["assessment_gate"]["all_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_self_hashed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("robustness manifest must be a JSON object")
    body = dict(value)
    expected = body.pop("manifest_sha256", None)
    if expected != stable_json_sha256(body):
        raise ValueError("robustness manifest content hash check failed")
    return value


def _validate_pack_binding(
    config: RobustnessEvaluationConfig, manifest: dict[str, Any]
) -> None:
    if manifest.get("pack_version") != config.pack_version:
        raise ValueError("robustness manifest version differs from the evaluation registration")
    if manifest.get("pack_sha256") != config.expected_pack_sha256:
        raise ValueError("robustness manifest binds a different pack")
    if manifest.get("config_sha256") != sha256_file(config.config_path):
        raise ValueError("robustness manifest binds a different evaluation configuration")
    if manifest.get("acceptance_gate", {}).get("all_passed") is not True:
        raise ValueError("robustness pack did not pass its construction gates")


def _build_report(
    *,
    config: RobustnessEvaluationConfig,
    pack_manifest: dict[str, Any],
    service_config: ServiceConfig,
    privacy_config: PrivacyConfig,
    routing_config: RoutingPolicyConfig,
    runtime_profile: RuntimeProfile,
    runtime: dict[str, Any],
    results: list[dict[str, Any]],
    prohibited_value_matches: int,
) -> dict[str, Any]:
    in_scope = [result for result in results if not result["out_of_scope"]]
    expected_security = [
        result for result in results if result["expected_routing_action"] == "security_queue"
    ]
    intent_passes = sum(result["intent_acceptable"] is True for result in in_scope)
    routing_passes = sum(result["routing_action_match"] is True for result in results)
    security_passes = sum(
        result["observed_routing_action"] == "security_queue" for result in expected_security
    )
    pii_passes = sum(result["pii_expectation_match"] is True for result in results)
    suggestion_count = sum(
        result["observed_routing_action"] == "suggest_queue" for result in results
    )
    experimental_accepts = sum(
        result["at_or_above_experimental_threshold"] is True for result in results
    )
    metrics = {
        "case_count": len(results),
        "in_scope_case_count": len(in_scope),
        "out_of_scope_case_count": len(results) - len(in_scope),
        "in_scope_acceptable_intent_count": intent_passes,
        "in_scope_acceptable_intent_rate": _fraction(intent_passes, len(in_scope)),
        "routing_action_match_count": routing_passes,
        "routing_action_match_rate": _fraction(routing_passes, len(results)),
        "expected_security_case_count": len(expected_security),
        "expected_security_routed_count": security_passes,
        "expected_security_routing_recall": _fraction(
            security_passes, len(expected_security)
        ),
        "pii_expectation_match_count": pii_passes,
        "pii_expectation_match_rate": _fraction(pii_passes, len(results)),
        "suggest_queue_count": suggestion_count,
        "at_or_above_experimental_threshold_count": experimental_accepts,
        "at_or_above_experimental_threshold_rate": _fraction(
            experimental_accepts, len(results)
        ),
    }
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["primary_family"]].append(result)
    for family, family_results in sorted(grouped.items()):
        eligible = [result for result in family_results if not result["out_of_scope"]]
        by_family[family] = {
            "case_count": len(family_results),
            "in_scope_case_count": len(eligible),
            "acceptable_intent_count": sum(
                result["intent_acceptable"] is True for result in eligible
            ),
            "acceptable_intent_rate": (
                _fraction(
                    sum(result["intent_acceptable"] is True for result in eligible),
                    len(eligible),
                )
                if eligible
                else None
            ),
            "routing_action_match_count": sum(
                result["routing_action_match"] is True for result in family_results
            ),
            "routing_action_match_rate": _fraction(
                sum(result["routing_action_match"] is True for result in family_results),
                len(family_results),
            ),
        }

    gate = {
        "in_scope_acceptable_intent_rate": (
            metrics["in_scope_acceptable_intent_rate"]
            >= config.minimum_in_scope_acceptable_intent_rate
        ),
        "expected_security_routing_recall": (
            metrics["expected_security_routing_recall"]
            >= config.minimum_expected_security_routing_recall
        ),
        "overall_routing_action_match_rate": (
            metrics["routing_action_match_rate"]
            >= config.minimum_overall_routing_action_match_rate
        ),
        "all_pii_expectations_matched": pii_passes == len(results),
        "zero_suggestion_actions": suggestion_count == 0,
        "text_free_report": prohibited_value_matches == 0,
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "module13_lora_mps_synthetic_robustness_assessment",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": "synthetic_stress_test_not_production_validation",
        "model_role": config.model_role,
        "champion_model_evaluated": False,
        "contains_input_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "source_evidence": {
            "config_sha256": sha256_file(config.config_path),
            "pack_sha256": config.expected_pack_sha256,
            "pack_manifest_sha256": pack_manifest["manifest_sha256"],
            "service_config_sha256": service_config.config_sha256,
            "privacy_config_sha256": privacy_config.config_sha256,
            "routing_config_sha256": routing_config.config_sha256,
            "runtime_profile_sha256": sha256_file(config.runtime_profile_path),
            "checkpoint_files_sha256": dict(
                sorted(service_config.predictor.expected_checkpoint_files_sha256.items())
            ),
        },
        "runtime_profile": runtime_profile.to_dict(),
        "runtime": runtime,
        "implementation_sha256": {
            "policy.py": sha256_file(Path("src/governed_banking/policy.py")),
            "portable_inference.py": sha256_file(
                Path("src/governed_banking/portable_inference.py")
            ),
            "privacy.py": sha256_file(Path("src/governed_banking/privacy.py")),
            "robustness.py": sha256_file(Path("src/governed_banking/robustness.py")),
            "run_robustness_evaluation.py": sha256_file(Path(__file__)),
        },
        "metrics": metrics,
        "by_primary_family": by_family,
        "observed_prediction_counts": dict(
            sorted(Counter(result["predicted_intent"] for result in results).items())
        ),
        "observed_routing_action_counts": dict(
            sorted(Counter(result["observed_routing_action"] for result in results).items())
        ),
        "cases": results,
        "assessment_gate": gate,
        "data_boundary": {
            "fixture_is_synthetic": True,
            "customer_data_access": False,
            "production_data_access": False,
            "official_test_access": False,
            "banking77_access_during_model_assessment": False,
            "production_validation": False,
            "uncertainty_status": "experimental_review_only",
        },
        "limitations": [
            "This evaluates the Module 10 LoRA research service, not the retained TF-IDF "
            "champion.",
            "The authored synthetic pack is a stress test and cannot establish production "
            "performance.",
            "Out-of-scope cases have no acceptable BANKING77 intent; their closed-set "
            "predictions are reported but not scored as classification errors.",
            "The uncertainty threshold remains experimental and cannot authorize automated "
            "suggestions.",
            "Representative claims require appropriately governed real-world banking messages "
            "and independent evaluation.",
        ],
    }
    report["assessment_gate"]["all_passed"] = all(gate.values())
    return report


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("metric denominator must be positive")
    return round(numerator / denominator, 10)


if __name__ == "__main__":
    main()

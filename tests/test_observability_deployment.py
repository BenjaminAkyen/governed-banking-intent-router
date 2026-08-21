from pathlib import Path

import yaml

from governed_banking.observability_config import ALLOWED_ATTRIBUTE_KEYS


def test_collectors_are_local_fail_closed_and_have_no_log_pipeline() -> None:
    paths = (
        Path("deploy/observability/collector.yaml"),
        Path("deploy/observability/collector-local.yaml"),
    )
    for path in paths:
        _assert_collector_privacy_contract(yaml.safe_load(path.read_text(encoding="utf-8")))


def _assert_collector_privacy_contract(config: dict) -> None:
    receiver = config["receivers"]["otlp"]["protocols"]["grpc"]
    redaction = config["processors"]["redaction"]
    pipelines = config["service"]["pipelines"]

    assert receiver["endpoint"] == "127.0.0.1:4317"
    assert redaction["allow_all_keys"] is False
    assert redaction["summary"] == "silent"
    assert tuple(redaction["allowed_keys"]) == ALLOWED_ATTRIBUTE_KEYS
    assert "logs" not in pipelines
    assert pipelines["metrics"]["processors"] == ["memory_limiter", "redaction", "batch"]
    assert pipelines["traces"]["processors"] == ["memory_limiter", "redaction", "batch"]


def test_container_entrypoints_require_observed_service() -> None:
    expected = (
        "governed_banking.observed_deployment_service:"
        "create_observed_app_from_environment"
    )
    for name in ("cpu.Dockerfile", "cuda.Dockerfile"):
        content = Path("deploy/docker", name).read_text(encoding="utf-8")
        assert expected in content
        assert "--no-access-log" in content


def test_kubernetes_observability_template_is_private_and_secret_backed() -> None:
    documents = list(
        yaml.safe_load_all(
            Path("deploy/kubernetes/observability-patch.yaml.template").read_text(
                encoding="utf-8"
            )
        )
    )
    deployment, config_map, service, policy = documents
    containers = deployment["spec"]["template"]["spec"]["containers"]
    router = next(item for item in containers if item["name"] == "router")
    collector = next(item for item in containers if item["name"] == "otel-collector")

    assert router["env"] == [
        {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": "http://127.0.0.1:4317"}
    ]
    authorization = next(
        item for item in collector["env"] if item["name"] == "TRACE_BACKEND_AUTHORIZATION"
    )
    assert "valueFrom" in authorization and "value" not in authorization
    assert config_map["kind"] == "ConfigMap"
    assert service["spec"]["type"] == "ClusterIP"
    assert policy["spec"]["ingress"][0]["ports"][0]["port"] == 9464
    namespace = policy["spec"]["ingress"][0]["from"][0]["namespaceSelector"]
    assert namespace["matchLabels"] == {"governed-banking-monitoring": "true"}

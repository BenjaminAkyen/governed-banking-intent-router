from pathlib import Path

import yaml


def test_container_profiles_never_register_mps_or_cpu_fallback() -> None:
    for name in ("cpu.Dockerfile", "cuda.Dockerfile"):
        content = Path("deploy/docker", name).read_text(encoding="utf-8")
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in content
        assert "USER 10001:10001" in content
        assert "/health/live" in content
        assert "--workers\", \"1" in content


def test_cuda_image_requires_an_explicit_cuda_pytorch_base() -> None:
    content = Path("deploy/docker/cuda.Dockerfile").read_text(encoding="utf-8")
    assert "ARG PYTORCH_CUDA_IMAGE\nFROM ${PYTORCH_CUDA_IMAGE}" in content
    assert "linux-cuda.yaml" in content


def test_cpu_image_requires_an_explicit_cpu_base() -> None:
    content = Path("deploy/docker/cpu.Dockerfile").read_text(encoding="utf-8")
    assert "ARG PYTHON_CPU_IMAGE\nFROM ${PYTHON_CPU_IMAGE}" in content
    assert "linux-cpu.yaml" in content


def test_gateway_contract_strips_identity_headers_before_injection() -> None:
    contract = yaml.safe_load(
        Path("deploy/gateway/identity-gateway-contract.yaml").read_text(encoding="utf-8")
    )
    origin = contract["origin_boundary"]
    injected = set(origin["inject_after_successful_identity_validation"])
    stripped = set(origin["remove_client_supplied_headers"])
    assert injected <= stripped
    assert origin["public_network_access"] is False
    assert contract["claims"]["production_approved"] is False


def test_kubernetes_template_has_health_rollback_and_origin_isolation() -> None:
    content = Path("deploy/kubernetes/router.yaml.template").read_text(encoding="utf-8")
    documents = list(yaml.safe_load_all(content))
    deployment, service, policy = documents
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item for item in container["env"]}
    readiness_command = " ".join(container["readinessProbe"]["exec"]["command"])
    liveness_command = " ".join(container["livenessProbe"]["exec"]["command"])
    assert "/health/ready" in readiness_command
    assert "/health/live" in liveness_command
    assert "GOVERNED_BANKING_ROLLBACK_REFERENCE" in environment
    assert "GOVERNED_BANKING_GATEWAY_ASSERTION" in environment
    assert service["spec"]["type"] == "ClusterIP"
    assert policy["spec"]["policyTypes"] == ["Ingress"]


def test_cuda_patch_requests_a_real_nvidia_gpu() -> None:
    patch = yaml.safe_load(
        Path("deploy/kubernetes/cuda-patch.yaml.template").read_text(encoding="utf-8")
    )
    container = patch["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["env"][0]["value"] == "configs/deployment/linux-cuda.yaml"

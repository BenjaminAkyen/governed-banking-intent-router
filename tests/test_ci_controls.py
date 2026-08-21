from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_changed_python import _valid_ref
from scripts.ci.check_coverage import evaluate_coverage
from scripts.ci.scan_secrets import _summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_PIN = re.compile(r"^\s*uses:\s*[^@\s]+@[0-9a-f]{40}(?:\s+#.*)?$")


def _workflow_text(name: str) -> str:
    return (PROJECT_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def test_every_external_action_is_commit_pinned() -> None:
    workflow_paths = sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml"))
    uses_lines = [
        line
        for path in workflow_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if "uses:" in line
    ]

    assert uses_lines
    assert all(ACTION_PIN.fullmatch(line) for line in uses_lines), uses_lines


def test_workflows_use_safe_events_and_do_not_reference_secrets() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml"))
    )

    assert "pull_request_target" not in combined
    assert "${{ secrets." not in combined
    assert "persist-credentials: false" in combined


def test_cpu_matrix_covers_supported_platforms_and_python_versions() -> None:
    workflow = _load_yaml(PROJECT_ROOT / ".github/workflows/quality.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    cpu = jobs["cpu-tests"]
    assert isinstance(cpu, dict)
    strategy = cpu["strategy"]
    assert isinstance(strategy, dict)
    matrix = strategy["matrix"]
    assert isinstance(matrix, dict)

    assert matrix["os"] == ["ubuntu-24.04", "macos-15", "windows-2025"]
    assert matrix["python"] == ["3.12", "3.13"]
    assert workflow["permissions"] == {"contents": "read"}


def test_codeql_has_only_required_repository_permissions() -> None:
    workflow = _load_yaml(PROJECT_ROOT / ".github/workflows/codeql.yml")

    assert workflow["permissions"] == {"contents": "read", "security-events": "write"}


def test_dependabot_covers_python_actions_and_containers() -> None:
    config = _load_yaml(PROJECT_ROOT / ".github/dependabot.yml")
    updates = config["updates"]
    assert isinstance(updates, list)

    assert {entry["package-ecosystem"] for entry in updates} == {
        "pip",
        "github-actions",
        "docker",
    }


def test_coverage_gate_handles_windows_paths_without_rounding() -> None:
    contract = {
        "schema_version": 1,
        "contract_version": "test",
        "measurement": "line_percent_covered",
        "modules": {"src/governed_banking/privacy.py": {"minimum_percent": 90}},
    }
    passing = {
        "files": {"src\\governed_banking\\privacy.py": {"summary": {"percent_covered": 90.0}}}
    }
    failing = {
        "files": {"src/governed_banking/privacy.py": {"summary": {"percent_covered": 89.999}}}
    }

    assert evaluate_coverage(passing, contract)["all_passed"] is True
    assert evaluate_coverage(failing, contract)["all_passed"] is False


def test_secret_summary_drops_secret_values_and_line_hashes() -> None:
    report = {
        "version": "test",
        "results": {
            "example.py": [
                {
                    "type": "GitHub Token",
                    "hashed_secret": "must-not-survive",
                    "line_number": 7,
                }
            ]
        },
    }

    summary = _summary(report)

    assert summary["finding_count"] == 1
    assert summary["findings"] == [{"file": "example.py", "type": "GitHub Token"}]
    assert "must-not-survive" not in json.dumps(summary)
    assert summary["contains_secret_values"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("HEAD", True),
        ("a" * 40, True),
        ("main", False),
        ("a" * 39, False),
        ("$(touch unsafe)", False),
    ],
)
def test_changed_file_checker_rejects_untrusted_git_refs(value: str, expected: bool) -> None:
    assert _valid_ref(value) is expected


def test_control_benchmark_is_metadata_only(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/ci/run_control_benchmark.py",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    fixture_messages = [
        json.loads(line)["input"]
        for line in (PROJECT_ROOT / "data/fixtures/pii-redaction-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    serialized = json.dumps(report)

    assert report["contains_message_text"] is False
    assert report["contains_redacted_text"] is False
    assert report["contains_message_hash"] is False
    assert report["performance_gate_applied"] is False
    assert all(message not in serialized for message in fixture_messages)


def test_ci_container_has_immutable_base_contract_and_non_root_runtime() -> None:
    dockerfile = (PROJECT_ROOT / "deploy/docker/ci.Dockerfile").read_text(encoding="utf-8")

    assert "ARG PYTHON_CI_IMAGE" in dockerfile
    assert "FROM ${PYTHON_CI_IMAGE}" in dockerfile
    assert "--no-deps ." in dockerfile
    assert "USER 65532:65532" in dockerfile

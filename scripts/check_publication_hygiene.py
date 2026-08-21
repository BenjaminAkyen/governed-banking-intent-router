#!/usr/bin/env python3
"""Fail when public repository material contains machine-local release residue."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_IGNORE_RULES = {
    ".env",
    ".ipynb_checkpoints/",
    ".venv/",
    "*.bin",
    "*.ckpt",
    "*.log",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "artifacts/",
    "checkpoints/",
    "data/raw/",
    "mlruns/",
    "wandb/",
}

FORBIDDEN_TRACKED_PREFIXES = (
    ".venv/",
    "artifacts/",
    "checkpoints/",
    "data/raw/",
    "lightning_logs/",
    "mlruns/",
    "runs/",
    "wandb/",
)

FORBIDDEN_TRACKED_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".joblib",
    ".log",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)

MACHINE_PATH_PATTERNS = (
    ("macOS user path", re.compile("/" + "Users" + r"/[^/\s\"']+/")),
    ("Linux user path", re.compile("/" + "home" + r"/[^/\s\"']+/")),
    ("Windows user path", re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+\\")),
    ("VS Code local resource URL", re.compile("file+" + r"\.vscode-resource")),
)

CREDENTIAL_PATTERNS = (
    ("OpenAI-style secret", re.compile("s" + r"k-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile("g" + r"h[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access-key identifier", re.compile("A" + r"KIA[0-9A-Z]{16}")),
    ("private-key block", re.compile("BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY")),
)

SCAN_EXCLUSIONS = {
    Path("scripts/check_publication_hygiene.py"),
}


def repository_root() -> Path:
    """Return the checkout containing this script."""

    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".gitignore").is_file():
            return candidate
    raise FileNotFoundError("could not locate the repository root")


def public_files(root: Path) -> tuple[Path, ...]:
    """Return tracked and unignored files so the check also covers pre-commit additions."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if relative not in SCAN_EXCLUSIONS and (root / relative).is_file():
            paths.append(relative)
    return tuple(sorted(paths, key=str))


def read_public_text(path: Path) -> str | None:
    """Decode a text file without treating model or image bytes as source material."""

    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_public_material(root: Path, paths: tuple[Path, ...]) -> list[str]:
    """Return machine-path, credential and tracked-artifact findings."""

    findings: list[str] = []
    for relative in paths:
        normalized = relative.as_posix()
        if normalized.startswith(FORBIDDEN_TRACKED_PREFIXES):
            findings.append(f"tracked runtime artifact: {normalized}")
        if normalized.lower().endswith(FORBIDDEN_TRACKED_SUFFIXES):
            findings.append(f"tracked model or runtime file: {normalized}")

        text = read_public_text(root / relative)
        if text is None:
            continue
        for label, pattern in (*MACHINE_PATH_PATTERNS, *CREDENTIAL_PATTERNS):
            if pattern.search(text):
                findings.append(f"{label}: {normalized}")
    return findings


def scan_credential_history(root: Path) -> list[str]:
    """Check Git patch history for high-confidence credential signatures without printing values."""

    result = subprocess.run(
        ["git", "log", "-p", "--all", "--format="],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    findings = []
    for label, pattern in CREDENTIAL_PATTERNS:
        if pattern.search(result.stdout):
            findings.append(f"{label} appears in Git patch history")
    return findings


def validate_ignore_rules(root: Path) -> list[str]:
    """Return required publication-boundary rules missing from .gitignore."""

    rules = {
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return [f"missing .gitignore rule: {rule}" for rule in sorted(REQUIRED_IGNORE_RULES - rules)]


def validate_notebooks(root: Path) -> list[str]:
    """Parse notebooks and reject structurally invalid documents."""

    findings: list[str] = []
    for path in sorted((root / "output/jupyter-notebook").glob("*.ipynb")):
        try:
            notebook: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            findings.append(f"invalid notebook JSON: {path.relative_to(root)}: {error}")
            continue
        if not isinstance(notebook.get("cells"), list):
            findings.append(f"notebook cells are invalid: {path.relative_to(root)}")
        if not isinstance(notebook.get("nbformat"), int):
            findings.append(f"notebook format is invalid: {path.relative_to(root)}")
    return findings


def execute_notebook_setup_cells(root: Path) -> list[str]:
    """Execute only root-discovery cells from the nested notebook directory."""

    findings: list[str] = []
    notebook_directory = root / "output/jupyter-notebook"
    previous_directory = Path.cwd()
    previous_matplotlib_config = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="publication-matplotlib-") as cache_directory:
            os.environ["MPLCONFIGDIR"] = cache_directory
            os.chdir(notebook_directory)
            for path in sorted(notebook_directory.glob("*.ipynb")):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                setup_source = _setup_source(notebook)
                if setup_source is None:
                    findings.append(f"notebook has no portable setup cell: {path.name}")
                    continue
                namespace: dict[str, Any] = {"__name__": "__notebook_setup_check__"}
                try:
                    exec(compile(setup_source, str(path), "exec"), namespace)
                except Exception as error:  # noqa: BLE001 - report every setup portability failure
                    findings.append(
                        f"notebook setup failed: {path.name}: {type(error).__name__}: {error}"
                    )
                    continue
                discovered = namespace.get("PROJECT_ROOT", namespace.get("ROOT"))
                if not isinstance(discovered, Path) or discovered.resolve() != root.resolve():
                    findings.append(f"notebook discovered the wrong repository root: {path.name}")
    finally:
        os.chdir(previous_directory)
        if previous_matplotlib_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_matplotlib_config
    return findings


def _setup_source(notebook: dict[str, Any]) -> str | None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        defines_project_root = "PROJECT_ROOT =" in source
        defines_root = "ROOT = find_" in source
        if defines_project_root or defines_root:
            return source
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-notebook-setup",
        action="store_true",
        help="execute only each notebook's root-discovery setup cell",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repository_root()
    paths = public_files(root)
    findings = [
        *validate_ignore_rules(root),
        *validate_notebooks(root),
        *scan_public_material(root, paths),
        *scan_credential_history(root),
    ]
    if args.execute_notebook_setup:
        findings.extend(execute_notebook_setup_cells(root))

    if findings:
        print("Publication hygiene failed:")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1

    setup_status = " and portable setup cells" if args.execute_notebook_setup else ""
    print(f"Publication hygiene passed for {len(paths)} public files{setup_status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

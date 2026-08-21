from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_publication_hygiene_and_nested_notebook_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/check_publication_hygiene.py"),
            "--execute-notebook-setup",
        ],
        cwd=PROJECT_ROOT / "output/jupyter-notebook",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Publication hygiene passed" in result.stdout

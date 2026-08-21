"""Apply Ruff formatting and lint gates only to changed tracked Python files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    return parser.parse_args()


def _valid_ref(value: str) -> bool:
    if value == "HEAD":
        return True
    return bool(SHA_PATTERN.fullmatch(value))


def _fallback_base() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def changed_python_files(base: str | None, head: str) -> list[str]:
    if not _valid_ref(head):
        raise ValueError("head must be HEAD or a full Git commit SHA")
    if not base or set(base) == {"0"}:
        base = _fallback_base()
    if base is None:
        return sorted(str(path) for path in Path(".").rglob("*.py") if ".venv" not in path.parts)
    if not _valid_ref(base):
        raise ValueError("base must be a full Git commit SHA")
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            base,
            head,
            "--",
            "*.py",
        ],
        check=True,
        capture_output=True,
    )
    return sorted(
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path and Path(path.decode("utf-8")).is_file()
    )


def main() -> int:
    args = _arguments()
    paths = changed_python_files(args.base, args.head)
    if not paths:
        print("No changed Python files require prospective checks.")
        return 0
    print("Prospective Python checks:")
    for path in paths:
        print(f"- {path}")
    subprocess.run([sys.executable, "-m", "ruff", "format", "--check", "--", *paths], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--", *paths], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

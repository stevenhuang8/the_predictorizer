"""The economy regression safety net (AI task 1).

The AI domain is meant to be purely additive, so the economy tests must keep
running unchanged. Passing or failing is the normal run's job; this file checks
that none of them quietly disappeared. tests/economy_baseline.json lists every
economy test, and conftest.py fails the run if one of them skips when it
shouldn't.

After deliberately adding or removing economy tests, rewrite the baseline:

    UPDATE_ECONOMY_BASELINE=1 uv run pytest tests/test_regression_safety.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import ECONOMY_BASELINE, ECONOMY_TEST_FILES

ROOT = Path(__file__).parent.parent


def collect_economy_tests() -> list[str]:
    """Node ids `pytest -m economy` would run, by collecting only."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "economy",
            "-p",
            "no:cacheprovider",
            "tests",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"collection failed:\n{result.stdout}\n{result.stderr}"
    )
    return sorted(line for line in result.stdout.splitlines() if "::" in line)


def load_baseline() -> dict[str, list[str]]:
    baseline: dict[str, list[str]] = json.loads(ECONOMY_BASELINE.read_text())
    return baseline


def test_every_economy_test_file_exists() -> None:
    # A renamed file would silently drop out of `-m economy`.
    present = {path.name for path in (ROOT / "tests").glob("test_*.py")}
    assert sorted(ECONOMY_TEST_FILES - present) == []


def test_every_baseline_economy_test_is_still_collected() -> None:
    collected = collect_economy_tests()
    if os.environ.get("UPDATE_ECONOMY_BASELINE"):
        allowed = load_baseline()["allowed_skips"] if ECONOMY_BASELINE.exists() else []
        ECONOMY_BASELINE.write_text(
            json.dumps({"tests": collected, "allowed_skips": allowed}, indent=2) + "\n"
        )
        return
    missing = sorted(set(load_baseline()["tests"]) - set(collected))
    assert missing == [], (
        f"{len(missing)} economy test(s) in the baseline are gone. If that was "
        "deliberate, rewrite the baseline (see this file's docstring)."
    )


def test_allowed_skips_are_baseline_tests() -> None:
    baseline = load_baseline()
    assert sorted(set(baseline["allowed_skips"]) - set(baseline["tests"])) == []

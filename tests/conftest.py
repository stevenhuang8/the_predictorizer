"""Shared fixtures: a throwaway database per test on the docker-compose Postgres.

The real eco_forecast database is never touched. Tests needing a database are
skipped if Postgres isn't reachable.

Also tags the economy tests (`pytest -m economy`) and, when every economy test
in the baseline ran, fails the session if any of them skipped that isn't
allowed to (see tests/economy_baseline.json and test_regression_safety.py).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, make_dsn
from psycopg2.extensions import connection as Connection

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "")

ECONOMY_BASELINE = Path(__file__).parent / "economy_baseline.json"

# Every test in these files is an economy test. New economy test files must be
# added here; AI-domain tests stay out so `-m economy` keeps meaning "the
# pipeline that was live before the AI domain".
ECONOMY_TEST_FILES = frozenset(
    {
        "test_backfill.py",
        "test_backtest_harness.py",
        "test_baselines.py",
        "test_benchmarks.py",
        "test_calibration.py",
        "test_cli.py",
        "test_explainability.py",
        "test_features.py",
        "test_fomc.py",
        "test_forecast_job.py",
        "test_forecast_schema.py",
        "test_fred_client.py",
        "test_ingest.py",
        "test_lightgbm_model.py",
        "test_metrics.py",
        "test_migrate.py",
        "test_model_comparison.py",
        "test_model_tracking.py",
        "test_postmortem.py",
        "test_queries.py",
        "test_regression_safety.py",
        "test_resolution_job.py",
        "test_statistical.py",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "economy: the economy pipeline's tests (the regression safety net)"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name in ECONOMY_TEST_FILES:
            item.add_marker(pytest.mark.economy)


# Node ids of the tests that skipped in this session.
_skipped: set[str] = set()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and not hasattr(report, "wasxfail"):
        _skipped.add(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run if baseline economy tests skipped that aren't allowed to.

    Only checked when every baseline test was collected, so running a single
    file or test isn't affected. A skip isn't a failure, so without this a run
    with Postgres down would look green while most of the net was off.
    """
    if not ECONOMY_BASELINE.exists() or session.config.option.collectonly:
        return
    baseline = json.loads(ECONOMY_BASELINE.read_text())
    expected = set(baseline["tests"])
    if not expected <= {item.nodeid for item in session.items}:
        return
    unexpected = sorted((_skipped & expected) - set(baseline["allowed_skips"]))
    if unexpected:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "economy regression safety net", red=True)
            reporter.write_line(
                f"{len(unexpected)} economy test(s) skipped that the baseline "
                "expects to run (is Postgres up? is FRED_API_KEY set?):"
            )
            for nodeid in unexpected:
                reporter.write_line(f"  {nodeid}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def test_dsn() -> Iterator[str]:
    """DSN of a freshly created, empty database; dropped after the test."""
    try:
        admin = psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    name = f"test_{uuid.uuid4().hex[:12]}"
    with admin.cursor() as cur:
        cur.execute(f"CREATE DATABASE {name}")
    params = admin.get_dsn_parameters()
    dsn = make_dsn(
        dbname=name,
        user=params["user"],
        host=params["host"],
        port=params["port"],
        password=admin.info.password,
    )
    try:
        yield dsn
    finally:
        with admin.cursor() as cur:
            # WITH (FORCE) drops it even if a pooled connection leaked.
            cur.execute(f"DROP DATABASE {name} WITH (FORCE)")
        admin.close()


@pytest.fixture
def conn(test_dsn: str) -> Iterator[Connection]:
    test_conn = psycopg2.connect(test_dsn)
    try:
        yield test_conn
    finally:
        test_conn.close()

"""Tests for the resumable backfill script (database fixtures in conftest.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eco_prediction.data.fred_client import (
    FREDAPIError,
    FREDClient,
    Observation,
    SeriesInfo,
)
from eco_prediction.db import connection as db
from eco_prediction.db.migrate import migrate
from eco_prediction.scripts.backfill_data import (
    Checkpoint,
    backfill,
    coverage,
    format_report,
)

START = date(1990, 1, 1)
P1, P2 = date(1990, 1, 1), date(1990, 2, 1)
V1, V2 = date(1990, 2, 15), date(1990, 3, 15)


class FakeFRED:
    """Serves a fixed two-vintage history for each known series."""

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.fetched: list[str] = []

    def get_series_info(self, series_id: str) -> SeriesInfo:
        if series_id not in self.known:
            raise FREDAPIError("Bad Request. The series does not exist.", 400)
        return SeriesInfo(
            series_id, series_id, "Index", "Monthly", "SA", P1, P2, "", None
        )

    def get_series_observations(self, series_id: str, **_: Any) -> list[Observation]:
        self.fetched.append(series_id)
        return [
            Observation(P1, V1, 100.0),
            Observation(P1, V2, 101.0),  # revised
            Observation(P2, V2, 102.0),
        ]


def client(fake: FakeFRED) -> FREDClient:
    return fake  # type: ignore[return-value]


@pytest.fixture
def pool(test_dsn: str) -> Iterator[None]:
    db.init_pool(test_dsn, maxconn=2)
    with db.connection() as conn:
        migrate(conn)
    try:
        yield
    finally:
        db.close_pool()


def test_backfill_checkpoints_each_series_and_resumes(
    pool: None, tmp_path: Path
) -> None:
    path = tmp_path / "ckpt.json"
    fake = FakeFRED({"A", "B"})
    failures = backfill(
        client(fake), ["A", "B"], Checkpoint(path, START, None), start=START
    )

    assert failures == {}
    saved = json.loads(path.read_text())
    assert saved["range"] == {"start": "1990-01-01", "end": None}
    assert saved["completed"]["A"]["inserted"] == 3

    # A resumed run with the same range fetches nothing.
    fake.fetched.clear()
    backfill(client(fake), ["A", "B"], Checkpoint(path, START, None), start=START)
    assert fake.fetched == []


def test_failed_series_is_retried_on_next_run(pool: None, tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    fake = FakeFRED({"A"})
    failures = backfill(
        client(fake), ["A", "B"], Checkpoint(path, START, None), start=START
    )
    assert set(failures) == {"B"}

    fake.known.add("B")
    fake.fetched.clear()
    failures = backfill(
        client(fake), ["A", "B"], Checkpoint(path, START, None), start=START
    )
    assert failures == {}
    assert fake.fetched == ["B"]


def test_checkpoint_for_another_range_or_restart_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    Checkpoint(path, START, None).mark_done("A", inserted=1)

    assert Checkpoint(path, START, None).is_done("A")
    assert not Checkpoint(path, date(1980, 1, 1), None).is_done("A")
    assert not Checkpoint(path, START, date(2000, 1, 1)).is_done("A")
    assert not Checkpoint(path, START, None, restart=True).is_done("A")


def test_unreadable_checkpoint_starts_over(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    path.write_text("{not json")
    assert Checkpoint(path, START, None).completed == {}


def test_coverage_report(pool: None, tmp_path: Path) -> None:
    backfill(
        client(FakeFRED({"A"})),
        ["A"],
        Checkpoint(tmp_path / "ckpt.json", START, None),
        start=START,
    )
    with db.connection() as conn:
        rows = coverage(conn, ["A", "MISSING"])

    a, missing = rows
    assert (a.rows, a.vintages, a.first_period, a.first_vintage) == (3, 2, P1, V1)
    assert missing.rows == 0
    report = format_report(rows, START)
    assert "MISSING" in report and "no data" in report
    assert "starts after" in format_report(rows, date(1985, 1, 1))
    assert "total rows: 3" in report

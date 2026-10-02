"""Tests for the FRED ingestion pipeline (database fixtures in conftest.py).

FakeFRED serves a fixed vintage history and mimics FRED's real-time
clipping, so incremental runs are exercised against the behavior that
matters. The live test also needs FRED_API_KEY.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from dotenv import load_dotenv
from psycopg2.extensions import connection as Connection

from eco_prediction.data.fred_client import (
    FREDAPIError,
    FREDClient,
    Observation,
    SeriesInfo,
)
from eco_prediction.data.ingest import ingest_many, ingest_series
from eco_prediction.db import connection as db
from eco_prediction.db.migrate import migrate
from eco_prediction.db.queries import get_series_value_as_of, get_vintage_history

JAN, FEB, MAR = date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)
V1, V2, V3 = date(2024, 2, 2), date(2024, 3, 8), date(2024, 4, 5)

INFO = SeriesInfo(
    series_id="ICSA",
    name="Initial Claims",
    units="Number",
    frequency="Weekly, Ending Saturday",  # longer than the original VARCHAR(20)
    seasonal_adjustment="Seasonally Adjusted",
    observation_start=JAN,
    observation_end=MAR,
    last_updated="2024-04-05",
    notes=None,
)


class FakeFRED:
    """In-memory FRED: `history` holds each period's true vintages."""

    def __init__(self, history: list[Observation], info: SeriesInfo = INFO) -> None:
        self.history = history
        self.info = info
        self.observation_calls: list[dict[str, Any]] = []

    def get_series_info(self, series_id: str) -> SeriesInfo:
        if series_id != self.info.series_id:
            raise FREDAPIError("Bad Request. The series does not exist.", 400)
        return self.info

    def get_vintage_dates(
        self, series_id: str, realtime_start: date | None = None, **_: Any
    ) -> list[date]:
        start = realtime_start or date.min
        return sorted({o.as_of for o in self.history if o.as_of >= start})

    def get_series_observations(
        self,
        series_id: str,
        realtime_start: date | None = None,
        *,
        observation_start: date | None = None,
        observation_end: date | None = None,
        **_: Any,
    ) -> list[Observation]:
        self.observation_calls.append(
            {"realtime_start": realtime_start, "observation_start": observation_start}
        )
        start = realtime_start or date.min
        rows = []
        for period in sorted({o.observed_at for o in self.history}):
            if observation_start and period < observation_start:
                continue
            if observation_end and period > observation_end:
                continue
            vintages = sorted(
                (o for o in self.history if o.observed_at == period),
                key=lambda o: o.as_of,
            )
            # Like FRED: a value in effect when the window opens is dated to its start.
            in_effect = [o for o in vintages if o.as_of < start]
            if in_effect:
                rows.append(replace(in_effect[-1], as_of=start))
            rows.extend(o for o in vintages if o.as_of >= start)
        return rows


def client(fake: FakeFRED) -> FREDClient:
    return fake  # type: ignore[return-value]


HISTORY = [
    Observation(JAN, V1, 210_000.0),
    Observation(JAN, V2, 212_000.0),  # revised
    Observation(FEB, V2, None),  # first print missing
    Observation(FEB, V3, 205_000.0),
]


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def count_rows(conn: Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return int(cur.fetchone()[0])  # type: ignore[index]


def test_full_ingest_stores_metadata_and_every_vintage(migrated: Connection) -> None:
    result = ingest_series(migrated, client(FakeFRED(HISTORY)), "ICSA")
    migrated.commit()

    assert (result.fetched, result.inserted, result.full) == (4, 4, True)
    with migrated.cursor() as cur:
        cur.execute(
            "SELECT so.name, s.name, s.frequency, s.seasonal_adjustment "
            "FROM series s JOIN sources so ON so.id = s.source_id"
        )
        assert cur.fetchone() == (
            "FRED",
            "Initial Claims",
            "Weekly, Ending Saturday",
            "Seasonally Adjusted",
        )
    history = get_vintage_history(migrated, "ICSA")
    assert (history["observed_at"] == "2024-01-01").sum() == 2  # two as_of dates
    # Point-in-time reads see the vintage known at the time.
    assert get_series_value_as_of(migrated, "ICSA", JAN, V1) == 210_000.0
    assert get_series_value_as_of(migrated, "ICSA", JAN, V3) == 212_000.0
    assert get_series_value_as_of(migrated, "ICSA", FEB, V2) is None
    assert get_series_value_as_of(migrated, "ICSA", FEB, V3) == 205_000.0


def test_full_reingest_is_idempotent(migrated: Connection) -> None:
    fake = FakeFRED(HISTORY)
    ingest_series(migrated, client(fake), "ICSA")
    result = ingest_series(migrated, client(fake), "ICSA", full=True)
    migrated.commit()

    assert (result.fetched, result.inserted) == (4, 0)
    assert count_rows(migrated, "observations") == 4
    assert count_rows(migrated, "series") == 1
    assert count_rows(migrated, "sources") == 1


def test_incremental_run_adds_only_new_vintages(migrated: Connection) -> None:
    fake = FakeFRED(HISTORY[:3])  # as of V2
    ingest_series(migrated, client(fake), "ICSA")

    v4 = date(2024, 5, 3)
    fake.history = HISTORY + [
        Observation(MAR, V3, 220_000.0),
        Observation(MAR, v4, 221_000.0),
    ]
    result = ingest_series(migrated, client(fake), "ICSA")
    migrated.commit()

    # JAN's value is re-reported at V3 by the windowed fetch, but unchanged: skipped.
    assert fake.observation_calls[-1]["realtime_start"] == V3
    assert (result.fetched, result.inserted, result.full) == (3, 3, False)
    history = get_vintage_history(migrated, "ICSA")
    assert len(history) == 6
    assert set(history["as_of"].dt.date) == {V1, V2, V3, v4}


def test_incremental_run_with_nothing_new_skips_fetch(migrated: Connection) -> None:
    fake = FakeFRED(HISTORY)
    ingest_series(migrated, client(fake), "ICSA")
    result = ingest_series(migrated, client(fake), "ICSA")

    assert (result.fetched, result.inserted) == (0, 0)
    assert len(fake.observation_calls) == 1


def test_rerun_refreshes_series_metadata(migrated: Connection) -> None:
    fake = FakeFRED(HISTORY)
    ingest_series(migrated, client(fake), "ICSA")
    fake.info = replace(INFO, units="Thousands")
    ingest_series(migrated, client(fake), "ICSA")

    with migrated.cursor() as cur:
        cur.execute("SELECT units FROM series WHERE series_id = 'ICSA'")
        assert cur.fetchone() == ("Thousands",)


def test_observation_range_is_passed_through(migrated: Connection) -> None:
    fake = FakeFRED(HISTORY)
    result = ingest_series(migrated, client(fake), "ICSA", observation_start=FEB)

    assert fake.observation_calls[0]["observation_start"] == FEB
    assert result.inserted == 2


@pytest.fixture
def pool(test_dsn: str) -> Iterator[None]:
    db.init_pool(test_dsn, maxconn=2)
    with db.connection() as conn:
        migrate(conn)
    try:
        yield
    finally:
        db.close_pool()


def test_ingest_many_commits_each_series_and_reports_failures(pool: None) -> None:
    results, failures = ingest_many(client(FakeFRED(HISTORY)), ["ICSA", "NOPE"])

    assert [r.series_id for r in results] == ["ICSA"]
    assert set(failures) == {"NOPE"}
    with db.connection() as conn:
        assert count_rows(conn, "observations") == 4  # ICSA was committed


load_dotenv()


@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set")
def test_live_ingest_recent_cpi(migrated: Connection) -> None:
    start = datetime.now(UTC).date() - timedelta(days=730)
    with FREDClient.from_env() as fred:
        result = ingest_series(migrated, fred, "CPIAUCSL", observation_start=start)
        # No vintages since: FRED's vintagedates would 500 if asked for that range.
        again = ingest_series(migrated, fred, "CPIAUCSL")
    migrated.commit()

    assert result.inserted > 0
    assert (again.fetched, again.inserted) == (0, 0)
    history = get_vintage_history(migrated, "CPIAUCSL")
    per_period = history.groupby("observed_at")["as_of"].nunique()
    assert (per_period > 1).any()  # seasonal revisions give periods several vintages

    # Point-in-time: a period's value as of its first release is the first print.
    row = history.sort_values(["observed_at", "as_of"]).iloc[0]
    assert get_series_value_as_of(
        migrated, "CPIAUCSL", row["observed_at"].date(), row["as_of"].date()
    ) == pytest.approx(row["value"])

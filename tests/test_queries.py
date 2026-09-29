"""Tests for point-in-time queries and the connection pool (database fixtures in conftest.py)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.db import connection as db
from eco_prediction.db.migrate import migrate
from eco_prediction.db.queries import (
    get_multiple_series_as_of,
    get_series_as_of,
    get_series_value_as_of,
    get_vintage_history,
)

JAN, FEB, MAR = date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)

# (series, observed_at, as_of, value). UNRATE Jan is revised once; Feb's first
# print is missing (NULL) and later filled in; Mar is published last.
OBSERVATIONS = [
    ("UNRATE", JAN, date(2024, 2, 2), 3.7),
    ("UNRATE", JAN, date(2024, 3, 8), 3.8),
    ("UNRATE", FEB, date(2024, 3, 8), None),
    ("UNRATE", FEB, date(2024, 4, 5), 3.9),
    ("UNRATE", MAR, date(2024, 4, 5), 3.8),
    ("CPIAUCSL", JAN, date(2024, 2, 13), 308.4),
    ("CPIAUCSL", FEB, date(2024, 3, 12), 310.3),
]


@pytest.fixture
def seeded(conn: Connection) -> Connection:
    migrate(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sources (name) VALUES ('FRED') RETURNING id")
        source_id = cur.fetchone()[0]  # type: ignore[index]
        pks = {}
        for code in ("UNRATE", "CPIAUCSL", "EMPTY"):
            cur.execute(
                "INSERT INTO series (source_id, series_id) VALUES (%s, %s) RETURNING id",
                (source_id, code),
            )
            pks[code] = cur.fetchone()[0]  # type: ignore[index]
        cur.executemany(
            "INSERT INTO observations (series_id, observed_at, as_of, value) VALUES (%s, %s, %s, %s)",
            [(pks[code], obs, as_of, v) for code, obs, as_of, v in OBSERVATIONS],
        )
    conn.commit()
    return conn


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2024, 2, 1), None),  # before first print
        (date(2024, 2, 2), 3.7),  # release day counts
        (date(2024, 3, 7), 3.7),
        (date(2024, 3, 8), 3.8),  # revision
        (date(2025, 1, 1), 3.8),
    ],
)
def test_scalar_value_as_of(
    seeded: Connection, as_of: date, expected: float | None
) -> None:
    assert get_series_value_as_of(seeded, "UNRATE", JAN, as_of) == expected


def test_scalar_unknown_series(seeded: Connection) -> None:
    assert get_series_value_as_of(seeded, "NOPE", JAN, date(2025, 1, 1)) is None


def test_series_as_of_hides_later_vintages(seeded: Connection) -> None:
    s = get_series_as_of(seeded, "UNRATE", date(2024, 3, 20))
    assert s.name == "UNRATE"
    assert list(s.index) == [pd.Timestamp(JAN), pd.Timestamp(FEB)]
    assert s[pd.Timestamp(JAN)] == 3.8
    assert math.isnan(s[pd.Timestamp(FEB)])  # first print was missing

    latest = get_series_as_of(seeded, "UNRATE", date(2024, 12, 31))
    assert latest.tolist() == [3.8, 3.9, 3.8]


def test_series_as_of_date_range(seeded: Connection) -> None:
    s = get_series_as_of(seeded, "UNRATE", date(2024, 12, 31), start=FEB, end=FEB)
    assert s.tolist() == [3.9]


def test_series_as_of_empty(seeded: Connection) -> None:
    for code in ("EMPTY", "NOPE"):
        s = get_series_as_of(seeded, code, date(2024, 12, 31))
        assert s.empty and s.dtype == "float64" and s.name == code
        assert isinstance(s.index, pd.DatetimeIndex)
    assert get_series_as_of(seeded, "UNRATE", date(2020, 1, 1)).empty


def test_multiple_series_as_of(seeded: Connection) -> None:
    df = get_multiple_series_as_of(
        seeded, ["UNRATE", "CPIAUCSL", "EMPTY"], date(2024, 3, 10)
    )
    assert sorted(df.columns) == ["CPIAUCSL", "UNRATE"]
    assert list(df.index) == [pd.Timestamp(JAN), pd.Timestamp(FEB)]
    assert df.loc[pd.Timestamp(JAN), "UNRATE"] == 3.8
    assert pd.isna(df.loc[pd.Timestamp(FEB), "UNRATE"])
    # CPI Feb not released until 3/12.
    assert df.loc[pd.Timestamp(JAN), "CPIAUCSL"] == 308.4
    assert pd.isna(df.loc[pd.Timestamp(FEB), "CPIAUCSL"])


def test_vintage_history(seeded: Connection) -> None:
    h = get_vintage_history(seeded, "UNRATE", end=FEB)
    assert list(h.columns) == ["observed_at", "as_of", "value"]
    assert h["as_of"].dt.date.tolist() == [
        date(2024, 2, 2),
        date(2024, 3, 8),
        date(2024, 3, 8),
        date(2024, 4, 5),
    ]
    assert h["value"].tolist()[:2] == [3.7, 3.8]
    assert get_vintage_history(seeded, "NOPE").empty


@pytest.fixture
def pool(test_dsn: str) -> Iterator[None]:
    db.init_pool(test_dsn, minconn=1, maxconn=1)
    try:
        yield
    finally:
        db.close_pool()


def test_pool_commits_on_success(pool: None) -> None:
    with db.cursor() as cur:
        cur.execute("CREATE TABLE t (x INT)")
        cur.execute("INSERT INTO t VALUES (1)")
    with db.cursor() as cur:
        cur.execute("SELECT x FROM t")
        assert cur.fetchall() == [(1,)]


def test_pool_rolls_back_on_error(pool: None) -> None:
    with db.cursor() as cur:
        cur.execute("CREATE TABLE t (x INT)")
    with pytest.raises(RuntimeError), db.cursor() as cur:
        cur.execute("INSERT INTO t VALUES (1)")
        raise RuntimeError("boom")
    # maxconn=1, so this is the same connection; it must be usable and clean.
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM t")
        assert cur.fetchone() == (0,)


def test_pool_discards_closed_connection(pool: None) -> None:
    with pytest.raises(RuntimeError), db.connection() as conn:
        conn.close()
        raise RuntimeError("connection lost")
    with db.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)

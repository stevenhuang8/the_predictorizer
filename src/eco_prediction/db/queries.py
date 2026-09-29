"""Point-in-time reads over vintage-tracked observations.

Every function takes an `as_of` date and returns data exactly as it was known
on that date: for each period, the latest vintage with `as_of <= as_of`.
Revisions published later are invisible, which is what keeps backtests free
of lookahead bias.

A period whose chosen vintage has a NULL value (the source reported it
missing) comes back as NaN in Series/DataFrames and None from the scalar
lookup. Series codes are the source's codes, e.g. 'CPIAUCSL'.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
from psycopg2.extensions import connection as Connection


def get_series_value_as_of(
    conn: Connection, series_id: str, observed_date: date, as_of_date: date
) -> float | None:
    """Value of one period of a series as it was known on as_of_date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.value::float8
            FROM observations o
            JOIN series s ON o.series_id = s.id
            WHERE s.series_id = %s
              AND o.observed_at = %s
              AND o.as_of <= %s
            ORDER BY o.as_of DESC
            LIMIT 1
            """,
            (series_id, observed_date, as_of_date),
        )
        row = cur.fetchone()
    return None if row is None else row[0]


def get_series_as_of(
    conn: Connection,
    series_id: str,
    as_of_date: date,
    start: date | None = None,
    end: date | None = None,
) -> pd.Series:
    """Whole series (optionally limited to observed_at in [start, end]) as known on as_of_date.

    Returns a float Series indexed by observed_at (DatetimeIndex), named series_id.
    Periods with no vintage published by as_of_date are absent.
    """
    frame = get_multiple_series_as_of(conn, [series_id], as_of_date, start, end)
    if series_id in frame:
        return frame[series_id]
    return pd.Series(
        dtype="float64", name=series_id, index=pd.DatetimeIndex([], name="observed_at")
    )


def get_multiple_series_as_of(
    conn: Connection,
    series_ids: Sequence[str],
    as_of_date: date,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Several series as known on as_of_date, one column per series.

    Rows are the union of observed_at dates across series (DatetimeIndex); a
    series with no value for a date is NaN there. Series with no data at all
    are omitted from the columns.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (s.series_id, o.observed_at)
                   s.series_id, o.observed_at, o.value::float8
            FROM observations o
            JOIN series s ON o.series_id = s.id
            WHERE s.series_id = ANY(%s)
              AND o.as_of <= %s
              AND (%s::date IS NULL OR o.observed_at >= %s::date)
              AND (%s::date IS NULL OR o.observed_at <= %s::date)
            ORDER BY s.series_id, o.observed_at, o.as_of DESC
            """,
            (list(series_ids), as_of_date, start, start, end, end),
        )
        rows = cur.fetchall()
    long = pd.DataFrame(rows, columns=["series_id", "observed_at", "value"])
    wide = long.pivot(index="observed_at", columns="series_id", values="value")
    wide.index = pd.DatetimeIndex(wide.index, name="observed_at")
    wide.columns.name = None
    return wide.astype("float64").sort_index()


def get_vintage_history(
    conn: Connection,
    series_id: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Every published vintage of a series, for revision analysis.

    Returns columns observed_at, as_of, value, sorted by observed_at then as_of.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.observed_at, o.as_of, o.value::float8
            FROM observations o
            JOIN series s ON o.series_id = s.id
            WHERE s.series_id = %s
              AND (%s::date IS NULL OR o.observed_at >= %s::date)
              AND (%s::date IS NULL OR o.observed_at <= %s::date)
            ORDER BY o.observed_at, o.as_of
            """,
            (series_id, start, start, end, end),
        )
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    frame["value"] = frame["value"].astype("float64")
    return frame

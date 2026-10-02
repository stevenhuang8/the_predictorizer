"""Load FRED series, with every ALFRED vintage, into the database.

Usage:
    uv run python -m eco_prediction.data.ingest                 # all V1_SERIES
    uv run python -m eco_prediction.data.ingest CPIAUCSL UNRATE # just these
    uv run python -m eco_prediction.data.ingest --full          # refetch all history

The first run for a series backfills its full vintage history. Later runs are
incremental: only vintages published after the latest stored `as_of` are
fetched, and rows that merely repeat the stored value are skipped (FRED
re-reports values already in effect when a real-time window opens). `--full`
refetches everything; inserts use ON CONFLICT DO NOTHING, so any run can be
repeated safely.

Each series is loaded in its own transaction: a failure rolls back that series
only and the run moves on to the next.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values

from eco_prediction.data.fred_client import (
    FREDAPIError,
    FREDClient,
    Observation,
    SeriesInfo,
    drop_unchanged,
)
from eco_prediction.db.connection import ConfigError, connection

FRED_SOURCE = (
    "FRED",
    "https://fred.stlouisfed.org",
    "Federal Reserve Economic Data, St. Louis Fed (vintages from ALFRED)",
)

V1_SERIES = (
    "CPIAUCSL",  # CPI, all items (SA)
    "UNRATE",  # unemployment rate
    "T10Y2Y",  # 10y minus 2y Treasury spread
    "T10Y3M",  # 10y minus 3m Treasury spread
    "DCOILWTICO",  # WTI crude oil, daily
    "GASREGW",  # regular gasoline, weekly
    "PPIACO",  # PPI, all commodities
    "CES0500000003",  # average hourly earnings, total private (wage growth)
    "ICSA",  # initial jobless claims
    "MICH",  # Michigan 1-year inflation expectations
    "CUUR0000SEHA",  # CPI rent of primary residence (NSA)
    "CUSR0000SEHA",  # CPI rent of primary residence (SA)
)


@dataclass(frozen=True)
class IngestResult:
    series_id: str
    fetched: int  # observation rows kept after dropping repeats
    inserted: int  # rows actually new to the database
    full: bool  # True if the whole vintage history was requested


def ensure_source(
    conn: Connection, name: str, url: str | None = None, description: str | None = None
) -> int:
    """Primary key of the named source, creating it if needed."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (name, url, description) VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE
                SET url = COALESCE(EXCLUDED.url, sources.url),
                    description = COALESCE(EXCLUDED.description, sources.description)
            RETURNING id
            """,
            (name, url, description),
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def upsert_series(conn: Connection, source_id: int, info: SeriesInfo) -> int:
    """Primary key of the series, inserting it or refreshing its metadata."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO series
                (source_id, series_id, name, units, frequency, seasonal_adjustment)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (series_id) DO UPDATE
                SET name = EXCLUDED.name,
                    units = EXCLUDED.units,
                    frequency = EXCLUDED.frequency,
                    seasonal_adjustment = EXCLUDED.seasonal_adjustment
            RETURNING id
            """,
            (
                source_id,
                info.series_id,
                info.name,
                info.units,
                info.frequency,
                info.seasonal_adjustment,
            ),
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def insert_observations(
    conn: Connection, series_pk: int, observations: Iterable[Observation]
) -> int:
    """Insert vintages, skipping any already stored. Returns the number inserted."""
    rows = [(series_pk, o.observed_at, o.as_of, o.value) for o in observations]
    if not rows:
        return 0
    with conn.cursor() as cur:
        inserted = execute_values(
            cur,
            """
            INSERT INTO observations (series_id, observed_at, as_of, value) VALUES %s
            ON CONFLICT (series_id, observed_at, as_of) DO NOTHING
            RETURNING 1
            """,
            rows,
            page_size=10_000,
            fetch=True,
        )
    return len(inserted)


def latest_stored(
    conn: Connection, series_pk: int
) -> tuple[date | None, dict[date, float | None]]:
    """Latest stored as_of for the series, and the current value of each period."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (observed_at) observed_at, as_of, value::float8
            FROM observations
            WHERE series_id = %s
            ORDER BY observed_at, as_of DESC
            """,
            (series_pk,),
        )
        rows = cur.fetchall()
    latest = max((as_of for _, as_of, _ in rows), default=None)
    return latest, {observed_at: value for observed_at, _, value in rows}


def ingest_series(
    conn: Connection,
    client: FREDClient,
    series_id: str,
    *,
    full: bool = False,
    observation_start: date | None = None,
    observation_end: date | None = None,
) -> IngestResult:
    """Fetch one series and store its metadata and new vintages. Does not commit.

    Incremental runs only add vintages for periods within the requested
    observation range; use `full=True` to backfill periods outside it.
    """
    info = client.get_series_info(series_id)
    source_pk = ensure_source(conn, *FRED_SOURCE)
    series_pk = upsert_series(conn, source_pk, info)

    since, prior = (None, {}) if full else latest_stored(conn, series_pk)
    if since is None:
        observations = client.get_series_observations(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
    else:
        new_vintages = client.get_vintage_dates(
            series_id, realtime_start=since + timedelta(days=1)
        )
        if not new_vintages:
            return IngestResult(series_id, 0, 0, full=False)
        # Rows already in effect on new_vintages[0] come back dated to it;
        # drop_unchanged removes those against what is stored.
        observations = drop_unchanged(
            client.get_series_observations(
                series_id,
                realtime_start=new_vintages[0],
                observation_start=observation_start,
                observation_end=observation_end,
            ),
            prior,
        )

    inserted = insert_observations(conn, series_pk, observations)
    return IngestResult(series_id, len(observations), inserted, full=since is None)


def ingest_many(
    client: FREDClient,
    series_ids: Sequence[str] = V1_SERIES,
    *,
    full: bool = False,
    observation_start: date | None = None,
    observation_end: date | None = None,
) -> tuple[list[IngestResult], dict[str, Exception]]:
    """Ingest each series in its own pooled transaction.

    Returns the successful results and the errors of any series that failed.
    """
    results: list[IngestResult] = []
    failures: dict[str, Exception] = {}
    for series_id in series_ids:
        try:
            with connection() as conn:
                results.append(
                    ingest_series(
                        conn,
                        client,
                        series_id,
                        full=full,
                        observation_start=observation_start,
                        observation_end=observation_end,
                    )
                )
        except FREDAPIError as exc:
            failures[series_id] = exc
    return results, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load FRED series and their vintages into the database."
    )
    parser.add_argument(
        "series",
        nargs="*",
        default=list(V1_SERIES),
        help="FRED series IDs (default: V1 set)",
    )
    parser.add_argument(
        "--full", action="store_true", help="refetch the full vintage history"
    )
    parser.add_argument(
        "--observation-start",
        type=date.fromisoformat,
        help="only periods on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--observation-end",
        type=date.fromisoformat,
        help="only periods on or before this date (YYYY-MM-DD)",
    )
    args = parser.parse_args(argv)

    try:
        with FREDClient.from_env() as client:
            results, failures = ingest_many(
                client,
                args.series,
                full=args.full,
                observation_start=args.observation_start,
                observation_end=args.observation_end,
            )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for r in results:
        mode = "full" if r.full else "incremental"
        print(
            f"{r.series_id:15} {mode:11} fetched={r.fetched:<7} inserted={r.inserted}"
        )
    for series_id, error in failures.items():
        print(f"{series_id:15} FAILED      {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

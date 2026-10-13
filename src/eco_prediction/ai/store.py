"""Storing AI-domain series and observations.

The economy's `data/ingest.py` writes FRED metadata and never sets `domain`
or `reconstructed`; these helpers do, so AI ingestion leaves it untouched.
Sources are registered with the economy's `ensure_source`, re-exported here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values

from eco_prediction.data.ingest import ensure_source, latest_stored

__all__ = [
    "AIObservation",
    "AISeriesInfo",
    "ensure_source",
    "insert_ai_observations",
    "latest_stored",
    "upsert_ai_series",
]


@dataclass(frozen=True)
class AISeriesInfo:
    series_id: str
    name: str
    units: str
    frequency: str
    seasonal_adjustment: str = "Not Seasonally Adjusted"


@dataclass(frozen=True)
class AIObservation:
    observed_at: date  # period the value describes
    as_of: date  # when the value was (or, if reconstructed, would have been) known
    value: float | None
    reconstructed: bool  # True = backfilled from timestamped records, not captured live


def upsert_ai_series(conn: Connection, source_pk: int, info: AISeriesInfo) -> int:
    """Primary key of the series, inserting it with domain 'ai' or refreshing it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO series
                (source_id, series_id, name, units, frequency,
                 seasonal_adjustment, domain)
            VALUES (%s, %s, %s, %s, %s, %s, 'ai')
            ON CONFLICT (series_id) DO UPDATE
                SET name = EXCLUDED.name,
                    units = EXCLUDED.units,
                    frequency = EXCLUDED.frequency,
                    seasonal_adjustment = EXCLUDED.seasonal_adjustment
            RETURNING id, domain
            """,
            (
                source_pk,
                info.series_id,
                info.name,
                info.units,
                info.frequency,
                info.seasonal_adjustment,
            ),
        )
        pk, domain = cur.fetchone()  # type: ignore[misc]
    if domain != "ai":
        raise ValueError(f"series {info.series_id!r} exists in domain {domain!r}")
    return int(pk)


def insert_ai_observations(
    conn: Connection, series_pk: int, observations: Iterable[AIObservation]
) -> int:
    """Insert rows, skipping any (period, as_of) already stored. Returns the count."""
    rows = [
        (series_pk, o.observed_at, o.as_of, o.value, o.reconstructed)
        for o in observations
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        inserted = execute_values(
            cur,
            """
            INSERT INTO observations
                (series_id, observed_at, as_of, value, reconstructed)
            VALUES %s
            ON CONFLICT (series_id, observed_at, as_of) DO NOTHING
            RETURNING 1
            """,
            rows,
            page_size=10_000,
            fetch=True,
        )
    return len(inserted)

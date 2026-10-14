"""Storing forecasting-platform questions and probabilities (migration 011).

Platform forecasts are benchmarks for ours, not targets, so they have their
own tables rather than `series`/`observations`. Shared by every platform:
Polymarket now, Metaculus once it's allowed (see AI_NOTES/Task8.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json, execute_values


@dataclass(frozen=True)
class PlatformQuestion:
    platform: str  # 'polymarket' or 'metaculus'
    external_id: str
    title: str
    resolution_criteria: str | None = None
    url: str | None = None
    opened_at: datetime | None = None
    closes_at: datetime | None = None
    resolved_at: datetime | None = None
    resolved_outcome: str | None = None  # 'yes', 'no' or None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlatformProbability:
    as_of: date
    probability: float
    reconstructed: bool  # True = from the platform's history, not captured live


def upsert_platform_question(
    conn: Connection, q: PlatformQuestion
) -> tuple[int, date | None, bool]:
    """(primary key, prices_through, whether it was new). Refreshes every field
    from the platform's current state, the outcome included."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO platform_questions
                (platform, external_id, title, resolution_criteria, url,
                 opened_at, closes_at, resolved_at, resolved_outcome, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (platform, external_id) DO UPDATE
                SET title = EXCLUDED.title,
                    resolution_criteria = EXCLUDED.resolution_criteria,
                    url = EXCLUDED.url,
                    opened_at = EXCLUDED.opened_at,
                    closes_at = EXCLUDED.closes_at,
                    resolved_at = EXCLUDED.resolved_at,
                    resolved_outcome = EXCLUDED.resolved_outcome,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            RETURNING id, prices_through, (xmax = 0)
            """,
            (
                q.platform,
                q.external_id,
                q.title,
                q.resolution_criteria,
                q.url,
                q.opened_at,
                q.closes_at,
                q.resolved_at,
                q.resolved_outcome,
                Json(q.metadata),
            ),
        )
        pk, through, inserted = cur.fetchone()  # type: ignore[misc]
    return int(pk), through, bool(inserted)


def insert_platform_probabilities(
    conn: Connection, question_pk: int, rows: Iterable[PlatformProbability]
) -> int:
    """Insert rows, keeping any already stored for the same day. Returns the count."""
    values = [(question_pk, r.as_of, r.probability, r.reconstructed) for r in rows]
    if not values:
        return 0
    with conn.cursor() as cur:
        inserted = execute_values(
            cur,
            """
            INSERT INTO platform_probabilities
                (question_id, as_of, probability, reconstructed)
            VALUES %s
            ON CONFLICT (question_id, as_of) DO NOTHING
            RETURNING 1
            """,
            values,
            page_size=10_000,
            fetch=True,
        )
    return len(inserted)


def set_prices_through(conn: Connection, question_pk: int, day: date) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE platform_questions SET prices_through = %s WHERE id = %s",
            (day, question_pk),
        )

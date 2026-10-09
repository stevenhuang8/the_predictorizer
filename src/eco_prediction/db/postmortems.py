"""Reading forecasts to review and writing their post-mortems.

Usage:
    from eco_prediction.db.postmortems import (
        forecasts_to_review, list_postmortems, save_postmortem,
    )

    for f in forecasts_to_review(conn):        # scored, not manually classified
        save_postmortem(conn, f.forecast_id, "bad_model", notes="...", evidence={...})
    save_postmortem(conn, 42, "regime_change", notes="COVID", classified_by="manual")
    list_postmortems(conn)                     # DataFrame, newest target first

An 'auto' save never replaces a 'manual' row; a 'manual' save replaces
anything. Like the rest of `db`, these functions don't commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd
from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

from eco_prediction.db.forecasts import json_value

MissCategory = Literal["bad_data", "bad_model", "regime_change", "expected_variance"]
MISS_CATEGORIES: tuple[MissCategory, ...] = (
    "bad_data",
    "bad_model",
    "regime_change",
    "expected_variance",
)


@dataclass(frozen=True)
class ForecastToReview:
    forecast_id: int
    target: str
    target_date: date
    forecast_date: date
    model_type: str
    prediction: float | None
    interval_lower: float | None
    interval_upper: float | None
    probabilities: dict[str, float] | None
    actual_value: float | None
    actual_outcome: str | None
    actual_as_of: date


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def forecasts_to_review(conn: Connection) -> list[ForecastToReview]:
    """Scored forecasts with no post-mortem, or only an automatic one."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, q.target::text, q.target_date, f.forecast_date, m.model_type,
                   f.prediction, f.interval_lower, f.interval_upper, f.probabilities,
                   r.actual_value, r.actual_outcome, r.actual_as_of
            FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            JOIN resolutions r ON r.question_id = q.id
            JOIN model_versions m ON m.id = f.model_version_id
            LEFT JOIN postmortems p ON p.forecast_id = f.id
            WHERE f.scored_at IS NOT NULL
              AND (p.id IS NULL OR p.classified_by = 'auto')
            ORDER BY f.id
            """
        )
        rows = cur.fetchall()
    return [
        ForecastToReview(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            _float(row[5]),
            _float(row[6]),
            _float(row[7]),
            row[8],
            _float(row[9]),
            row[10],
            row[11],
        )
        for row in rows
    ]


def save_postmortem(
    conn: Connection,
    forecast_id: int,
    category: MissCategory,
    *,
    notes: str | None = None,
    revised_data_impact: float | None = None,
    evidence: Mapping[str, Any] | None = None,
    classified_by: Literal["auto", "manual"] = "auto",
) -> bool:
    """Insert or update a forecast's post-mortem; False if a manual one blocked it."""
    if category not in MISS_CATEGORIES:
        raise ValueError(f"category must be one of {MISS_CATEGORIES}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO postmortems (forecast_id, miss_category, notes,
                                     revised_data_impact, evidence, classified_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (forecast_id) DO UPDATE SET
                miss_category = EXCLUDED.miss_category,
                notes = EXCLUDED.notes,
                revised_data_impact = EXCLUDED.revised_data_impact,
                evidence = EXCLUDED.evidence,
                classified_by = EXCLUDED.classified_by,
                updated_at = NOW()
            WHERE postmortems.classified_by = 'auto' OR EXCLUDED.classified_by = 'manual'
            RETURNING id
            """,
            (
                forecast_id,
                category,
                notes,
                revised_data_impact,
                Json(json_value(evidence)) if evidence is not None else None,
                classified_by,
            ),
        )
        return cur.fetchone() is not None


def list_postmortems(conn: Connection) -> pd.DataFrame:
    """Every post-mortem with its forecast and question, newest target first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.forecast_id, q.target::text AS target, q.target_date,
                   q.horizon_months, q.lead_days, m.model_type, f.is_backtest,
                   p.miss_category::text AS category, p.classified_by,
                   p.revised_data_impact, p.notes
            FROM postmortems p
            JOIN forecasts f ON f.id = p.forecast_id
            JOIN questions q ON q.id = f.question_id
            JOIN model_versions m ON m.id = f.model_version_id
            ORDER BY q.target_date DESC, q.target, m.model_type
            """
        )
        columns = [c.name for c in cur.description or []]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    frame["revised_data_impact"] = frame["revised_data_impact"].astype("float64")
    return frame

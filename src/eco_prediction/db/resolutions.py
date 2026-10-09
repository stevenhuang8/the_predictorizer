"""Resolving questions and storing forecast scores.

Usage:
    from eco_prediction.db.resolutions import (
        save_resolution, save_score, scored_forecasts, unresolved_questions,
        unscored_forecasts,
    )

    for q in unresolved_questions(conn):            # OpenQuestion(id, target, ...)
        save_resolution(conn, q.id, actual_as_of=date(2026, 12, 10), value=3.1)
    for f in unscored_forecasts(conn):              # forecasts of resolved questions
        save_score(conn, f.id, error=..., in_interval=..., brier_score=None)
    for q in numeric_resolutions(conn):             # refresh revised values
        save_latest(conn, q.question_id, value=3.5, as_of=date(2027, 2, 10))
    scored_forecasts(conn)                          # DataFrame for score summaries

Like the rest of `db`, these functions don't commit; the caller does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from psycopg2.extensions import connection as Connection


@dataclass(frozen=True)
class OpenQuestion:
    id: int
    target: str
    target_date: date
    horizon_months: int | None
    lead_days: int | None


@dataclass(frozen=True)
class UnscoredForecast:
    """A forecast whose question has resolved, with what it needs to be scored."""

    id: int
    prediction: float | None
    interval_lower: float | None
    interval_upper: float | None
    probabilities: dict[str, float] | None
    actual_value: float | None
    actual_outcome: str | None


def unresolved_questions(conn: Connection) -> list[OpenQuestion]:
    """Questions with no resolution yet, oldest target first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.target_date, q.horizon_months, q.lead_days
            FROM questions q
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE r.id IS NULL
            ORDER BY q.target_date, q.id
            """
        )
        return [OpenQuestion(*row) for row in cur.fetchall()]


def save_resolution(
    conn: Connection,
    question_id: int,
    *,
    actual_as_of: date,
    value: float | None = None,
    outcome: str | None = None,
) -> int:
    """Record a question's actual value (numeric) or outcome (FOMC)."""
    if (value is None) == (outcome is None):
        raise ValueError("give exactly one of value and outcome")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resolutions
                (question_id, actual_value, actual_outcome, actual_as_of, resolved_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (question_id, value, outcome, actual_as_of),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


@dataclass(frozen=True)
class NumericResolution:
    question_id: int
    target: str
    target_date: date
    latest_as_of: date | None


def numeric_resolutions(conn: Connection) -> list[NumericResolution]:
    """Resolved numeric questions, with the vintage of their stored latest value."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.target_date, r.latest_as_of
            FROM resolutions r JOIN questions q ON q.id = r.question_id
            WHERE r.actual_value IS NOT NULL
            ORDER BY q.target_date, q.id
            """
        )
        return [NumericResolution(*row) for row in cur.fetchall()]


def save_latest(
    conn: Connection, question_id: int, *, value: float, as_of: date
) -> None:
    """Record the latest published value of a resolved question's answer."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE resolutions SET latest_value = %s, latest_as_of = %s
            WHERE question_id = %s
            """,
            (value, as_of, question_id),
        )


def unscored_forecasts(conn: Connection) -> list[UnscoredForecast]:
    """Forecasts of resolved questions that haven't been scored."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.prediction, f.interval_lower, f.interval_upper,
                   f.probabilities, r.actual_value, r.actual_outcome
            FROM forecasts f
            JOIN resolutions r ON r.question_id = f.question_id
            WHERE f.scored_at IS NULL
            ORDER BY f.id
            """
        )
        rows = cur.fetchall()
    return [
        UnscoredForecast(
            id,
            _float(prediction),
            _float(lower),
            _float(upper),
            probabilities,
            _float(actual_value),
            actual_outcome,
        )
        for id, prediction, lower, upper, probabilities, actual_value, actual_outcome in rows
    ]


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def save_score(
    conn: Connection,
    forecast_id: int,
    *,
    error: float | None = None,
    in_interval: bool | None = None,
    brier_score: float | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE forecasts
            SET error = %s, in_interval = %s, brier_score = %s, scored_at = NOW()
            WHERE id = %s
            """,
            (error, in_interval, brier_score, forecast_id),
        )


def scored_forecasts(
    conn: Connection, *, include_backtest: bool = False
) -> pd.DataFrame:
    """Every scored forecast with its question, model and actual, one row each."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.target::text AS target, q.target_date, q.horizon_months,
                   q.lead_days, m.model_type, m.parameters, f.id AS forecast_id,
                   f.forecast_date, f.is_backtest, f.prediction, f.interval_lower,
                   f.interval_upper, f.error, f.in_interval, f.brier_score,
                   r.actual_value, r.actual_outcome, r.latest_value
            FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            JOIN resolutions r ON r.question_id = q.id
            JOIN model_versions m ON m.id = f.model_version_id
            WHERE f.scored_at IS NOT NULL AND (%s OR NOT f.is_backtest)
            ORDER BY q.target, q.target_date, m.model_type
            """,
            (include_backtest,),
        )
        columns = [c.name for c in cur.description or []]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    numeric = [
        "prediction", "interval_lower", "interval_upper", "error", "brier_score",
        "actual_value", "latest_value",
    ]  # fmt: skip
    frame[numeric] = frame[numeric].astype("float64")
    return frame

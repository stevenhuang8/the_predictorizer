"""Resolving questions and storing forecast scores.

Usage:
    from eco_prediction.db.resolutions import (
        save_resolution, save_score, scored_forecasts, unresolved_questions,
        unscored_forecasts,
    )

    for q in unresolved_questions(conn):            # OpenQuestion(id, target, ...)
        save_resolution(conn, q.id, actual_as_of=date(2026, 12, 10), value=3.1)
    resolve_binary_question(conn, question_id, "yes", date(2027, 1, 1))  # AI
    for f in unscored_forecasts(conn):              # forecasts of resolved questions
        save_score(conn, f.id, error=..., in_interval=..., brier_score=None)
    for q in numeric_resolutions(conn):             # refresh revised values
        save_latest(conn, q.question_id, value=3.5, as_of=date(2027, 2, 10))
    scored_forecasts(conn)                          # DataFrame for score summaries

Like the rest of `db`, these functions don't commit; the caller does.

The readers that feed the economy's resolution job (`unresolved_questions`,
`numeric_resolutions`, `unscored_forecasts`, `scored_forecasts`) default to
the economy domain; the AI domain resolves and scores its questions with
`ai/resolutions.py` (AI task 11) and passes `domain="ai"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

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
    question_type: str = "numeric_interval"  # 'binary' is scored differently


# The answer each question type takes (migrations 003 and 010).
OUTCOMES = {
    "numeric_interval": None,  # a value
    "probability": frozenset({"cut", "hold", "hike"}),  # FOMC
    "binary": frozenset({"yes", "no"}),
}


def unresolved_questions(conn: Connection) -> list[OpenQuestion]:
    """Economy questions with no resolution yet, oldest target first.

    AI questions resolve from other series (AI task 11), so they're left out.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.target_date, q.horizon_months, q.lead_days
            FROM questions q
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE r.id IS NULL AND q.domain = 'economy'
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
    """Record a question's actual value (numeric) or outcome (FOMC, binary).

    Raises ValueError if the answer doesn't fit the question's type: a value
    for numeric questions, cut/hold/hike for FOMC, yes/no for binary ones.
    """
    if (value is None) == (outcome is None):
        raise ValueError("give exactly one of value and outcome")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT question_type::text FROM questions WHERE id = %s", (question_id,)
        )
        found = cur.fetchone()
        if found is None:
            raise ValueError(f"no question with id {question_id}")
        allowed = OUTCOMES[found[0]]
        if allowed is None and outcome is not None:
            raise ValueError(f"question {question_id} is numeric; give a value")
        if allowed is not None and outcome not in allowed:
            raise ValueError(
                f"question {question_id} ({found[0]}) resolves to one of "
                f"{sorted(allowed)}, not {outcome if value is None else value!r}"
            )
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


def resolve_binary_question(
    conn: Connection,
    question_id: int,
    outcome: Literal["yes", "no"],
    as_of_date: date,
) -> int:
    """Record a binary question's yes/no outcome, known as of as_of_date.

    Raises ValueError if the question isn't binary or the outcome isn't
    yes/no, and psycopg2's UniqueViolation if it's already resolved.
    """
    return save_resolution(conn, question_id, actual_as_of=as_of_date, outcome=outcome)


@dataclass(frozen=True)
class NumericResolution:
    question_id: int
    target: str
    target_date: date
    latest_as_of: date | None


def numeric_resolutions(conn: Connection) -> list[NumericResolution]:
    """Resolved numeric economy questions, with the vintage of their latest value."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.target_date, r.latest_as_of
            FROM resolutions r JOIN questions q ON q.id = r.question_id
            WHERE r.actual_value IS NOT NULL AND q.domain = 'economy'
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


def unscored_forecasts(
    conn: Connection, *, domain: str = "economy"
) -> list[UnscoredForecast]:
    """Forecasts of resolved questions in `domain` that haven't been scored."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.prediction, f.interval_lower, f.interval_upper,
                   f.probabilities, r.actual_value, r.actual_outcome,
                   q.question_type::text
            FROM forecasts f
            JOIN resolutions r ON r.question_id = f.question_id
            JOIN questions q ON q.id = f.question_id
            WHERE f.scored_at IS NULL AND q.domain = %s
            ORDER BY f.id
            """,
            (domain,),
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
            question_type,
        )
        for (
            id, prediction, lower, upper, probabilities, actual_value, actual_outcome,
            question_type,
        ) in rows
    ]  # fmt: skip


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
    conn: Connection, *, include_backtest: bool = False, domain: str = "economy"
) -> pd.DataFrame:
    """Every scored forecast in `domain` with its question, model and actual."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.target::text AS target, q.target_date, q.horizon_months,
                   q.lead_days, q.question_type::text AS question_type,
                   q.question_key, m.model_type, m.parameters, f.id AS forecast_id,
                   f.forecast_date, f.is_backtest, f.prediction, f.interval_lower,
                   f.interval_upper, f.error, f.in_interval, f.brier_score,
                   r.actual_value, r.actual_outcome, r.latest_value
            FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            JOIN resolutions r ON r.question_id = q.id
            JOIN model_versions m ON m.id = f.model_version_id
            WHERE f.scored_at IS NOT NULL AND (%s OR NOT f.is_backtest)
              AND q.domain = %s
            ORDER BY q.target, q.target_date, m.model_type
            """,
            (include_backtest, domain),
        )
        columns = [c.name for c in cur.description or []]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    numeric = [
        "prediction", "interval_lower", "interval_upper", "error", "brier_score",
        "actual_value", "latest_value",
    ]  # fmt: skip
    frame[numeric] = frame[numeric].astype("float64")
    return frame

"""Resolving AI questions and scoring their forecasts (AI task 11).

    from eco_prediction.ai.resolutions import resolve_and_score_ai
    result = resolve_and_score_ai(conn, date(2026, 12, 10))
    conn.commit()

The economy's resolution job only reads economy questions; this is the AI
counterpart. AI task 21 wraps it in a scheduled job. Each kind of AI question
resolves differently:
- **Numeric** (`question_key` = series id, AI task 9): the series' first live
  (not reconstructed) value for the target month. Live pulls only count
  ended months, and confirm backfilled months with a live row
  (`ai/monthly_ingest.py`), so every month a live pull has seen has one.
- **Threshold** (`'<series>>=<threshold>'`, AI task 10): yes if that same
  first live value is at least the threshold, else no. A question and its
  numeric twin resolve from the same row.
- **Curated** (`platform_question_id`, AI task 10): the Polymarket market's
  outcome, once it's yes or no. 50/50 and disputed outcomes stay open.
An answer counts only if it was known by `as_of` (its row's `as_of`, or the
market's resolution date), so resolving as of a past date is point-in-time.

Resolutions are stored once and not revised. Scores follow the question
type: numeric forecasts get `error` and `in_interval`, binary ones the binary
Brier score (p - outcome)^2 on their probability of yes.

Nothing here commits; the caller does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from psycopg2.extensions import connection as Connection

from eco_prediction.backtest import metrics
from eco_prediction.db.resolutions import (
    UnscoredForecast,
    save_resolution,
    save_score,
    unscored_forecasts,
)

THRESHOLD_SEPARATOR = ">="


@dataclass(frozen=True)
class AIOpenQuestion:
    id: int
    target: str
    question_type: str
    target_date: date
    question_key: str
    platform_question_id: int | None


@dataclass(frozen=True)
class AIAnswer:
    actual_as_of: date  # when the answer was known
    value: float | None = None  # numeric questions
    outcome: str | None = None  # 'yes' or 'no'


def unresolved_ai_questions(conn: Connection) -> list[AIOpenQuestion]:
    """AI questions with no resolution yet, oldest target first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.question_type::text, q.target_date,
                   q.question_key, q.platform_question_id
            FROM questions q
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE r.id IS NULL AND q.domain = 'ai'
            ORDER BY q.target_date, q.id
            """
        )
        return [AIOpenQuestion(*row) for row in cur.fetchall()]


def first_live_value(
    conn: Connection, series_id: str, month: date, as_of: date
) -> tuple[float, date] | None:
    """(value, as_of) of the series' first live row for `month` known by as_of."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.value, o.as_of
            FROM observations o JOIN series s ON s.id = o.series_id
            WHERE s.series_id = %s AND o.observed_at = %s AND o.as_of <= %s
              AND NOT o.reconstructed AND o.value IS NOT NULL
            ORDER BY o.as_of LIMIT 1
            """,
            (series_id, month, as_of),
        )
        row = cur.fetchone()
    return None if row is None else (float(row[0]), row[1])


def parse_threshold_key(key: str) -> tuple[str, float]:
    """'EPOCH_SWEBENCH_SOTA>=87.6' -> ('EPOCH_SWEBENCH_SOTA', 87.6)."""
    series_id, sep, threshold = key.partition(THRESHOLD_SEPARATOR)
    if not sep or not series_id:
        raise ValueError(f"not a threshold question key: {key!r}")
    return series_id, float(threshold)


def _market_outcome(
    conn: Connection, platform_question_id: int, as_of: date
) -> AIAnswer | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolved_outcome, resolved_at FROM platform_questions WHERE id = %s",
            (platform_question_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    resolved_on = row[1].date()
    return AIAnswer(resolved_on, outcome=row[0]) if resolved_on <= as_of else None


def answer(conn: Connection, q: AIOpenQuestion, as_of: date) -> AIAnswer | None:
    """The question's answer as known by as_of; None if it isn't known yet."""
    if q.platform_question_id is not None:
        return _market_outcome(conn, q.platform_question_id, as_of)
    if q.question_type == "binary":
        series_id, threshold = parse_threshold_key(q.question_key)
        found = first_live_value(conn, series_id, q.target_date, as_of)
        if found is None:
            return None
        value, known = found
        return AIAnswer(known, outcome="yes" if value >= threshold else "no")
    found = first_live_value(conn, q.question_key, q.target_date, as_of)
    return None if found is None else AIAnswer(found[1], value=found[0])


def score(forecast: UnscoredForecast) -> dict[str, float | bool | None]:
    """Score columns for one AI forecast against its question's resolution."""
    error, in_interval, brier = None, None, None
    if forecast.question_type == "binary":
        if forecast.probabilities is None or forecast.actual_outcome is None:
            raise ValueError(f"forecast {forecast.id}: binary but no probability")
        brier = metrics.brier_score_binary(
            float(forecast.probabilities["yes"]), forecast.actual_outcome
        )
    elif forecast.prediction is not None and forecast.actual_value is not None:
        error = forecast.prediction - forecast.actual_value
        if forecast.interval_lower is not None and forecast.interval_upper is not None:
            in_interval = (
                forecast.interval_lower
                <= forecast.actual_value
                <= forecast.interval_upper
            )
    return {"error": error, "in_interval": in_interval, "brier_score": brier}


@dataclass
class AIResolveResult:
    resolved: int = 0
    still_open: int = 0
    scored: int = 0


def resolve_and_score_ai(conn: Connection, as_of: date) -> AIResolveResult:
    """Resolve every AI question answered by as_of, then score unscored AI
    forecasts of resolved questions. No commit."""
    result = AIResolveResult()
    for q in unresolved_ai_questions(conn):
        found = answer(conn, q, as_of)
        if found is None:
            result.still_open += 1
            continue
        save_resolution(
            conn,
            q.id,
            actual_as_of=found.actual_as_of,
            value=found.value,
            outcome=found.outcome,
        )
        result.resolved += 1
    for forecast in unscored_forecasts(conn, domain="ai"):
        save_score(conn, forecast.id, **score(forecast))  # type: ignore[arg-type]
        result.scored += 1
    return result

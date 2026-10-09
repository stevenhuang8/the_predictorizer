"""Writing questions, model versions, feature snapshots and forecasts.

Usage:
    from eco_prediction.db.forecasts import get_explanation, save_forecast

    question_id, created = get_or_create_question(
        conn, "unemployment", date(2027, 1, 1), 3, resolution_rule="..."
    )
    fomc_week_ahead, _ = get_or_create_question(
        conn, "fomc_decision", date(2027, 1, 27), lead_days=7
    )
    model_version_id = get_or_create_model_version(
        conn, "lightgbm", "unemployment-h3-20260930", parameters={...}
    )
    snapshot_id = save_feature_snapshot(conn, date(2026, 9, 30), {"unrate": 4.3})

    forecast_id = save_forecast(
        conn, question_id, model_version_id, date(2026, 10, 1),
        model.predict(view, target_period),
        explanation=explainer.explain(view, target_period),
    )
    conn.commit()
    get_explanation(conn, forecast_id)          # Explanation(...) or None

    save_probability_forecast(conn, fomc_question_id, model_version_id,
                              date(2026, 10, 1), {"cut": .2, "hold": .7, "hike": .1})

Like the rest of `db`, these functions don't commit; the caller does.
JSON values (parameters, features) may hold NaN or numpy numbers; NaN is
stored as null.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

from eco_prediction.backtest.harness import Forecast
from eco_prediction.models.explainability import Explanation

QUESTION_TYPES = {
    "cpi_yoy": "numeric_interval",
    "unemployment": "numeric_interval",
    "fomc_decision": "probability",
}


def _json_value(value: Any) -> Any:
    """A JSON-safe copy: NaN to None, numpy scalars to Python numbers."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, date):
        return value.isoformat()
    return value


def get_or_create_question(
    conn: Connection,
    target: str,
    target_date: date,
    horizon_months: int | None = None,
    *,
    lead_days: int | None = None,
    resolution_rule: str | None = None,
) -> tuple[int, bool]:
    """The question's id, and whether it was created by this call.

    Give exactly one of `horizon_months` (asked in the month `horizon_months`
    before target_date's month) or `lead_days` (asked that many days before).
    """
    if (horizon_months is None) == (lead_days is None):
        raise ValueError("give exactly one of horizon_months and lead_days")
    key = (target, target_date, horizon_months, lead_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO questions (target, question_type, target_date,
                                   horizon_months, lead_days, resolution_rule)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT questions_target_date_horizon_key DO NOTHING
            RETURNING id
            """,
            (
                target,
                QUESTION_TYPES[target],
                target_date,
                horizon_months,
                lead_days,
                resolution_rule,
            ),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0]), True
        cur.execute(
            """
            SELECT id FROM questions
            WHERE target = %s AND target_date = %s
              AND horizon_months IS NOT DISTINCT FROM %s
              AND lead_days IS NOT DISTINCT FROM %s
            """,
            key,
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0]), False


def get_or_create_model_version(
    conn: Connection,
    model_type: str,
    version_tag: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    training_start: date | None = None,
    training_end: date | None = None,
    code_hash: str | None = None,
) -> int:
    """The id of (model_type, version_tag); created with these details if new."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_versions (model_type, version_tag, parameters,
                                        training_start, training_end, code_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_type, version_tag) DO NOTHING
            RETURNING id
            """,
            (
                model_type,
                version_tag,
                Json(_json_value(parameters)) if parameters is not None else None,
                training_start,
                training_end,
                code_hash,
            ),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT id FROM model_versions"
                " WHERE model_type = %s AND version_tag = %s",
                (model_type, version_tag),
            )
            row = cur.fetchone()
    assert row is not None
    return int(row[0])


def save_feature_snapshot(
    conn: Connection, snapshot_date: date, features: Mapping[str, Any]
) -> int:
    """Insert the features as known on snapshot_date and return the snapshot id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feature_snapshots (snapshot_date, features)
            VALUES (%s, %s) RETURNING id
            """,
            (snapshot_date, Json(_json_value(features))),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def forecast_exists(
    conn: Connection,
    question_id: int,
    model_type: str,
    version_tag: str,
    forecast_date: date,
    *,
    is_backtest: bool = False,
) -> bool:
    """Whether this model version already forecast this question on this date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM forecasts f
            JOIN model_versions m ON m.id = f.model_version_id
            WHERE f.question_id = %s AND m.model_type = %s AND m.version_tag = %s
              AND f.forecast_date = %s AND f.is_backtest = %s
            """,
            (question_id, model_type, version_tag, forecast_date, is_backtest),
        )
        return cur.fetchone() is not None


def save_forecast(
    conn: Connection,
    question_id: int,
    model_version_id: int,
    forecast_date: date,
    forecast: Forecast,
    *,
    explanation: Explanation | None = None,
    feature_snapshot_id: int | None = None,
    is_backtest: bool = False,
    rationale: str | None = None,
) -> int:
    """Insert one numeric forecast and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO forecasts (
                question_id, model_version_id, feature_snapshot_id, forecast_date,
                is_backtest, prediction, interval_lower, interval_upper,
                shap_values, rationale
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                question_id,
                model_version_id,
                feature_snapshot_id,
                forecast_date,
                is_backtest,
                forecast.point,
                forecast.lower,
                forecast.upper,
                Json(explanation.to_json()) if explanation else None,
                rationale,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def save_probability_forecast(
    conn: Connection,
    question_id: int,
    model_version_id: int,
    forecast_date: date,
    probabilities: Mapping[str, float],
    *,
    feature_snapshot_id: int | None = None,
    is_backtest: bool = False,
    rationale: str | None = None,
) -> int:
    """Insert one categorical forecast (e.g. FOMC cut/hold/hike) and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO forecasts (
                question_id, model_version_id, feature_snapshot_id, forecast_date,
                is_backtest, probabilities, rationale
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                question_id,
                model_version_id,
                feature_snapshot_id,
                forecast_date,
                is_backtest,
                Json(_json_value(probabilities)),
                rationale,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def get_explanation(conn: Connection, forecast_id: int) -> Explanation | None:
    """The stored explanation of a forecast; None if it has none.

    Raises KeyError if there is no forecast with that id.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT shap_values FROM forecasts WHERE id = %s", (forecast_id,))
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"no forecast with id {forecast_id}")
    return None if row[0] is None else Explanation.from_json(row[0])

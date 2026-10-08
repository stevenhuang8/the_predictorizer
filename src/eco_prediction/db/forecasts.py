"""Writing forecasts, with their SHAP explanations, to the forecasts table.

Usage:
    from eco_prediction.db.forecasts import get_explanation, save_forecast

    forecast_id = save_forecast(
        conn, question_id, model_version_id, date(2026, 10, 1),
        model.predict(view, target_period),
        explanation=explainer.explain(view, target_period),
    )
    conn.commit()
    get_explanation(conn, forecast_id)          # Explanation(...) or None

Like the rest of `db`, these functions don't commit; the caller does.
"""

from __future__ import annotations

from datetime import date

from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

from eco_prediction.backtest.harness import Forecast
from eco_prediction.models.explainability import Explanation


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

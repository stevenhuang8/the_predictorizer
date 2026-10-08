"""Tests for SHAP explanations and storing them with forecasts."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecast, add_months
from eco_prediction.db.forecasts import get_explanation, save_forecast
from eco_prediction.db.migrate import migrate
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.explainability import Explanation, SHAPExplainer
from eco_prediction.models.lightgbm_model import LightGBMModel

START = date(2005, 1, 1)
LEAD = 5  # MICH leads unemployment; see test_lightgbm_model.py
GRID: dict[str, list[Any]] = {
    "max_depth": [3],
    "learning_rate": [0.1],
    "min_child_samples": [10],
}
CUTOFF = date(2020, 3, 31)
TARGET = date(2020, 7, 1)  # 3 months after the April 1 forecast


def store() -> VintageStore:
    """MICH is white noise; unemployment is 5 + MICH from LEAD months earlier."""
    rng = np.random.default_rng(0)
    periods: list[date] = []
    period = START
    while period <= date(2023, 12, 1):
        periods.append(period)
        period = add_months(period, 1)
    mich = rng.uniform(-2, 2, len(periods))
    rows: dict[str, list[tuple[date, date, float]]] = {"MICH": [], "UNRATE": []}
    for i, period in enumerate(periods):
        published = add_months(period, 1).replace(day=10)
        rows["MICH"].append((period, published, float(mich[i])))
        if i >= LEAD:
            rows["UNRATE"].append((period, published, 5.0 + float(mich[i - LEAD])))
    return VintageStore(
        {
            series_id: pd.DataFrame(r, columns=["observed_at", "as_of", "value"])
            for series_id, r in rows.items()
        }
    )


STORE = store()


def view(cutoff: date = CUTOFF) -> PointInTimeData:
    return PointInTimeData(STORE, TARGETS["unemployment"], cutoff)


def fitted(**kwargs: Any) -> LightGBMModel:
    model = LightGBMModel(3, param_grid=GRID, **kwargs)
    model.fit(view())
    return model


@pytest.fixture(scope="module")
def change_model() -> LightGBMModel:
    return fitted()


@pytest.mark.parametrize("predict_change", [True, False])
def test_contributions_add_up_to_the_prediction(predict_change: bool) -> None:
    model = fitted(predict_change=predict_change)
    explanation = SHAPExplainer(model).explain(view(), TARGET)
    forecast = model.predict(view(), TARGET)
    assert explanation.prediction == pytest.approx(forecast.point, abs=1e-6)
    assert set(explanation.contributions) == set(model.feature_names)
    latest = view().target_history().iloc[-1]
    assert explanation.offset == (pytest.approx(latest) if predict_change else 0.0)


def test_top_features_are_the_signal(change_model: LightGBMModel) -> None:
    explanation = SHAPExplainer(change_model).explain(view(), TARGET)
    # The change to explain is 5 + MICH - UNRATE.
    assert {name for name, _ in explanation.top(2)} == {"mich", "unrate"}
    top = explanation.top(3)
    assert [abs(v) for _, v in top] == sorted((abs(v) for _, v in top), reverse=True)


def test_json_round_trip(change_model: LightGBMModel) -> None:
    explanation = SHAPExplainer(change_model).explain(view(), TARGET)
    assert Explanation.from_json(explanation.to_json()) == explanation


def test_needs_a_fitted_model() -> None:
    with pytest.raises(NotFittedError):
        SHAPExplainer(LightGBMModel(3))


def test_refuses_to_explain_a_refit_model() -> None:
    model = fitted()
    explainer = SHAPExplainer(model)
    model.fit(view(date(2021, 3, 31)))
    with pytest.raises(ValueError, match="refit"):
        explainer.explain(view(date(2021, 3, 31)), date(2021, 7, 1))


# Database (skipped if Postgres isn't reachable; fixtures in conftest.py).


@pytest.fixture
def ids(conn: Connection) -> tuple[int, int]:
    """A migrated database with one question and one model version."""
    migrate(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO questions (target, question_type, horizon_months, target_date)
            VALUES ('unemployment', 'numeric_interval', 3, %s) RETURNING id
            """,
            (TARGET,),
        )
        question = cur.fetchone()
        cur.execute(
            """
            INSERT INTO model_versions (model_type, version_tag)
            VALUES ('lightgbm', 'test') RETURNING id
            """
        )
        model = cur.fetchone()
    assert question is not None and model is not None
    return int(question[0]), int(model[0])


def test_stores_and_retrieves_explanation(
    conn: Connection, ids: tuple[int, int], change_model: LightGBMModel
) -> None:
    explanation = SHAPExplainer(change_model).explain(view(), TARGET)
    forecast = change_model.predict(view(), TARGET)
    forecast_id = save_forecast(
        conn,
        *ids,
        date(2020, 4, 1),
        forecast,
        explanation=explanation,
        is_backtest=True,
    )
    conn.commit()

    assert get_explanation(conn, forecast_id) == explanation
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT prediction::float8, interval_lower::float8, is_backtest,
                   (shap_values->'contributions'->>'mich')::float8
            FROM forecasts WHERE id = %s
            """,
            (forecast_id,),
        )
        assert cur.fetchone() == pytest.approx(
            (
                forecast.point,
                forecast.lower,
                True,
                explanation.contributions["mich"],
            )
        )


def test_forecast_without_explanation(conn: Connection, ids: tuple[int, int]) -> None:
    forecast_id = save_forecast(conn, *ids, date(2020, 4, 1), Forecast(4.2))
    assert get_explanation(conn, forecast_id) is None
    with pytest.raises(KeyError):
        get_explanation(conn, forecast_id + 1)

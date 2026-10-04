"""Tests for the model tracking schema (database fixtures in conftest.py)."""

from __future__ import annotations

from datetime import date
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json

from eco_prediction.db.migrate import migrate

FEATURES = {"yield_spread_10y2y": 0.5, "oil_price": 80.25, "cpi_lag1": None}
SHAP = {"yield_spread_10y2y": -0.12, "oil_price": 0.31}


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def insert(conn: Connection, table: str, **cols: Any) -> int:
    names = ", ".join(cols)
    marks = ", ".join(["%s"] * len(cols))
    values = [Json(v) if isinstance(v, dict) else v for v in cols.values()]
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} ({names}) VALUES ({marks}) RETURNING id", values
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def question(conn: Connection, target: str = "cpi_yoy") -> int:
    question_type = "probability" if target == "fomc_decision" else "numeric_interval"
    return insert(
        conn,
        "questions",
        target=target,
        question_type=question_type,
        horizon_months=3,
        target_date=date(2026, 12, 1),
    )


def model(conn: Connection, model_type: str = "lightgbm", tag: str = "v1") -> int:
    return insert(
        conn,
        "model_versions",
        model_type=model_type,
        version_tag=tag,
        parameters={"n_estimators": 200, "learning_rate": 0.05},
        training_start=date(1990, 1, 1),
        training_end=date(2026, 9, 30),
        code_hash="6da64b4",
    )


def forecast(conn: Connection, question_id: int, model_id: int, **cols: Any) -> int:
    defaults: dict[str, Any] = {"forecast_date": date(2026, 10, 1), "prediction": 2.9}
    return insert(
        conn,
        "forecasts",
        question_id=question_id,
        model_version_id=model_id,
        **{**defaults, **cols},
    )


def test_linked_forecast_round_trips_json(migrated: Connection) -> None:
    q, m = question(migrated), model(migrated)
    snapshot = insert(
        migrated,
        "feature_snapshots",
        snapshot_date=date(2026, 9, 30),
        features=FEATURES,
    )
    f = forecast(
        migrated,
        q,
        m,
        feature_snapshot_id=snapshot,
        interval_lower=2.4,
        interval_upper=3.4,
        shap_values=SHAP,
        rationale="Oil up; spread still inverted.",
    )
    migrated.commit()

    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT q.target, mv.model_type, mv.parameters, fs.features,
                   f.prediction::float8, f.shap_values, f.is_backtest
            FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            JOIN model_versions mv ON mv.id = f.model_version_id
            JOIN feature_snapshots fs ON fs.id = f.feature_snapshot_id
            WHERE f.id = %s
            """,
            (f,),
        )
        assert cur.fetchone() == (
            "cpi_yoy",
            "lightgbm",
            {"n_estimators": 200, "learning_rate": 0.05},
            FEATURES,
            2.9,
            SHAP,
            False,
        )
        # JSONB fields are queryable.
        cur.execute(
            "SELECT (features->>'oil_price')::float8 FROM feature_snapshots WHERE id = %s",
            (snapshot,),
        )
        assert cur.fetchone() == (80.25,)


def test_categorical_forecast_with_probabilities(migrated: Connection) -> None:
    probs = {"cut": 0.2, "hold": 0.7, "hike": 0.1}
    f = forecast(
        migrated,
        question(migrated, "fomc_decision"),
        model(migrated, "fomc_lgbm"),
        prediction=None,
        probabilities=probs,
    )
    with migrated.cursor() as cur:
        cur.execute("SELECT probabilities FROM forecasts WHERE id = %s", (f,))
        assert cur.fetchone() == (probs,)


def test_revision_chain_and_scores(migrated: Connection) -> None:
    q, m = question(migrated), model(migrated)
    first = forecast(migrated, q, m, prediction=2.9)
    revised = forecast(
        migrated,
        q,
        m,
        forecast_date=date(2026, 11, 1),
        prediction=3.1,
        previous_forecast_id=first,
    )
    with migrated.cursor() as cur:
        cur.execute(
            "UPDATE forecasts SET error = prediction - 3.0, scored_at = NOW() WHERE question_id = %s",
            (q,),
        )
        cur.execute(
            """
            SELECT r.prediction::float8, p.prediction::float8, r.error::float8
            FROM forecasts r JOIN forecasts p ON p.id = r.previous_forecast_id
            WHERE r.id = %s
            """,
            (revised,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[:2] == (3.1, 2.9)
    assert row[2] == pytest.approx(0.1)


def test_model_versions_are_unique_per_type(migrated: Connection) -> None:
    model(migrated, "lightgbm", "v1")
    model(migrated, "arima", "v1")
    with pytest.raises(psycopg2.errors.UniqueViolation):
        model(migrated, "lightgbm", "v1")


@pytest.mark.parametrize(
    ("cols", "error"),
    [
        ({"question_id": 999}, psycopg2.errors.ForeignKeyViolation),
        ({"model_version_id": 999}, psycopg2.errors.ForeignKeyViolation),
        ({"feature_snapshot_id": 999}, psycopg2.errors.ForeignKeyViolation),
        ({"previous_forecast_id": 999}, psycopg2.errors.ForeignKeyViolation),
        ({"forecast_date": None}, psycopg2.errors.NotNullViolation),
        ({"prediction": None}, psycopg2.errors.CheckViolation),  # nothing predicted
        ({"interval_lower": 2.0}, psycopg2.errors.CheckViolation),  # half an interval
        (
            {"interval_lower": 3.5, "interval_upper": 2.5},
            psycopg2.errors.CheckViolation,
        ),
        ({"probabilities": Json([0.2, 0.8])}, psycopg2.errors.CheckViolation),
        ({"shap_values": Json(1.5)}, psycopg2.errors.CheckViolation),
    ],
)
def test_invalid_forecasts_are_rejected(
    migrated: Connection, cols: dict[str, Any], error: type[Exception]
) -> None:
    base = {"question_id": question(migrated), "model_id": model(migrated)}
    q = cols.pop("question_id", base["question_id"])
    m = cols.pop("model_version_id", base["model_id"])
    with pytest.raises(error):
        forecast(migrated, q, m, **cols)


def test_invalid_model_and_snapshot_rows_are_rejected(migrated: Connection) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation):
        insert(
            migrated,
            "model_versions",
            model_type="arima",
            version_tag="bad",
            training_start=date(2020, 1, 1),
            training_end=date(2019, 1, 1),
        )
    migrated.rollback()
    with pytest.raises(psycopg2.errors.CheckViolation):
        insert(
            migrated,
            "feature_snapshots",
            snapshot_date=date(2026, 9, 30),
            features=Json([1, 2]),
        )


def test_deletes_cascade_from_questions_and_unlink_revisions(
    migrated: Connection,
) -> None:
    q, m = question(migrated), model(migrated)
    other_q = question(migrated, "unemployment")
    first = forecast(migrated, other_q, m)
    revised = forecast(migrated, q, m, previous_forecast_id=first)

    with migrated.cursor() as cur:
        cur.execute("DELETE FROM questions WHERE id = %s", (other_q,))
        cur.execute(
            "SELECT previous_forecast_id FROM forecasts WHERE id = %s", (revised,)
        )
        assert cur.fetchone() == (None,)
        cur.execute("SELECT count(*) FROM forecasts")
        assert cur.fetchone() == (1,)
        # A model version with forecasts can't be deleted.
        with pytest.raises(psycopg2.errors.ForeignKeyViolation):
            cur.execute("DELETE FROM model_versions WHERE id = %s", (m,))

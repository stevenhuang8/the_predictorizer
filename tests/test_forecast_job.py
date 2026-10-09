"""Tests for the monthly forecast job.

Database tests run on a throwaway, migrated database (conftest.py) and skip if
Postgres isn't reachable. The store is synthetic: unemployment led by MICH (as
in test_lightgbm_model.py) and a fed funds target with random decisions that
the 3-month bill anticipates (as in test_fomc.py). Models use small grids.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecast, add_months
from eco_prediction.data.fomc import OUTCOMES, meetings_between
from eco_prediction.db.forecasts import get_explanation, get_or_create_question
from eco_prediction.db.migrate import migrate
from eco_prediction.features.engineering import FeatureEngineer
from eco_prediction.models.baselines import RandomWalkModel
from eco_prediction.models.fomc_model import FOMCForecaster, Persistence
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.scheduler.forecast_job import (
    FOMCModelSpec,
    NumericModelSpec,
    Question,
    month_forecast_date,
    plan_questions,
    pre_meeting_due,
    run_forecast_job,
    run_pre_meeting_job,
    try_lock,
    unlock,
)

FORECAST_DATE = date(2023, 8, 1)  # FOMC meetings in Sep and Nov 2023, not Feb 2024
LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [3],
    "learning_rate": [0.1],
    "min_child_samples": [10],
}
FOMC_GRID: dict[str, list[Any]] = {
    "max_depth": [2],
    "min_data_in_leaf": [5],
    "learning_rate": [0.1],
}
CODE = {"cut": -1, "hold": 0, "hike": 1}


def _frame(rows: list[tuple[date, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def synthetic_store(last: date = date(2023, 12, 31)) -> VintageStore:
    rng = np.random.default_rng(22)
    months = pd.date_range("2005-01-01", last, freq="MS")
    mich = rng.uniform(-2, 2, len(months))
    monthly: dict[str, list[tuple[date, date, float]]] = {"MICH": [], "UNRATE": []}
    for i, month in enumerate(months):
        period = month.date()
        published = add_months(period, 1).replace(day=10)
        monthly["MICH"].append((period, published, float(mich[i])))
        if i >= 5:
            monthly["UNRATE"].append((period, published, 5.0 + float(mich[i - 5])))

    meetings = meetings_between(date(2009, 1, 1), last)
    decided = {m: str(rng.choice(OUTCOMES, p=[0.2, 0.55, 0.25])) for m in meetings}
    days = pd.date_range("2008-12-16", last, freq="D")
    upper = pd.Series(2.0, index=days)
    for m, outcome in decided.items():
        upper[upper.index > pd.Timestamp(m)] += 0.25 * CODE[outcome]
    upcoming = np.array(
        [
            CODE[decided[next(m for m in meetings if m >= d.date())]]
            if d.date() <= meetings[-1]
            else 0
            for d in days
        ]
    )
    tbill = upper.to_numpy() - 0.125 + 0.2 * upcoming
    tbill += rng.normal(0, 0.03, len(days))

    def daily(values: np.ndarray) -> pd.DataFrame:
        return _frame(
            [
                (d.date(), (d + timedelta(days=1)).date(), float(v))
                for d, v in zip(days, values)
            ]
        )

    return VintageStore(
        {
            **{k: _frame(v) for k, v in monthly.items()},
            "DFEDTARU": daily(upper.to_numpy()),
            "DTB3": daily(tbill),
        }
    )


STORE = synthetic_store()


class Counting:
    """Counts how often specs build a model, to prove reruns don't refit."""

    def __init__(self) -> None:
        self.builds: Counter[str] = Counter()
        self.features = FeatureEngineer()  # shared, as the job's defaults do

    def numeric(self) -> tuple[NumericModelSpec, ...]:
        def random_walk(h: int) -> RandomWalkModel:
            self.builds["random_walk"] += 1
            return RandomWalkModel()

        def lightgbm(h: int) -> LightGBMModel:
            self.builds["lightgbm"] += 1
            return LightGBMModel(h, param_grid=LGBM_GRID, features=self.features)

        return (
            NumericModelSpec("random_walk", random_walk),
            NumericModelSpec("lightgbm", lightgbm, explain=True),
        )

    def fomc(self) -> tuple[FOMCModelSpec, ...]:
        def persistence(lead: int) -> Persistence:
            self.builds["fomc_persistence"] += 1
            return Persistence(lead, train_start=date(2009, 1, 1))

        def lightgbm(lead: int) -> FOMCForecaster:
            self.builds["fomc_lightgbm"] += 1
            return FOMCForecaster(
                lead,
                train_start=date(2009, 1, 1),
                param_grid=FOMC_GRID,
                features=self.features,
            )

        return (
            FOMCModelSpec("fomc_persistence", persistence),
            FOMCModelSpec("fomc_lightgbm", lightgbm),
        )


def run(
    conn: Connection, counting: Counting, when: date = FORECAST_DATE, **kwargs: Any
) -> Any:
    return run_forecast_job(
        conn,
        STORE,
        when,
        numeric_models=counting.numeric(),
        fomc_models=counting.fomc(),
        targets=("unemployment",),
        code_hash="abc1234",
        **kwargs,
    )


def rows(conn: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@pytest.fixture
def db(conn: Connection) -> Connection:
    migrate(conn)
    return conn


# Planning (no database)


def test_plan_questions_for_october_2026() -> None:
    questions = plan_questions(date(2026, 10, 1))
    numeric = [q for q in questions if q.target != "fomc_decision"]
    assert {(q.target, q.target_date, q.horizon_months) for q in numeric} == {
        (t, d, h)
        for t in ("cpi_yoy", "unemployment")
        for d, h in (
            (date(2026, 11, 1), 1),
            (date(2027, 1, 1), 3),
            (date(2027, 4, 1), 6),
        )
    }
    # No meeting in November 2026; January 27 and April 28, 2027.
    assert [q for q in questions if q.target == "fomc_decision"] == [
        Question("fomc_decision", date(2027, 1, 27), 3),
        Question("fomc_decision", date(2027, 4, 28), 6),
    ]


def test_plan_questions_mid_month_uses_the_calendar_month() -> None:
    questions = plan_questions(date(2026, 10, 17), include_fomc=False)
    assert {q.target_date for q in questions} == {
        date(2026, 11, 1),
        date(2027, 1, 1),
        date(2027, 4, 1),
    }


# The job (database)


def test_job_stores_questions_forecasts_and_snapshots(db: Connection) -> None:
    result = run(db, Counting(), is_backtest=False)
    assert result.failures == []
    assert result.questions_created == 5  # 3 unemployment + Sep and Nov meetings
    assert result.forecasts_inserted == 3 * 2 + 2 * 2

    questions = rows(
        db,
        "SELECT target, question_type, target_date, horizon_months, resolution_rule"
        " FROM questions ORDER BY target, target_date",
    )
    assert [(q[0], q[1], q[2], q[3]) for q in questions] == [
        ("unemployment", "numeric_interval", date(2023, 9, 1), 1),
        ("unemployment", "numeric_interval", date(2023, 11, 1), 3),
        ("unemployment", "numeric_interval", date(2024, 2, 1), 6),
        ("fomc_decision", "probability", date(2023, 9, 20), 1),
        ("fomc_decision", "probability", date(2023, 11, 1), 3),
    ]
    assert all(q[4] for q in questions)

    forecasts = rows(
        db,
        """
        SELECT m.model_type, m.version_tag, m.training_end, m.code_hash, m.parameters,
               f.forecast_date, f.is_backtest, f.prediction, f.interval_lower,
               f.probabilities, f.shap_values IS NOT NULL, s.snapshot_date, s.features
        FROM forecasts f
        JOIN model_versions m ON m.id = f.model_version_id
        LEFT JOIN feature_snapshots s ON s.id = f.feature_snapshot_id
        """,
    )
    by_type = Counter(f[0] for f in forecasts)
    assert by_type == {
        "random_walk": 3,
        "lightgbm": 3,
        "fomc_persistence": 2,
        "fomc_lightgbm": 2,
    }
    cutoff = date(2023, 7, 31)
    for f in forecasts:
        model_type, tag, training_end, code_hash, params = f[:5]
        forecast_date, is_backtest, prediction, lower, probs, has_shap = f[5:11]
        snapshot_date, features = f[11:]
        assert forecast_date == FORECAST_DATE and is_backtest is False
        assert training_end == cutoff and code_hash == "abc1234"
        assert tag.endswith("-20230731")
        if model_type.startswith("fomc"):
            assert prediction is None and set(probs) == set(OUTCOMES)
            assert sum(probs.values()) == pytest.approx(1)
            assert "lead_days" in params
        else:
            assert 0 < prediction < 10 and lower is not None
            assert params["target"] == "unemployment"
        explained = model_type in ("lightgbm", "fomc_lightgbm")
        assert has_shap == (model_type == "lightgbm")
        assert (snapshot_date == cutoff) if explained else (snapshot_date is None)
        if explained:
            assert features and all(isinstance(k, str) for k in features)
    # Leads are measured from the forecast date to each meeting.
    tags = {f[1] for f in forecasts if f[0] == "fomc_lightgbm"}
    assert tags == {"fomc-lead50-20230731", "fomc-lead92-20230731"}


def test_lightgbm_forecast_matches_its_stored_explanation(db: Connection) -> None:
    run(db, Counting(), is_backtest=False)
    forecast_id, prediction = rows(
        db,
        """
        SELECT f.id, f.prediction FROM forecasts f
        JOIN model_versions m ON m.id = f.model_version_id
        JOIN questions q ON q.id = f.question_id
        WHERE m.model_type = 'lightgbm' AND q.horizon_months = 3
        """,
    )[0]
    explanation = get_explanation(db, forecast_id)
    assert explanation is not None
    assert explanation.prediction == pytest.approx(float(prediction))


def test_rerun_on_the_same_day_adds_nothing_and_fits_nothing(db: Connection) -> None:
    run(db, Counting(), is_backtest=False)
    counting = Counting()
    result = run(db, counting, is_backtest=False)
    assert result.questions_created == 0
    assert result.forecasts_inserted == 0
    assert result.forecasts_skipped == 10
    assert counting.builds == {}
    assert rows(db, "SELECT count(*) FROM forecasts")[0][0] == 10
    assert rows(db, "SELECT count(*) FROM feature_snapshots")[0][0] == 5


def test_next_month_adds_new_questions_without_duplicates(db: Connection) -> None:
    run(db, Counting(), is_backtest=False)
    result = run(db, Counting(), date(2023, 9, 1), is_backtest=False)
    # Oct, Dec 2023 and Mar 2024; FOMC on Dec 13 and Mar 20 (none in October).
    assert result.questions_created == 5
    assert result.forecasts_inserted == 3 * 2 + 2 * 2
    duplicates = rows(
        db,
        """
        SELECT question_id, model_version_id, forecast_date, count(*)
        FROM forecasts GROUP BY 1, 2, 3 HAVING count(*) > 1
        """,
    )
    assert duplicates == []


def test_a_failing_model_does_not_lose_the_others(db: Connection) -> None:
    class Broken:
        def fit(self, data: PointInTimeData) -> None:
            raise RuntimeError("boom")

        def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
            raise AssertionError

    counting = Counting()
    specs = (*counting.numeric(), NumericModelSpec("broken", lambda h: Broken()))
    result = run_forecast_job(
        db,
        STORE,
        FORECAST_DATE,
        is_backtest=False,
        numeric_models=specs,
        include_fomc=False,
        targets=("unemployment",),
    )
    assert len(result.failures) == 3 and "boom" in result.failures[0]
    assert result.forecasts_inserted == 6
    # Nothing half-written for the broken model.
    assert rows(
        db, "SELECT count(*) FROM model_versions WHERE model_type = 'broken'"
    ) == [(0,)]

    # A rerun with the model fixed fills in only its three forecasts.
    fixed = (
        *counting.numeric(),
        NumericModelSpec("broken", lambda h: RandomWalkModel()),
    )
    again = run_forecast_job(
        db,
        STORE,
        FORECAST_DATE,
        is_backtest=False,
        numeric_models=fixed,
        include_fomc=False,
        targets=("unemployment",),
    )
    assert (again.forecasts_inserted, again.forecasts_skipped) == (3, 6)


def test_past_dates_default_to_backtest_and_do_not_collide(db: Connection) -> None:
    live = run(db, Counting(), is_backtest=False, include_fomc=False)
    backtest = run(db, Counting(), include_fomc=False)  # 2023 is before today
    assert backtest.is_backtest is True
    assert backtest.forecasts_inserted == live.forecasts_inserted == 6
    assert rows(
        db, "SELECT is_backtest, count(*) FROM forecasts GROUP BY 1 ORDER BY 1"
    ) == [(False, 6), (True, 6)]


def test_unique_index_rejects_a_duplicate_forecast(db: Connection) -> None:
    run(db, Counting(), is_backtest=False, include_fomc=False)
    with pytest.raises(psycopg2.errors.UniqueViolation), db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO forecasts (question_id, model_version_id, forecast_date,
                                   prediction)
            SELECT question_id, model_version_id, forecast_date, 1.0
            FROM forecasts LIMIT 1
            """
        )
    db.rollback()


def test_month_forecast_date_finds_this_months_live_run(db: Connection) -> None:
    assert month_forecast_date(db, date(2023, 8, 20)) is None
    run(db, Counting(), is_backtest=False, include_fomc=False)
    assert month_forecast_date(db, date(2023, 8, 20)) == FORECAST_DATE
    assert month_forecast_date(db, date(2023, 9, 2)) is None
    # Backtests don't count as this month's live run.
    run(db, Counting(), date(2023, 10, 1), is_backtest=True, include_fomc=False)
    assert month_forecast_date(db, date(2023, 10, 5)) is None


def test_advisory_lock_blocks_a_second_run(db: Connection, test_dsn: str) -> None:
    other = psycopg2.connect(test_dsn)
    try:
        assert try_lock(db)
        assert not try_lock(other)
        unlock(db)
        assert try_lock(other)
        unlock(other)
    finally:
        other.close()


# Pre-meeting FOMC questions (lead_days, migration 006)


def test_pre_meeting_due_from_a_week_before_until_the_meeting() -> None:
    meeting = date(2026, 10, 28)
    assert pre_meeting_due(date(2026, 10, 20)) == []
    assert pre_meeting_due(date(2026, 10, 21)) == [meeting]  # due today
    assert pre_meeting_due(date(2026, 10, 27)) == [meeting]  # catching up
    assert pre_meeting_due(date(2026, 10, 28)) == []  # decided today: too late


def test_pre_meeting_run_is_its_own_question(db: Connection) -> None:
    meeting = date(2023, 9, 20)
    monthly = run(db, Counting(), is_backtest=False)  # asks about it at h=1
    counting = Counting()
    result = run_pre_meeting_job(
        db, STORE, meeting, is_backtest=False, fomc_models=counting.fomc()
    )
    assert monthly.failures == result.failures == []
    assert (result.questions_created, result.forecasts_inserted) == (1, 2)
    assert result.forecast_date == date(2023, 9, 13)

    assert rows(
        db,
        """
        SELECT horizon_months, lead_days, count(f.*), min(f.forecast_date),
               min(m.version_tag), min(m.training_end)
        FROM questions q
        JOIN forecasts f ON f.question_id = q.id
        JOIN model_versions m ON m.id = f.model_version_id
        WHERE q.target = 'fomc_decision' AND q.target_date = %s
        GROUP BY 1, 2 ORDER BY 2 NULLS FIRST
        """,
        meeting,
    ) == [
        (1, None, 2, FORECAST_DATE, "fomc-lead50-20230731", date(2023, 7, 31)),
        (None, 7, 2, date(2023, 9, 13), "fomc-lead7-20230912", date(2023, 9, 12)),
    ]

    again = run_pre_meeting_job(
        db, STORE, meeting, is_backtest=False, fomc_models=Counting().fomc()
    )
    assert (again.forecasts_inserted, again.forecasts_skipped) == (0, 2)
    # Pre-meeting forecasts aren't the month's monthly run.
    assert month_forecast_date(db, date(2023, 9, 30)) is None


def test_pre_meeting_run_defaults_to_backtest_for_past_meetings(
    db: Connection,
) -> None:
    result = run_pre_meeting_job(
        db, STORE, date(2023, 11, 1), fomc_models=Counting().fomc()
    )
    assert result.is_backtest is True


def test_a_question_has_exactly_one_horizon(db: Connection) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        get_or_create_question(db, "fomc_decision", date(2027, 1, 27), 3, lead_days=7)
    with pytest.raises(ValueError, match="exactly one"):
        get_or_create_question(db, "fomc_decision", date(2027, 1, 27))
    for horizon, lead in ((3, 7), (None, None)):
        with pytest.raises(psycopg2.errors.CheckViolation), db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO questions (target, question_type, target_date,
                                       horizon_months, lead_days)
                VALUES ('fomc_decision', 'probability', '2027-01-27', %s, %s)
                """,
                (horizon, lead),
            )
        db.rollback()


def test_lead_questions_are_unique_and_reused(db: Connection) -> None:
    meeting = date(2027, 1, 27)
    first, created = get_or_create_question(db, "fomc_decision", meeting, lead_days=7)
    again, created_again = get_or_create_question(
        db, "fomc_decision", meeting, lead_days=7
    )
    monthly, _ = get_or_create_question(db, "fomc_decision", meeting, 3)
    assert created and not created_again and first == again != monthly
    with pytest.raises(psycopg2.errors.UniqueViolation), db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO questions (target, question_type, target_date, lead_days)
            VALUES ('fomc_decision', 'probability', %s, 7)
            """,
            (meeting,),
        )
    db.rollback()

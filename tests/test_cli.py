"""Tests for the eco-forecast CLI.

Forecast commands run on the synthetic stores of test_lightgbm_model.py
(unemployment led by MICH) and test_fomc.py (a T-bill that prices each
decision), with small model grids. Inspect runs on a migrated throwaway
database.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection
from test_fomc import STORE as FOMC_STORE
from test_lightgbm_model import STORE as UNEMPLOYMENT_STORE

from eco_prediction import cli
from eco_prediction.backtest.harness import Forecast
from eco_prediction.db.forecasts import (
    get_or_create_model_version,
    get_or_create_question,
    save_forecast,
)
from eco_prediction.db.migrate import migrate
from eco_prediction.db.postmortems import save_postmortem
from eco_prediction.db.resolutions import save_resolution, save_score
from eco_prediction.models.baselines import RandomWalkModel
from eco_prediction.models.fomc_model import FOMCForecaster, Persistence
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.scheduler.forecast_job import FOMCModelSpec, NumericModelSpec

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
NUMERIC = (
    NumericModelSpec("random_walk", lambda h: RandomWalkModel()),
    NumericModelSpec("lightgbm", lambda h: LightGBMModel(h, param_grid=LGBM_GRID)),
)
FOMC = (
    FOMCModelSpec("fomc_persistence", lambda lead: Persistence(lead)),
    FOMCModelSpec(
        "fomc_lightgbm",
        lambda lead: FOMCForecaster(
            lead, train_start=date(2009, 1, 1), param_grid=FOMC_GRID
        ),
    ),
)


# Parsing


def test_parser() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["forecast", "--target", "cpi"])
    assert (args.target, args.horizon, args.model) == ("cpi", 3, None)
    args = parser.parse_args(["inspect", "--question-id", "4"])
    assert args.question_id == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--target", "gdp"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast"])  # --target is required
    with pytest.raises(SystemExit):
        cli.main(["inspect", "--refresh"])  # only backtest passes options through


def test_backtest_passes_its_arguments_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    monkeypatch.setattr("eco_prediction.scripts.model_comparison.main", fake)
    assert cli.main(["backtest", "--refresh", "--start", "2018-01-01"]) == 0
    assert seen == [["--refresh", "--start", "2018-01-01"]]


def test_horizon_must_be_positive() -> None:
    with pytest.raises(SystemExit, match="at least 1"):
        cli.main(["forecast", "--target", "cpi", "--horizon", "0"])


# Forecasting (synthetic stores, nothing stored)


def test_forecast_numeric_shows_every_model_and_drivers() -> None:
    out = cli.forecast_numeric(
        UNEMPLOYMENT_STORE, "unemployment", 3, date(2020, 6, 1), specs=NUMERIC
    )
    assert "September 2020, 3 month(s) ahead" in out
    assert "Data through 2020-05-31" in out
    assert "random_walk" in out and "lightgbm" in out
    # LightGBM's top SHAP driver is the leading indicator or the target's lag.
    assert "mich" in out or "unrate" in out


def test_forecast_numeric_one_model_or_an_error() -> None:
    out = cli.forecast_numeric(
        UNEMPLOYMENT_STORE,
        "unemployment",
        1,
        date(2020, 6, 1),
        model="random_walk",
        specs=NUMERIC,
    )
    assert "random_walk" in out and "lightgbm" not in out
    with pytest.raises(SystemExit, match="unknown model"):
        cli.forecast_numeric(
            UNEMPLOYMENT_STORE,
            "unemployment",
            1,
            date(2020, 6, 1),
            model="x",
            specs=NUMERIC,
        )


def test_forecast_fomc_next_meeting_by_default() -> None:
    out = cli.forecast_fomc(FOMC_STORE, date(2019, 7, 20), specs=FOMC)
    assert "FOMC decision on 2019-07-31, 11 days ahead" in out
    assert "fomc_persistence" in out and "fomc_lightgbm" in out
    assert "cut" in out and "hold" in out and "hike" in out


@pytest.mark.parametrize(
    ("meeting", "message"),
    [
        (date(2019, 7, 30), "not a scheduled FOMC decision day"),
        (date(2019, 6, 19), "already happened"),
    ],
)
def test_forecast_fomc_rejects_bad_meetings(meeting: date, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        cli.forecast_fomc(FOMC_STORE, date(2019, 7, 20), meeting=meeting, specs=FOMC)


# Inspecting (database)


@pytest.fixture
def db(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def test_list_and_inspect_questions(db: Connection) -> None:
    assert cli.list_questions(db) == "No questions yet."
    question, _ = get_or_create_question(
        db, "unemployment", date(2023, 3, 1), 3, resolution_rule="First release."
    )
    version = get_or_create_model_version(db, "lightgbm", "test")
    forecast = save_forecast(
        db, question, version, date(2022, 12, 1), Forecast(3.2, 3.1, 3.3)
    )
    save_forecast(
        db,
        question,
        version,
        date(2022, 12, 1),
        Forecast(3.5, 3.0, 4.0),
        is_backtest=True,
    )
    db.commit()

    listing = cli.list_questions(db)
    assert "unemployment" in listing and "3 months" in listing
    detail = cli.inspect_question(db, question)
    assert "Status: open" in detail and "3.20 [3.10, 3.30]" in detail
    assert "backtest" in detail and "live" in detail

    save_resolution(db, question, actual_as_of=date(2023, 4, 7), value=3.6)
    save_score(db, forecast, error=-0.4, in_interval=False)
    save_postmortem(db, forecast, "bad_model", notes="Missed an ordinary move.")
    db.commit()
    detail = cli.inspect_question(db, question)
    assert "resolved: 3.60 (published 2023-04-07)" in detail
    assert "error -0.40 (outside)" in detail and "bad_model" in detail
    assert "Missed an ordinary move." in detail
    with pytest.raises(SystemExit, match="no question"):
        cli.inspect_question(db, 999)

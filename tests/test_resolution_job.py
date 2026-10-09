"""Tests for resolving questions and scoring forecasts.

The store is built by hand so every resolved value can be checked:
- UNRATE for Jan-Jun 2023, each month published on the 10th of the next;
  March (first released as 3.6) is revised to 3.5 on 2023-05-10.
- CPIAUCSL for Mar 2022 (100) and Mar 2023 (105): 5.0% YoY as first
  released; Mar 2022 is revised to 101 on 2023-05-10, which must not matter.
- DFEDTARU daily through 2023-06-30, published the next day: a hike at the
  2023-02-01 meeting (4.50 -> 4.75), holds after.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import VintageStore
from eco_prediction.backtest.harness import Forecast
from eco_prediction.db.forecasts import (
    get_or_create_model_version,
    get_or_create_question,
    save_forecast,
    save_probability_forecast,
)
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import UnscoredForecast, scored_forecasts
from eco_prediction.scheduler.resolution_job import (
    resolve_and_score,
    score,
    score_summary,
)

FOMC_HIKE = date(2023, 2, 1)
FOMC_JUNE = date(2023, 6, 14)


def _frame(rows: list[tuple[date, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def build_store() -> VintageStore:
    unrate = [
        (date(2023, m, 1), date(2023, m + 1, 10), v)
        for m, v in zip(range(1, 7), [3.4, 3.6, 3.6, 3.4, 3.7, 3.6])
    ]
    unrate.append((date(2023, 3, 1), date(2023, 5, 10), 3.5))  # revision
    cpi = [
        (date(2022, 3, 1), date(2022, 4, 10), 100.0),
        (date(2023, 3, 1), date(2023, 4, 10), 105.0),
        (date(2022, 3, 1), date(2023, 5, 10), 101.0),  # revision
    ]
    days = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    upper = [4.5 if d.date() <= FOMC_HIKE else 4.75 for d in days]
    fed = [(d.date(), (d + timedelta(days=1)).date(), v) for d, v in zip(days, upper)]
    return VintageStore(
        {"UNRATE": _frame(unrate), "CPIAUCSL": _frame(cpi), "DFEDTARU": _frame(fed)}
    )


STORE = build_store()


@pytest.fixture
def db(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def model(conn: Connection, model_type: str = "random_walk") -> int:
    return get_or_create_model_version(conn, model_type, "test")


def numeric_forecast(
    conn: Connection,
    target: str,
    period: date,
    point: float,
    lower: float | None,
    upper: float | None,
    *,
    model_type: str = "random_walk",
    is_backtest: bool = False,
) -> int:
    question, _ = get_or_create_question(conn, target, period, 1)
    return save_forecast(
        conn,
        question,
        model(conn, model_type),
        period - timedelta(days=40),
        Forecast(point, lower, upper),
        is_backtest=is_backtest,
    )


def fomc_forecast(
    conn: Connection,
    meeting: date,
    probs: dict[str, float],
    *,
    horizon_months: int | None = 1,
    lead_days: int | None = None,
) -> int:
    question, _ = get_or_create_question(
        conn, "fomc_decision", meeting, horizon_months, lead_days=lead_days
    )
    return save_probability_forecast(
        conn,
        question,
        model(conn, "fomc_lightgbm"),
        meeting - timedelta(days=lead_days or 30),
        probs,
    )


def rows(conn: Connection, sql: str, *params: object) -> list[tuple[object, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def scores(conn: Connection, forecast_id: int) -> tuple[object, ...]:
    return rows(
        conn,
        "SELECT error, in_interval, brier_score, scored_at IS NOT NULL"
        " FROM forecasts WHERE id = %s",
        forecast_id,
    )[0]


# Scoring one forecast (no database)


def test_score_numeric_and_probability() -> None:
    numeric = UnscoredForecast(1, 3.8, 3.4, 4.0, None, 3.6, None)
    assert score(numeric) == pytest.approx(
        {"error": 0.2, "in_interval": True, "brier_score": None}
    )
    no_interval = UnscoredForecast(2, 3.0, None, None, None, 3.6, None)
    assert score(no_interval)["in_interval"] is None
    fomc = UnscoredForecast(
        3, None, None, None, {"cut": 0.1, "hold": 0.3, "hike": 0.6}, None, "hike"
    )
    assert score(fomc) == pytest.approx(
        {"error": None, "in_interval": None, "brier_score": 0.01 + 0.09 + 0.16}
    )


# Resolving and scoring (database)


def test_numeric_question_resolves_with_the_first_release(db: Connection) -> None:
    good = numeric_forecast(db, "unemployment", date(2023, 3, 1), 3.8, 3.4, 4.0)
    bad = numeric_forecast(
        db, "unemployment", date(2023, 3, 1), 3.0, 2.8, 3.2, model_type="arima"
    )
    db.commit()
    result = resolve_and_score(db, STORE, date(2023, 6, 30))
    assert len(result.resolved) == 1 and result.scored == 2

    ((value, outcome, as_of),) = rows(
        db, "SELECT actual_value, actual_outcome, actual_as_of FROM resolutions"
    )
    # The first release, not the 3.5 revision.
    assert float(value) == pytest.approx(3.6)  # type: ignore[arg-type]
    assert outcome is None and as_of == date(2023, 4, 10)
    error, in_interval, brier, scored = scores(db, good)
    assert float(error) == pytest.approx(0.2) and in_interval is True  # type: ignore[arg-type]
    assert brier is None and scored is True
    error, in_interval, _, _ = scores(db, bad)
    assert float(error) == pytest.approx(-0.6) and in_interval is False  # type: ignore[arg-type]


def test_cpi_yoy_uses_the_year_earlier_value_as_known_then(db: Connection) -> None:
    numeric_forecast(db, "cpi_yoy", date(2023, 3, 1), 4.0, 3.0, 5.0)
    db.commit()
    resolve_and_score(db, STORE, date(2023, 6, 30))
    value, as_of = rows(db, "SELECT actual_value, actual_as_of FROM resolutions")[0]
    # 105 / 100 (not the later 101): 5.0%, published 2023-04-10.
    assert float(value) == pytest.approx(5.0) and as_of == date(2023, 4, 10)  # type: ignore[arg-type]


def test_unpublished_answers_leave_questions_open(db: Connection) -> None:
    june = numeric_forecast(db, "unemployment", date(2023, 6, 1), 3.6, 3.2, 4.0)
    march = numeric_forecast(db, "unemployment", date(2023, 3, 1), 3.6, 3.2, 4.0)
    db.commit()
    # June is published 2023-07-10; March on 2023-04-10, after this run date.
    result = resolve_and_score(db, STORE, date(2023, 4, 9))
    assert result.resolved == [] and result.still_open == 2 and result.scored == 0
    assert scores(db, june)[3] is False and scores(db, march)[3] is False

    result = resolve_and_score(db, STORE, date(2023, 4, 10))
    assert len(result.resolved) == 1 and result.still_open == 1
    assert scores(db, march)[3] is True and scores(db, june)[3] is False


def test_fomc_questions_resolve_to_the_decision(db: Connection) -> None:
    probs = {"cut": 0.1, "hold": 0.3, "hike": 0.6}
    monthly = fomc_forecast(db, FOMC_HIKE, probs)
    week_ahead = fomc_forecast(db, FOMC_HIKE, probs, horizon_months=None, lead_days=7)
    june = fomc_forecast(db, FOMC_JUNE, probs)
    db.commit()

    resolve_and_score(db, STORE, FOMC_JUNE)  # June's decision isn't out yet
    resolved = rows(
        db,
        """
        SELECT q.target_date, q.lead_days, r.actual_outcome, r.actual_as_of
        FROM resolutions r JOIN questions q ON q.id = r.question_id
        ORDER BY q.lead_days NULLS FIRST
        """,
    )
    # Published when the day after the meeting was: 2023-02-03.
    assert resolved == [
        (FOMC_HIKE, None, "hike", date(2023, 2, 3)),
        (FOMC_HIKE, 7, "hike", date(2023, 2, 3)),
    ]
    for forecast_id in (monthly, week_ahead):
        assert float(scores(db, forecast_id)[2]) == pytest.approx(0.26)  # type: ignore[arg-type]
    assert scores(db, june)[3] is False

    resolve_and_score(db, STORE, date(2023, 6, 16))
    assert rows(
        db,
        "SELECT actual_outcome FROM resolutions r JOIN questions q"
        " ON q.id = r.question_id WHERE q.target_date = %s",
        FOMC_JUNE,
    ) == [("hold",)]
    assert float(scores(db, june)[2]) == pytest.approx(0.01 + 0.49 + 0.36)  # type: ignore[arg-type]


def test_reruns_change_nothing_and_late_forecasts_get_scored(db: Connection) -> None:
    numeric_forecast(db, "unemployment", date(2023, 3, 1), 3.8, 3.4, 4.0)
    db.commit()
    resolve_and_score(db, STORE, date(2023, 6, 30))
    again = resolve_and_score(db, STORE, date(2023, 6, 30))
    assert (again.resolved, again.scored) == ([], 0)
    assert rows(db, "SELECT count(*) FROM resolutions") == [(1,)]

    # A backtest forecast added after resolution is scored on the next run.
    late = numeric_forecast(
        db,
        "unemployment",
        date(2023, 3, 1),
        3.5,
        3.3,
        3.7,
        model_type="lightgbm",
        is_backtest=True,
    )
    db.commit()
    assert resolve_and_score(db, STORE, date(2023, 6, 30)).scored == 1
    assert float(scores(db, late)[0]) == pytest.approx(-0.1)  # type: ignore[arg-type]


def test_score_summary_by_target_horizon_and_model(db: Connection) -> None:
    for period, point in ((date(2023, 2, 1), 3.8), (date(2023, 3, 1), 3.2)):
        numeric_forecast(db, "unemployment", period, point, point - 0.3, point + 0.3)
    numeric_forecast(
        db, "unemployment", date(2023, 3, 1), 9.0, 8.0, 10.0, is_backtest=True,
        model_type="lightgbm",
    )  # fmt: skip
    probs = {"cut": 0.1, "hold": 0.3, "hike": 0.6}
    fomc_forecast(db, FOMC_HIKE, probs)
    fomc_forecast(db, FOMC_HIKE, probs, horizon_months=None, lead_days=7)
    db.commit()
    resolve_and_score(db, STORE, date(2023, 6, 30))

    live = score_summary(scored_forecasts(db))
    assert set(zip(live["target"], live["horizon"], live["model_type"])) == {
        ("unemployment", "1 months", "random_walk"),
        ("fomc_decision", "1 months", "fomc_lightgbm"),
        ("fomc_decision", "7 days", "fomc_lightgbm"),
    }
    unemployment = live[live["target"] == "unemployment"].iloc[0]
    # Errors +0.2 (Feb: 3.8 vs 3.6) and -0.4 (Mar: 3.2 vs 3.6).
    assert unemployment["n"] == 2
    assert unemployment["rmse"] == pytest.approx(((0.04 + 0.16) / 2) ** 0.5)
    assert unemployment["bias"] == pytest.approx(-0.1)
    assert unemployment["coverage"] == pytest.approx(0.5)
    # Widths 0.6. Feb's 3.5-4.1 holds 3.6; March's 2.9-3.5 misses it by 0.1.
    # An interval score is the width plus 10x any miss.
    assert unemployment["interval_score"] == pytest.approx((0.6 + 0.6 + 10 * 0.1) / 2)
    # Against the latest values (March revised to 3.5): +0.2 and -0.3.
    assert unemployment["mae_latest"] == pytest.approx(0.25)
    assert unemployment["bias_latest"] == pytest.approx(-0.05)
    assert unemployment["rmse_latest"] == pytest.approx(((0.04 + 0.09) / 2) ** 0.5)
    week = live[live["horizon"] == "7 days"].iloc[0]
    assert week["brier"] == pytest.approx(0.26) and pd.isna(week["rmse"])
    assert pd.isna(week["mae_latest"])

    with_backtest = score_summary(scored_forecasts(db, include_backtest=True))
    assert with_backtest["is_backtest"].sum() == 1


def test_latest_value_follows_revisions_but_scores_do_not(db: Connection) -> None:
    unrate = numeric_forecast(db, "unemployment", date(2023, 3, 1), 3.8, 3.4, 4.0)
    numeric_forecast(db, "cpi_yoy", date(2023, 3, 1), 4.0, 3.0, 5.0)
    fomc_forecast(db, FOMC_HIKE, {"cut": 0.1, "hold": 0.3, "hike": 0.6})
    db.commit()

    def latest() -> dict[str, tuple[object, object]]:
        return {
            target: (None if value is None else float(value), as_of)  # type: ignore[arg-type]
            for target, value, as_of in rows(
                db,
                "SELECT q.target::text, r.latest_value, r.latest_as_of"
                " FROM resolutions r JOIN questions q ON q.id = r.question_id",
            )
        }

    # Before the 2023-05-10 revisions the latest value is the first release.
    result = resolve_and_score(db, STORE, date(2023, 5, 1))
    assert result.latest_refreshed == 2
    first = latest()
    assert first["unemployment"] == (pytest.approx(3.6), date(2023, 4, 10))
    assert first["cpi_yoy"] == (pytest.approx(5.0), date(2023, 4, 10))
    assert first["fomc_decision"] == (None, None)

    # No new vintage of UNRATE or CPIAUCSL: nothing to refresh.
    assert resolve_and_score(db, STORE, date(2023, 5, 1)).latest_refreshed == 0

    # After them: UNRATE March 3.5; CPI YoY 105 / 101 (the revised base).
    assert resolve_and_score(db, STORE, date(2023, 5, 10)).latest_refreshed == 2
    revised = latest()
    assert revised["unemployment"] == (pytest.approx(3.5), date(2023, 5, 10))
    assert revised["cpi_yoy"] == (
        pytest.approx((105 / 101 - 1) * 100),
        date(2023, 5, 10),
    )
    # The official resolution and score are unchanged.
    assert rows(
        db,
        "SELECT count(*) FROM resolutions WHERE actual_value IS NOT NULL"
        " AND actual_as_of = '2023-04-10'",
    ) == [(2,)]
    assert float(scores(db, unrate)[0]) == pytest.approx(0.2)  # type: ignore[arg-type]

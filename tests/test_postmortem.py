"""Tests for the post-mortem classifier.

Rule tests call `classify_numeric` / `classify_fomc` directly. The database
tests use a hand-built store: unemployment for Jan-Jun 2023 (each month
published on the 10th of the next), with March first released as 3.6 and
revised to 3.3 on 2023-05-10; a fed funds hike at the 2023-02-01 meeting
that the 3-month bill priced (bill 0.2 above the target beforehand).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2
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
from eco_prediction.db.postmortems import list_postmortems, save_postmortem
from eco_prediction.postmortem.classifier import (
    classify_fomc,
    classify_numeric,
    run_postmortems,
)
from eco_prediction.scheduler.resolution_job import resolve_and_score

RNG = np.random.default_rng(20)
CALM_MOVES = RNG.normal(0, 0.2, 240)  # 99th percentile about 0.5


# Numeric rules


def test_near_miss_is_expected_variance() -> None:
    # Half-width 0.5, band 0.75; error -0.6 is outside the interval but inside the band.
    c = classify_numeric(
        prediction=3.0,
        lower=2.5,
        upper=3.5,
        actual_first=3.6,
        actual_latest=3.6,
        last_known=3.5,
        historical_moves=CALM_MOVES,
    )
    assert c.category == "expected_variance"
    assert c.evidence["expected_band"] == pytest.approx(0.75)
    assert c.revised_data_impact == pytest.approx(0)


def test_a_revision_that_rescues_the_forecast_is_bad_data() -> None:
    c = classify_numeric(
        prediction=3.0,
        lower=2.8,
        upper=3.2,
        actual_first=3.6,  # error -0.6, band 0.3
        actual_latest=3.2,  # error -0.2: inside the band
        last_known=3.5,
        historical_moves=CALM_MOVES,
    )
    assert c.category == "bad_data"
    assert c.revised_data_impact == pytest.approx(0.4)


def test_a_small_revision_that_only_tips_the_band_is_not_bad_data() -> None:
    # Error +0.7, band 0.6; a 0.1 revision brings it to +0.6 (inside the band)
    # but removes only 14% of the error.
    c = classify_numeric(
        prediction=6.7,
        lower=6.3,
        upper=7.1,
        actual_first=6.0,
        actual_latest=6.1,
        last_known=6.2,
        historical_moves=CALM_MOVES,
    )
    assert c.category == "bad_model"
    assert c.revised_data_impact == pytest.approx(0.1)


def test_a_move_beyond_history_is_regime_change() -> None:
    c = classify_numeric(
        prediction=4.5,
        lower=4.2,
        upper=4.8,
        actual_first=14.7,  # April 2020
        actual_latest=14.8,
        last_known=4.4,
        historical_moves=CALM_MOVES,
    )
    assert c.category == "regime_change"
    assert c.evidence["move_from_last_known"] == pytest.approx(10.3)
    assert c.evidence["regime_threshold"] < 1


def test_a_normal_move_missed_is_bad_model() -> None:
    c = classify_numeric(
        prediction=3.0,
        lower=2.9,
        upper=3.1,
        actual_first=3.6,
        actual_latest=3.6,
        last_known=3.5,  # a 0.1 move: ordinary
        historical_moves=CALM_MOVES,
    )
    assert c.category == "bad_model"


def test_regime_change_needs_enough_history() -> None:
    c = classify_numeric(
        prediction=4.5,
        lower=4.2,
        upper=4.8,
        actual_first=14.7,
        actual_latest=14.7,
        last_known=4.4,
        historical_moves=CALM_MOVES[:20],
    )
    assert c.category == "bad_model" and c.evidence["regime_threshold"] is None


def test_without_an_interval_bad_data_needs_half_the_error_revised_away() -> None:
    def classify(latest: float) -> str:
        return classify_numeric(
            prediction=3.0,
            lower=None,
            upper=None,
            actual_first=3.6,
            actual_latest=latest,
            last_known=3.5,
            historical_moves=CALM_MOVES,
        ).category

    assert classify(3.25) == "bad_data"  # error -0.6 -> -0.25: more than half gone
    assert classify(3.45) == "bad_model"  # -0.6 -> -0.45: not enough


# FOMC rules


@pytest.mark.parametrize(
    ("probabilities", "outcome", "spread", "category"),
    [
        ({"cut": 0.1, "hold": 0.6, "hike": 0.3}, "hike", 0.0, "expected_variance"),
        ({"cut": 0.05, "hold": 0.9, "hike": 0.05}, "hike", 0.2, "bad_model"),
        ({"cut": 0.05, "hold": 0.9, "hike": 0.05}, "hike", 0.02, "regime_change"),
        ({"cut": 0.05, "hold": 0.9, "hike": 0.05}, "cut", -0.25, "bad_model"),
        ({"cut": 0.5, "hold": 0.1, "hike": 0.4}, "hold", 0.03, "bad_model"),
        ({"cut": 0.5, "hold": 0.1, "hike": 0.4}, "hold", 0.3, "regime_change"),
        ({"cut": 0.05, "hold": 0.9, "hike": 0.05}, "hike", float("nan"), "bad_model"),
    ],
    ids=[
        "had-a-chance",
        "priced-hike",
        "unpriced-hike",
        "priced-cut",
        "priced-hold",
        "unpriced-hold",
        "no-bill-data",
    ],
)
def test_fomc_rules(
    probabilities: dict[str, float], outcome: str, spread: float, category: str
) -> None:
    c = classify_fomc(probabilities, outcome, spread)
    assert c.category == category
    assert c.revised_data_impact is None


# Database


def _frame(rows: list[tuple[date, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])


def build_store() -> VintageStore:
    unrate = [
        (date(2023, m, 1), date(2023, m + 1, 10), v)
        for m, v in zip(range(1, 7), [3.4, 3.6, 3.6, 3.4, 3.7, 3.6])
    ]
    unrate.append((date(2023, 3, 1), date(2023, 5, 10), 3.3))  # revision
    days = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    meeting = date(2023, 2, 1)
    upper = np.array([4.5 if d.date() <= meeting else 4.75 for d in days])
    bill = upper - 0.125 + np.where(days.date <= meeting, 0.2, 0.0)

    def daily(values: np.ndarray) -> pd.DataFrame:
        return _frame(
            [
                (d.date(), (d + timedelta(days=1)).date(), float(v))
                for d, v in zip(days, values)
            ]
        )

    return VintageStore(
        {"UNRATE": _frame(unrate), "DFEDTARU": daily(upper), "DTB3": daily(bill)}
    )


STORE = build_store()


@pytest.fixture
def db(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def add_forecasts(conn: Connection) -> dict[str, int]:
    version = get_or_create_model_version(conn, "lightgbm", "test")
    march, _ = get_or_create_question(conn, "unemployment", date(2023, 3, 1), 1)
    meeting, _ = get_or_create_question(conn, "fomc_decision", date(2023, 2, 1), 1)
    ids = {
        # 3.2 +/- 0.1 vs 3.6 first, 3.3 after revision: band 0.15.
        "march": save_forecast(
            conn, march, version, date(2023, 2, 1), Forecast(3.2, 3.1, 3.3)
        ),
        # 5% on a hike the bill priced (+0.2).
        "fomc": save_probability_forecast(
            conn,
            meeting,
            version,
            date(2023, 1, 24),
            {"cut": 0.05, "hold": 0.9, "hike": 0.05},
        ),
    }
    conn.commit()
    return ids


def categories(conn: Connection) -> dict[int, tuple[str, str]]:
    frame = list_postmortems(conn)
    return {
        int(fid): (str(category), str(by))
        for fid, category, by in zip(
            frame["forecast_id"], frame["category"], frame["classified_by"]
        )
    }


def test_postmortems_and_a_revision_that_arrives_later(db: Connection) -> None:
    ids = add_forecasts(db)
    resolve_and_score(db, STORE, date(2023, 4, 20))  # before the revision
    result = run_postmortems(db, STORE, date(2023, 4, 20))
    assert result.classified == 2
    # Too little history for a regime threshold, no revision yet: bad_model.
    assert categories(db) == {
        ids["march"]: ("bad_model", "auto"),
        ids["fomc"]: ("bad_model", "auto"),
    }
    with db.cursor() as cur:
        cur.execute(
            "SELECT evidence->>'tbill_spread', notes FROM postmortems WHERE forecast_id = %s",
            (ids["fomc"],),
        )
        spread, notes = cur.fetchone()  # type: ignore[misc]
    assert float(spread) == pytest.approx(0.2) and "priced" in notes

    # After the 2023-05-10 revision to 3.3 the forecast's error is -0.1: bad data.
    run_postmortems(db, STORE, date(2023, 6, 30))
    assert categories(db)[ids["march"]] == ("bad_data", "auto")
    with db.cursor() as cur:
        cur.execute(
            "SELECT revised_data_impact, updated_at > created_at FROM postmortems"
            " WHERE forecast_id = %s",
            (ids["march"],),
        )
        impact, updated = cur.fetchone()  # type: ignore[misc]
    assert float(impact) == pytest.approx(0.4 - 0.1) and updated is True


def test_manual_classification_is_never_overwritten(db: Connection) -> None:
    ids = add_forecasts(db)
    resolve_and_score(db, STORE, date(2023, 6, 30))
    run_postmortems(db, STORE, date(2023, 6, 30))
    assert save_postmortem(
        db, ids["fomc"], "regime_change", notes="judgment call", classified_by="manual"
    )
    db.commit()
    result = run_postmortems(db, STORE, date(2023, 6, 30))
    assert result.classified == 1  # only the automatic one is reviewed
    assert categories(db)[ids["fomc"]] == ("regime_change", "manual")
    # An automatic save can't replace a manual row; a manual one can.
    assert not save_postmortem(db, ids["fomc"], "bad_model")
    assert save_postmortem(db, ids["fomc"], "bad_model", classified_by="manual")


def test_unscored_forecasts_get_no_postmortem(db: Connection) -> None:
    add_forecasts(db)
    assert run_postmortems(db, STORE, date(2023, 6, 30)).classified == 0


def test_schema_constraints(db: Connection) -> None:
    ids = add_forecasts(db)
    with pytest.raises(ValueError, match="category"):
        save_postmortem(db, ids["march"], "unlucky")  # type: ignore[arg-type]
    with pytest.raises(psycopg2.errors.InvalidTextRepresentation), db.cursor() as cur:
        cur.execute(
            "INSERT INTO postmortems (forecast_id, miss_category) VALUES (%s, 'unlucky')",
            (ids["march"],),
        )
    db.rollback()

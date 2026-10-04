"""Tests for the random walk and historical mean baselines."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, add_months
from eco_prediction.models.baselines import (
    HistoricalMeanModel,
    NotFittedError,
    RandomWalkModel,
)

START = date(2015, 1, 1)


def level(period: date) -> float:
    """Rises 0.1 a month from 5.0, with a +1/-1 zigzag on alternate months."""
    months = (period.year - START.year) * 12 + period.month - 1
    return 5.0 + 0.1 * months + (1.0 if months % 2 else -1.0)


def store(last: date = date(2023, 12, 1)) -> VintageStore:
    rows = []
    period = START
    while period <= last:
        rows.append((period, add_months(period, 1).replace(day=10), level(period)))
        period = add_months(period, 1)
    frame = pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])
    return VintageStore({"UNRATE": frame})


def view(cutoff: date) -> PointInTimeData:
    return PointInTimeData(store(), TARGETS["unemployment"], cutoff)


def test_random_walk_predicts_last_known_value() -> None:
    model = RandomWalkModel()
    data = view(date(2020, 3, 20))  # Feb 2020 published 03-10
    model.fit(data)
    forecast = model.predict(data, date(2020, 5, 1))
    assert forecast.point == pytest.approx(level(date(2020, 2, 1)))


def test_random_walk_interval_spans_past_changes_over_the_same_gap() -> None:
    model = RandomWalkModel(coverage=0.8)
    data = view(date(2020, 3, 20))
    model.fit(data)

    # Last known Feb 2020; Apr is 2 months on, so every past 2-month change
    # is exactly +0.2 and the interval collapses onto it.
    even = model.predict(data, date(2020, 4, 1))
    assert even.lower == pytest.approx(even.point + 0.2)
    assert even.upper == pytest.approx(even.point + 0.2)

    # 3-month changes alternate +2.3 and -1.7 (zigzag); the 80% band spans both.
    odd = model.predict(data, date(2020, 5, 1))
    assert odd.lower is not None and odd.upper is not None
    assert odd.lower - odd.point == pytest.approx(-1.7)
    assert odd.upper - odd.point == pytest.approx(2.3)


def test_random_walk_needs_fit() -> None:
    with pytest.raises(NotFittedError):
        RandomWalkModel().predict(view(date(2020, 3, 20)), date(2020, 4, 1))


def test_historical_mean_predicts_training_mean_with_normal_interval() -> None:
    model = HistoricalMeanModel(coverage=0.8)
    data = view(date(2020, 3, 20))
    model.fit(data)
    history = data.target_history()

    forecast = model.predict(data, date(2020, 6, 1))
    assert forecast.point == pytest.approx(history.mean())
    half = 1.2816 * history.std()
    assert forecast.lower == pytest.approx(history.mean() - half, abs=1e-3)
    assert forecast.upper == pytest.approx(history.mean() + half, abs=1e-3)


def test_historical_mean_is_stable_between_retrains() -> None:
    bt = WalkForwardBacktest(
        date(2020, 1, 1), date(2020, 12, 1), "quarterly", store=store()
    )
    results = bt.run(HistoricalMeanModel(), "unemployment", horizon_months=3)

    per_fit = results.groupby("trained_as_of")["prediction"].nunique()
    assert (per_fit == 1).all()
    assert len(per_fit) == 4
    # The trend makes each refit's mean higher than the last.
    assert (
        results.groupby("trained_as_of")["prediction"].first().is_monotonic_increasing
    )


def test_historical_mean_needs_two_values() -> None:
    sparse = VintageStore(
        {
            "UNRATE": pd.DataFrame(
                [(START, date(2015, 2, 10), 5.0)],
                columns=["observed_at", "as_of", "value"],
            )
        }
    )
    with pytest.raises(ValueError, match="at least 2"):
        HistoricalMeanModel().fit(
            PointInTimeData(sparse, TARGETS["unemployment"], date(2015, 3, 1))
        )


def test_historical_mean_needs_fit() -> None:
    with pytest.raises(NotFittedError):
        HistoricalMeanModel().predict(view(date(2020, 3, 20)), date(2020, 4, 1))


@pytest.mark.parametrize("coverage", [0, 1, 1.5])
def test_coverage_must_be_a_probability(coverage: float) -> None:
    with pytest.raises(ValueError):
        RandomWalkModel(coverage)
    with pytest.raises(ValueError):
        HistoricalMeanModel(coverage)


@pytest.mark.parametrize("model", [RandomWalkModel(), HistoricalMeanModel()])
def test_baselines_run_in_harness(model: RandomWalkModel | HistoricalMeanModel) -> None:
    bt = WalkForwardBacktest(date(2020, 1, 1), date(2021, 12, 1), store=store())
    results = bt.run(model, "unemployment", horizon_months=3)
    assert len(results) == 24
    assert results["actual"].notna().all()
    assert results[["lower", "upper"]].notna().all().all()
    assert (results["lower"] <= results["upper"]).all()

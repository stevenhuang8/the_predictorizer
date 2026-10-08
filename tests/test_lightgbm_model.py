"""Tests for the LightGBM model."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, add_months, summarize
from eco_prediction.models.baselines import NotFittedError, RandomWalkModel
from eco_prediction.models.lightgbm_model import LightGBMModel

START = date(2005, 1, 1)
LEAD = 5  # MICH leads unemployment by this many months
SMALL_GRID = {"max_depth": [3], "learning_rate": [0.1], "min_child_samples": [5, 10]}


def store(last: date = date(2023, 12, 1)) -> VintageStore:
    """MICH is white noise; unemployment is 5 + MICH from LEAD months earlier.

    Both are published on the 10th of the next month. At the end of month m
    MICH is known through m - 1, and a 3-month forecast made on the 1st of
    m + 1 targets m + 4, so the latest `mich` feature determines it.
    """
    rng = np.random.default_rng(0)
    periods: list[date] = []
    period = START
    while period <= last:
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


def view(cutoff: date) -> PointInTimeData:
    return PointInTimeData(STORE, TARGETS["unemployment"], cutoff)


def model(**kwargs: object) -> LightGBMModel:
    return LightGBMModel(3, param_grid=SMALL_GRID, **kwargs)  # type: ignore[arg-type]


def test_training_rows_line_up_with_harness_forecasts() -> None:
    X, y = model(predict_change=False).training_frame(view(date(2020, 6, 30)))
    # The row as of 2019-12-31 is a forecast made 2020-01-01 for April 2020's
    # value, which equals MICH for November 2019: the latest known then.
    assert y[pd.Timestamp(2019, 12, 31)] == pytest.approx(
        5.0 + X["mich"][pd.Timestamp(2019, 12, 31)]
    )
    # Labels stop at the latest published period, May 2020.
    assert X.index[-1] == pd.Timestamp(2020, 1, 31)


def test_change_labels_are_relative_to_latest_known_value() -> None:
    data = view(date(2020, 6, 30))
    _, levels = model(predict_change=False).training_frame(data)
    X, changes = model().training_frame(data)
    # As of 2019-12-31 unemployment is known through November 2019.
    row = pd.Timestamp(2019, 12, 31)
    assert changes[row] == pytest.approx(levels[row] - X["unrate"][row])
    # The first rows have no unemployment yet (it starts LEAD months in).
    assert len(changes) < len(levels)


def test_learns_the_leading_indicator() -> None:
    m = model()
    data = view(date(2020, 3, 31))
    m.fit(data)
    importance = m.feature_importance()
    # The label is the change from the latest unemployment, 5 + MICH - UNRATE.
    assert set(importance.index[:2]) == {"mich", "unrate"}
    assert set(importance.index) == set(m.feature_names)
    assert m.best_params["min_child_samples"] in (5, 10)
    assert m.best_params["n_estimators"] >= 1

    forecast = m.predict(data, date(2020, 7, 1))
    expected = 5.0 + STORE.known("MICH", date(2020, 3, 31))["value"].iloc[-1]
    assert forecast.point == pytest.approx(expected, abs=0.5)
    assert forecast.lower is not None and forecast.upper is not None
    assert forecast.lower <= forecast.point <= forecast.upper


def test_sparse_features_are_dropped() -> None:
    m = model()
    m.fit(view(date(2020, 3, 31)))
    # Only MICH and UNRATE exist; e.g. the oil and claims features are all NaN.
    assert "oil_mom" not in m.feature_names
    assert "claims" not in m.feature_names
    assert "unrate_lag3" in m.feature_names


def test_interval_widens_with_coverage() -> None:
    narrow, wide = model(coverage=0.5), model(coverage=0.95)
    data = view(date(2020, 3, 31))
    narrow.fit(data)
    wide.fit(data)
    n = narrow.predict(data, date(2020, 7, 1))
    w = wide.predict(data, date(2020, 7, 1))
    assert n.lower is not None and n.upper is not None
    assert w.lower is not None and w.upper is not None
    assert w.upper - w.lower > n.upper - n.lower


def test_horizon_must_match() -> None:
    m = model()
    data = view(date(2020, 3, 31))
    m.fit(data)
    with pytest.raises(ValueError, match="3 months ahead"):
        m.predict(data, date(2020, 6, 1))


def test_needs_enough_history() -> None:
    with pytest.raises(ValueError, match="at least 36"):
        model().fit(view(date(2007, 6, 30)))


def test_needs_fit() -> None:
    m = model()
    with pytest.raises(NotFittedError):
        m.predict(view(date(2020, 3, 31)), date(2020, 7, 1))
    with pytest.raises(NotFittedError):
        m.feature_importance()


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        LightGBMModel(0)
    with pytest.raises(ValueError):
        LightGBMModel(3, coverage=1.5)


def test_beats_random_walk_in_harness() -> None:
    bt = WalkForwardBacktest(date(2020, 1, 1), date(2021, 12, 1), store=STORE)
    gbm = bt.run(model(), "unemployment", horizon_months=3)
    rw = bt.run(RandomWalkModel(), "unemployment", horizon_months=3)
    assert len(gbm) == 24
    assert gbm[["prediction", "lower", "upper"]].notna().all().all()
    gbm_rmse, rw_rmse = summarize(gbm)["rmse"], summarize(rw)["rmse"]
    assert gbm_rmse is not None and rw_rmse is not None
    assert gbm_rmse < 0.5 * rw_rmse

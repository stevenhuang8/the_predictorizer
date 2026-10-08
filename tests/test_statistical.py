"""Tests for the AutoARIMA and AutoETS models."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, add_months
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.statistical import (
    ARIMAModel,
    ETSModel,
    StatsForecastModel,
    monthly_history,
)

START = date(2010, 1, 1)
MODELS = [ARIMAModel, ETSModel]


def level(period: date) -> float:
    """Rises 0.1 a month from 5.0, with a +1/-1 zigzag on alternate months."""
    months = (period.year - START.year) * 12 + period.month - 1
    return 5.0 + 0.1 * months + (1.0 if months % 2 else -1.0)


def store(
    last: date = date(2023, 12, 1),
    skip: frozenset[date] = frozenset(),
    noise: float = 0.0,
    shift_from: date | None = None,
) -> VintageStore:
    """Monthly `level` values; `shift_from` adds 5.0 from that period on."""
    rng = np.random.default_rng(0)
    rows = []
    period = START
    while period <= last:
        if period not in skip:
            value = level(period) + noise * rng.standard_normal()
            if shift_from is not None and period >= shift_from:
                value += 5.0
            rows.append((period, add_months(period, 1).replace(day=10), value))
        period = add_months(period, 1)
    frame = pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])
    return VintageStore({"UNRATE": frame})


def view(cutoff: date, source: VintageStore | None = None) -> PointInTimeData:
    return PointInTimeData(source or store(), TARGETS["unemployment"], cutoff)


@pytest.mark.parametrize("model_cls", MODELS)
def test_recovers_trend_and_zigzag(model_cls: type[StatsForecastModel]) -> None:
    model = model_cls()
    data = view(date(2020, 3, 20))  # Feb 2020 published 03-10
    model.fit(data)
    for target in (date(2020, 3, 1), date(2020, 4, 1), date(2020, 8, 1)):
        forecast = model.predict(data, target)
        assert forecast.point == pytest.approx(level(target), abs=0.05)
        assert forecast.lower is not None and forecast.upper is not None
        assert forecast.lower <= forecast.point <= forecast.upper


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_starts_from_latest_known_period(
    model_cls: type[StatsForecastModel],
) -> None:
    model = model_cls()
    model.fit(view(date(2020, 3, 20)))
    # Without refitting, predict runs the model over the history known at
    # predict time. A forecast from the training cutoff (Feb) would ignore a
    # level shift published since; this one moves toward it.
    target = date(2020, 12, 1)
    plain = model.predict(view(date(2020, 12, 20)), target)
    shifted = model.predict(
        view(date(2020, 12, 20), store(shift_from=date(2020, 6, 1))), target
    )
    assert plain.point == pytest.approx(level(target), abs=0.05)
    assert shifted.point - plain.point > 0.5
    assert model.train_start == START


@pytest.mark.parametrize("model_cls", MODELS)
def test_interval_widens_with_coverage_and_horizon(
    model_cls: type[StatsForecastModel],
) -> None:
    data = view(date(2020, 3, 20), store(noise=0.3))
    narrow, wide = model_cls(coverage=0.5), model_cls(coverage=0.95)
    narrow.fit(data)
    wide.fit(data)

    def width(model: StatsForecastModel, target: date) -> float:
        f = model.predict(data, target)
        assert f.lower is not None and f.upper is not None
        return f.upper - f.lower

    assert width(narrow, date(2020, 3, 1)) < width(wide, date(2020, 3, 1))
    assert width(wide, date(2020, 3, 1)) < width(wide, date(2020, 12, 1))


def test_gaps_are_interpolated() -> None:
    gap = date(2019, 6, 1)
    history = monthly_history(view(date(2020, 3, 20), store(skip=frozenset({gap}))))
    assert history.notna().all()
    assert history[pd.Timestamp(gap)] == pytest.approx(
        (level(date(2019, 5, 1)) + level(date(2019, 7, 1))) / 2
    )


@pytest.mark.parametrize("model_cls", MODELS)
def test_fits_across_a_missing_month(model_cls: type[StatsForecastModel]) -> None:
    data = view(date(2020, 3, 20), store(skip=frozenset({date(2019, 6, 1)})))
    model = model_cls()
    model.fit(data)
    assert np.isfinite(model.predict(data, date(2020, 5, 1)).point)


@pytest.mark.parametrize("model_cls", MODELS)
def test_needs_two_years_of_history(model_cls: type[StatsForecastModel]) -> None:
    with pytest.raises(ValueError, match="at least 24 months"):
        model_cls().fit(view(date(2011, 6, 20)))


@pytest.mark.parametrize("model_cls", MODELS)
def test_target_must_follow_last_known_period(
    model_cls: type[StatsForecastModel],
) -> None:
    model = model_cls()
    data = view(date(2020, 3, 20))
    model.fit(data)
    with pytest.raises(ValueError, match="not after the last known period"):
        model.predict(data, date(2020, 2, 1))


@pytest.mark.parametrize("model_cls", MODELS)
def test_needs_fit(model_cls: type[StatsForecastModel]) -> None:
    with pytest.raises(NotFittedError):
        model_cls().predict(view(date(2020, 3, 20)), date(2020, 4, 1))


@pytest.mark.parametrize("coverage", [0, 1, 1.5])
def test_coverage_must_be_a_probability(coverage: float) -> None:
    for model_cls in MODELS:
        with pytest.raises(ValueError):
            model_cls(coverage=coverage)


@pytest.mark.parametrize("model_cls", MODELS)
def test_runs_in_harness(model_cls: type[StatsForecastModel]) -> None:
    bt = WalkForwardBacktest(
        date(2020, 1, 1), date(2021, 12, 1), "quarterly", store=store(noise=0.1)
    )
    results = bt.run(model_cls(), "unemployment", horizon_months=3)
    assert len(results) == 24
    assert results["actual"].notna().all()
    assert results[["prediction", "lower", "upper"]].notna().all().all()
    assert (results["lower"] <= results["prediction"]).all()
    assert (results["prediction"] <= results["upper"]).all()
    assert (results["prediction"] - results["actual"]).abs().max() < 1.0

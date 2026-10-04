"""Naive baselines every model must beat.

Both implement the harness's `Forecaster` protocol:

    from eco_prediction.models.baselines import HistoricalMeanModel, RandomWalkModel

    bt.run(RandomWalkModel(), "cpi_yoy", horizon_months=3)
    bt.run(HistoricalMeanModel(), "unemployment", horizon_months=6)

Intervals are central `coverage` intervals (80% by default). For the random
walk they are empirical: the spread of past changes over the same number of
months as the forecast step. For the historical mean they assume normality,
mean +/- z * std (z = 1.28 at 80%).
"""

from __future__ import annotations

from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd

from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.backtest.harness import Forecast, months_between


class NotFittedError(Exception):
    pass


def _check_coverage(coverage: float) -> float:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    return coverage


class RandomWalkModel:
    """Predicts the last known value of the target.

    The interval adds the `coverage` quantiles of past k-month changes, where
    k is the gap from the last known period to the target period, measured on
    the training history.
    """

    def __init__(self, coverage: float = 0.8) -> None:
        self.coverage = _check_coverage(coverage)
        self.history: pd.Series | None = None

    def fit(self, data: PointInTimeData) -> None:
        self.history = data.target_history().asfreq("MS")

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.history is None:
            raise NotFittedError("call fit() first")
        latest = data.target_history()
        if latest.empty:
            raise ValueError(f"no {data.target.name} data known by {data.cutoff}")
        last = float(latest.iloc[-1])
        steps = months_between(latest.index[-1].date(), target_period)
        changes = (self.history - self.history.shift(steps)).dropna()
        if steps < 1 or changes.empty:
            return Forecast(last)
        tail = (1 - self.coverage) / 2
        low, high = np.quantile(changes.to_numpy(), [tail, 1 - tail])
        return Forecast(last, last + float(low), last + float(high))


class HistoricalMeanModel:
    """Predicts the mean of the training history, with a normal interval."""

    def __init__(self, coverage: float = 0.8) -> None:
        self.coverage = _check_coverage(coverage)
        self.z = NormalDist().inv_cdf(0.5 + coverage / 2)
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, data: PointInTimeData) -> None:
        history = data.target_history()
        if len(history) < 2:
            raise ValueError(
                f"need at least 2 {data.target.name} values to fit, got {len(history)}"
            )
        self.mean = float(history.mean())
        self.std = float(history.std())

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.mean is None or self.std is None:
            raise NotFittedError("call fit() first")
        half = self.z * self.std
        return Forecast(self.mean, self.mean - half, self.mean + half)

"""ARIMA and ETS for the AI series, on a transformed scale (AI task 14).

    from eco_prediction.ai.statistical_models import AIARIMA, AIETS

    bt.run(AIARIMA(), AI_TARGETS["ARXIV_AI_SUBMISSIONS"], horizon_months=3)
    bt.run(AIETS(), AI_TARGETS["EPOCH_SWEBENCH_SOTA"], horizon_months=6)

These wrap the economy's `ARIMAModel` and `ETSModel` (statsforecast AutoARIMA
and AutoETS, `models/statistical.py`) unchanged: the inner model is fit and
run on a transformed target history, and its forecast and interval are
transformed back. Each series kind (`ai.targets.kind_of`) has its own scale:
- **count** (arXiv, HF, GitHub): log(y + offset). Growth is multiplicative,
  so a drift on the log scale is a constant growth rate, and the interval
  widens with the level. The offset is 0, or 1 if the training history has a
  zero or negative month, fixed at fit time.
- **percent** (benchmark SOTA): log(100 + 0.1 - y), the log of the headroom
  left below 100. A record closing a share of the remaining gap is a
  constant step on this scale, and the back-transform can't pass 100.
- **log10** (largest training run): already a log scale; used as is.
Running maxima (percent, log10) are then floored at their last known value,
as in the baselines.

Back-transformed points are medians, not means: exp of a symmetric log-scale
forecast sits below the mean. For counts the gap is about half the log-scale
variance (under 1% at typical 1-3 month spreads).

Seasonality defaults to 12 months for counts (arXiv has deadline and holiday
cycles) and none for the step series. Both need `MIN_HISTORY` (24) months to
fit, like the economy's; ARC-AGI-2 (from 2025-03) has too few until 2027.
The inner models read the data through a view of the original, so the
harness's look-ahead check still sees every read.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from eco_prediction.ai.targets import Kind, kind_of
from eco_prediction.backtest.data import PointInTimeData, Target
from eco_prediction.backtest.harness import Forecast
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.statistical import (
    MIN_HISTORY,
    ARIMAModel,
    ETSModel,
    StatsForecastModel,
)

HEADROOM_EPSILON = 0.1  # percentage points, so a score of 100 has a finite log

__all__ = ["AIARIMA", "AIETS", "MIN_HISTORY", "AIStatsModel", "Scale", "scale_for"]


@dataclass(frozen=True)
class Scale:
    """A monotone transform of the target and its inverse."""

    name: str
    forward: Callable[[pd.Series], pd.Series]
    inverse: Callable[[float], float]
    decreasing: bool = False  # the inverse flips an interval's ends


def scale_for(kind: Kind, history: pd.Series) -> Scale:
    """The scale a series of `kind` is modelled on, given its training history."""
    if kind == "count":
        offset = 0.0 if history.empty or history.min() > 0 else 1.0
        return Scale(
            f"log(y + {offset:g})",
            lambda s: np.log(s + offset),
            lambda x: math.exp(x) - offset,
        )
    if kind == "percent":
        top = 100 + HEADROOM_EPSILON
        return Scale(
            "log(100.1 - y)",
            lambda s: np.log(top - s),
            lambda x: top - math.exp(x),
            decreasing=True,
        )
    return Scale("y", lambda s: s, lambda x: x)


class _ScaledView:
    """A `PointInTimeData` whose target history is on `scale`. Reads go through
    the original view, so its look-ahead bookkeeping is kept."""

    def __init__(self, data: PointInTimeData, scale: Scale) -> None:
        self.data = data
        self.scale = scale
        self.target: Target = data.target
        self.cutoff = data.cutoff

    def target_history(self, start: date | None = None) -> pd.Series:
        return self.scale.forward(self.data.target_history(start))


class AIStatsModel:
    """An economy statsforecast model run on an AI series' own scale."""

    def __init__(
        self,
        build: Callable[[int, float], StatsForecastModel],
        season_length: int | None = None,
        coverage: float = 0.8,
    ) -> None:
        self.build = build
        self.season_length = season_length
        self.coverage = coverage
        self.inner: StatsForecastModel | None = None
        self.scale: Scale | None = None

    def fit(self, data: PointInTimeData) -> None:
        kind = kind_of(data.target.series_id)
        season = self.season_length or (12 if kind == "count" else 1)
        self.scale = scale_for(kind, data.target_history())
        self.inner = self.build(season, self.coverage)
        self.inner.fit(self._view(data))

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.inner is None or self.scale is None:
            raise NotFittedError("call fit() first")
        kind = kind_of(data.target.series_id)
        f = self.inner.predict(self._view(data), target_period)
        inv = self.scale.inverse
        point = inv(f.point)
        lower = None if f.lower is None else inv(f.lower)
        upper = None if f.upper is None else inv(f.upper)
        if self.scale.decreasing:
            lower, upper = upper, lower
        if kind != "count":
            last = float(data.target_history().iloc[-1])
            point = max(point, last)
            lower = None if lower is None else max(lower, last)
            upper = None if upper is None else max(upper, last)
        return Forecast(point, lower, upper)

    def _view(self, data: PointInTimeData) -> PointInTimeData:
        assert self.scale is not None
        return cast(PointInTimeData, _ScaledView(data, self.scale))


class AIARIMA(AIStatsModel):
    """AutoARIMA on the series' scale (orders chosen by AICc at each fit)."""

    def __init__(self, season_length: int | None = None, coverage: float = 0.8) -> None:
        super().__init__(lambda s, c: ARIMAModel(s, c), season_length, coverage)


class AIETS(AIStatsModel):
    """AutoETS on the series' scale (components chosen by AICc at each fit)."""

    def __init__(self, season_length: int | None = None, coverage: float = 0.8) -> None:
        super().__init__(lambda s, c: ETSModel(s, c), season_length, coverage)

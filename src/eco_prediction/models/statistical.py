"""Statistical time-series models (statsforecast's AutoARIMA and AutoETS).

Both implement the harness's `Forecaster` protocol, like the baselines:

    from eco_prediction.models.statistical import ARIMAModel, ETSModel

    bt.run(ARIMAModel(), "cpi_yoy", horizon_months=3)
    bt.run(ETSModel(season_length=1), "unemployment", horizon_months=6)

`fit` selects and estimates the model on the training view. `predict` keeps
those parameters but runs the model over the target history known at predict
time (statsforecast's `forward`), so between retrains a forecast starts from
the latest published month and uses revised values, rather than from the
training cutoff. The history passed to `forward` starts where the training
history did, so a rolling window's states line up with the fit.

The step count is the gap from the last known period to `target_period`, and
the forecast for that step is returned with its central `coverage` interval
(the model's own, 80% by default).

Gaps in the monthly history (e.g. October 2025 CPI, never published) are
filled by linear interpolation; ARIMA and ETS need an unbroken series.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA, AutoETS  # type: ignore[import-untyped]

from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.backtest.harness import Forecast, months_between
from eco_prediction.models.baselines import NotFittedError, _check_coverage

MIN_HISTORY = 24  # months


def monthly_history(data: PointInTimeData, start: date | None = None) -> pd.Series:
    """The target history on a complete monthly index, interior gaps interpolated."""
    history = data.target_history(start).asfreq("MS")
    return history.interpolate(limit_area="inside")


class StatsForecastModel:
    """Fits a statsforecast model on the target history; see the module docstring."""

    def __init__(self, coverage: float = 0.8) -> None:
        self.coverage = _check_coverage(coverage)
        self.level = round(coverage * 100, 6)
        self.fitted: Any = None
        self.train_start: date | None = None

    def build(self) -> Any:
        """A new, unfitted statsforecast model."""
        raise NotImplementedError

    def fit(self, data: PointInTimeData) -> None:
        history = monthly_history(data)
        if len(history) < MIN_HISTORY:
            raise ValueError(
                f"need at least {MIN_HISTORY} months of {data.target.name} to fit, "
                f"got {len(history)}"
            )
        self.fitted = self.build().fit(history.to_numpy())
        self.train_start = history.index[0].date()

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.fitted is None:
            raise NotFittedError("call fit() first")
        history = monthly_history(data, self.train_start)
        if history.empty:
            raise ValueError(f"no {data.target.name} data known by {data.cutoff}")
        steps = months_between(history.index[-1].date(), target_period)
        if steps < 1:
            raise ValueError(
                f"{target_period} is not after the last known period "
                f"{history.index[-1].date()}"
            )
        out = self.fitted.forward(y=history.to_numpy(), h=steps, level=[self.level])
        # Keys are "lo-80" or "lo-80.0" depending on the model; there is one level.
        (lo,) = (k for k in out if k.startswith("lo-"))
        (hi,) = (k for k in out if k.startswith("hi-"))
        return Forecast(*(float(np.asarray(out[k])[-1]) for k in ("mean", lo, hi)))


class ARIMAModel(StatsForecastModel):
    """AutoARIMA: orders chosen by stepwise AICc search at each fit."""

    def __init__(self, season_length: int = 12, coverage: float = 0.8) -> None:
        super().__init__(coverage)
        self.season_length = season_length

    def build(self) -> AutoARIMA:
        return AutoARIMA(season_length=self.season_length)


class ETSModel(StatsForecastModel):
    """AutoETS: error, trend and season components chosen by AICc at each fit."""

    def __init__(self, season_length: int = 12, coverage: float = 0.8) -> None:
        super().__init__(coverage)
        self.season_length = season_length

    def build(self) -> AutoETS:
        return AutoETS(season_length=self.season_length)

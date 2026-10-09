"""Walk-forward backtesting with point-in-time data.

Usage:
    from eco_prediction.backtest.harness import WalkForwardBacktest, summarize

    bt = WalkForwardBacktest(date(2020, 1, 1), date(2023, 12, 1), "quarterly")
    results = bt.run(model, "cpi_yoy", horizon_months=3)
    summarize(results)  # {'n_resolved': 48, 'rmse': ..., 'mae': ..., ...}

Forecasts are made on the first of each month from `start_date` to `end_date`.
A forecast made on day D uses only data published by the end of D - 1 (`as_of`
is a date, and same-day releases may come after the forecast). It targets the
period `horizon_months` after D's month: on 2020-03-01 with horizon 3, the
June 2020 value.

Models implement `Forecaster`: `fit(data)` and `predict(data, target_period)`,
where `data` is a `PointInTimeData` view that cannot return anything published
after its cutoff. The model is refit on the first forecast date and then every
`retrain_frequency`. Its training view covers all history ("expanding", from
`train_start` if given) or only the last `window_months` ("rolling").

Each forecast is scored against the actual value. By default that is the
first release, the number a forecaster would have been judged on at the time.
`resolve_with="latest"` scores against the latest vintage instead. Forecasts
whose target has not been published yet get no actual and no score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Protocol

import pandas as pd

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import (
    TARGETS,
    LookaheadError,
    PointInTimeData,
    Target,
    VintageStore,
)

RETRAIN_MONTHS = {"monthly": 1, "quarterly": 3, "annually": 12}

RetrainFrequency = Literal["monthly", "quarterly", "annually"]
Window = Literal["expanding", "rolling"]
Resolution = Literal["first_release", "latest"]


@dataclass(frozen=True)
class Forecast:
    point: float
    lower: float | None = None  # interval bounds, if the model gives them
    upper: float | None = None


class Forecaster(Protocol):
    def fit(self, data: PointInTimeData) -> None: ...

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast: ...


def month_start(d: date) -> date:
    return d.replace(day=1)


def add_months(d: date, months: int) -> date:
    """First of the month `months` after d's month."""
    index = d.year * 12 + d.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def months_between(start: date, end: date) -> int:
    """Whole months from start's month to end's month."""
    return (end.year - start.year) * 12 + end.month - start.month


class WalkForwardBacktest:
    def __init__(
        self,
        start_date: date,
        end_date: date,
        retrain_frequency: RetrainFrequency = "quarterly",
        *,
        window: Window = "expanding",
        window_months: int | None = None,
        train_start: date | None = None,
        resolve_with: Resolution = "first_release",
        store: VintageStore | None = None,
    ) -> None:
        if end_date < start_date:
            raise ValueError("end_date is before start_date")
        if retrain_frequency not in RETRAIN_MONTHS:
            raise ValueError(f"retrain_frequency must be one of {list(RETRAIN_MONTHS)}")
        if window == "rolling" and not window_months:
            raise ValueError("a rolling window needs window_months")
        if window == "expanding" and window_months is not None:
            raise ValueError("window_months only applies to a rolling window")
        if resolve_with not in ("first_release", "latest"):
            raise ValueError("resolve_with must be 'first_release' or 'latest'")
        self.start_date = start_date
        self.end_date = end_date
        self.retrain_frequency = retrain_frequency
        self.window = window
        self.window_months = window_months
        self.train_start = train_start
        self.resolve_with = resolve_with
        self.store = store or VintageStore.from_db()

    def get_forecast_dates(self) -> list[date]:
        """First of each month in [start_date, end_date]."""
        first = month_start(self.start_date)
        if first < self.start_date:
            first = add_months(first, 1)
        dates = []
        while first <= self.end_date:
            dates.append(first)
            first = add_months(first, 1)
        return dates

    def training_start(self, cutoff: date) -> date | None:
        """Earliest period the model may train on when fit at `cutoff`."""
        if self.window == "rolling":
            assert self.window_months is not None
            return add_months(month_start(cutoff), -self.window_months + 1)
        return self.train_start

    def run(
        self, model: Forecaster, target: str | Target, horizon_months: int
    ) -> pd.DataFrame:
        """Forecast, resolve and score every forecast date. One row per date."""
        if horizon_months < 1:
            raise ValueError("horizon_months must be at least 1")
        tgt = TARGETS[target] if isinstance(target, str) else target
        retrain_every = RETRAIN_MONTHS[self.retrain_frequency]

        rows: list[dict[str, Any]] = []
        trained_as_of: date | None = None
        for i, forecast_date in enumerate(self.get_forecast_dates()):
            cutoff = forecast_date - timedelta(days=1)
            if i % retrain_every == 0:
                train = PointInTimeData(
                    self.store, tgt, cutoff, start=self.training_start(cutoff)
                )
                model.fit(train)
                _check_no_lookahead(train)
                trained_as_of = cutoff

            view = PointInTimeData(self.store, tgt, cutoff)
            target_period = add_months(forecast_date, horizon_months)
            forecast = model.predict(view, target_period)
            _check_no_lookahead(view)

            actual, actual_as_of = self.get_actual_value(tgt, target_period)
            rows.append(
                {
                    "forecast_date": forecast_date,
                    "data_as_of": cutoff,
                    "trained_as_of": trained_as_of,
                    "max_as_of_seen": view.max_as_of_seen,
                    "target_period": target_period,
                    "horizon_months": horizon_months,
                    "prediction": forecast.point,
                    "lower": forecast.lower,
                    "upper": forecast.upper,
                    "actual": actual,
                    "actual_as_of": actual_as_of,
                    **score_prediction(forecast, actual),
                }
            )
        return _results_frame(rows)

    def get_actual_value(
        self, target: Target, period: date
    ) -> tuple[float | None, date | None]:
        """The target's value for `period` and the vintage it was taken from.

        (None, None) if the period hasn't been published.
        """
        return actual_value(self.store, target, period, self.resolve_with)


def actual_value(
    store: VintageStore,
    target: Target,
    period: date,
    resolve_with: Resolution = "first_release",
) -> tuple[float | None, date | None]:
    """A target period's resolved value and the vintage it comes from.

    "first_release" uses the first vintage that published the period;
    "latest" the store's latest vintage. (None, None) if not published.
    Shared by the backtest harness and the live resolution job, so both
    resolve forecasts the same way.
    """
    first = store.first_release(target.series_id, period)
    if first is None:
        return None, None
    vintage = (
        first
        if resolve_with == "first_release"
        else store.latest_vintage(target.series_id) or first
    )
    # Resolve with the transform applied to that vintage's whole series, so
    # CPI YoY uses the year-earlier value as known at the same time.
    view = PointInTimeData(store, target, vintage)
    value = view.target_history().get(pd.Timestamp(period))
    if value is None or math.isnan(value):
        return None, None
    return float(value), vintage


def score_prediction(forecast: Forecast, actual: float | None) -> dict[str, Any]:
    """Per-forecast error terms; all None if there is no actual yet."""
    if actual is None:
        return {
            "error": None,
            "abs_error": None,
            "squared_error": None,
            "in_interval": None,
        }
    error = forecast.point - actual
    in_interval = None
    if forecast.lower is not None and forecast.upper is not None:
        in_interval = forecast.lower <= actual <= forecast.upper
    return {
        "error": error,
        "abs_error": abs(error),
        "squared_error": error**2,
        "in_interval": in_interval,
    }


def summarize(results: pd.DataFrame) -> dict[str, float | int | None]:
    """Aggregate scores over the resolved forecasts of a `run`."""
    resolved = results[results["actual"].notna()]
    with_interval = resolved[resolved["lower"].notna() & resolved["upper"].notna()]
    n = len(resolved)
    pred, act = resolved["prediction"], resolved["actual"]
    return {
        "n_forecasts": len(results),
        "n_resolved": n,
        "rmse": metrics.rmse(pred, act) if n else None,
        "mae": metrics.mae(pred, act) if n else None,
        "bias": metrics.bias(pred, act) if n else None,
        "interval_coverage": metrics.interval_coverage(
            zip(with_interval["lower"], with_interval["upper"]),
            with_interval["actual"],
        )
        if len(with_interval)
        else None,
    }


def _check_no_lookahead(view: PointInTimeData) -> None:
    if view.max_as_of_seen is not None and view.max_as_of_seen > view.cutoff:
        raise LookaheadError(
            f"model saw data from {view.max_as_of_seen}, after cutoff {view.cutoff}"
        )


_FLOAT_COLUMNS = [
    "prediction",
    "lower",
    "upper",
    "actual",
    "error",
    "abs_error",
    "squared_error",
]


def _results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "forecast_date",
            "data_as_of",
            "trained_as_of",
            "max_as_of_seen",
            "target_period",
            "horizon_months",
            "prediction",
            "lower",
            "upper",
            "actual",
            "actual_as_of",
            "error",
            "abs_error",
            "squared_error",
            "in_interval",
        ],
    )
    frame[_FLOAT_COLUMNS] = frame[_FLOAT_COLUMNS].astype("float64")
    frame["in_interval"] = frame["in_interval"].astype("boolean")
    return frame

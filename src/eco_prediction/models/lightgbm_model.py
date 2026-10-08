"""LightGBM regression on engineered features, tuned by time-series CV.

Implements the harness's `Forecaster` protocol, like the baselines:

    from eco_prediction.models.lightgbm_model import LightGBMModel

    bt.run(LightGBMModel(horizon_months=3), "cpi_yoy", horizon_months=3)

The model forecasts directly: one model per horizon, so `horizon_months` must
match the harness's. A training row is the `FeatureEngineer` vector as known
at a month end, labelled with the target `horizon_months` after the following
month starts, the same gap a harness forecast made the next day spans. Labels
are the target as known at the training cutoff; rows whose target period
hasn't been published by then are left out.

With `predict_change` (the default) the model learns the change from the
latest target value known at the row's as-of date, and adds it back to the
latest value at predict time. Trees can't extrapolate beyond the labels they
were trained on, so a level model is capped at past extremes (2021 inflation,
2020 unemployment); a change model only needs the size of moves to be
familiar. Features missing in more than
`max_missing` of the training rows are dropped; other gaps stay NaN, which
LightGBM handles natively.

`fit` searches `param_grid` with `TimeSeriesSplit`. The folds are separated by
`horizon_months` rows, so no training label is published after a validation
row's as-of date. Each candidate is trained with early stopping on its
validation fold; the grid point with the lowest mean validation RMSE wins and
the final model is refit on every row with the mean best iteration count.

Intervals are empirical: the central `coverage` quantiles of the winning
candidate's out-of-fold residuals, added to the point forecast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from itertools import product
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.backtest.harness import Forecast, add_months, months_between
from eco_prediction.features.engineering import FeatureEngineer, drop_sparse
from eco_prediction.models.baselines import NotFittedError, _check_coverage

DEFAULT_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [5, 10, 20],
}
FIXED_PARAMS: dict[str, Any] = {
    "random_state": 0,
    "deterministic": True,
    "n_jobs": 1,  # the matrices are tiny; threads cost more than they save
    "verbose": -1,
}
MIN_ROWS = 36


def origin(cutoff: date) -> date:
    """First of the month a forecast made the day after `cutoff` is made in."""
    return (cutoff + timedelta(days=1)).replace(day=1)


class LightGBMModel:
    """Gradient boosting on engineered features; see the module docstring."""

    def __init__(
        self,
        horizon_months: int,
        *,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        max_estimators: int = 500,
        early_stopping_rounds: int = 50,
        max_missing: float = 0.5,
        coverage: float = 0.8,
        predict_change: bool = True,
        features: FeatureEngineer | None = None,
    ) -> None:
        if horizon_months < 1:
            raise ValueError("horizon_months must be at least 1")
        self.horizon = horizon_months
        self.param_grid = dict(param_grid or DEFAULT_GRID)
        self.n_splits = n_splits
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.max_missing = max_missing
        self.coverage = _check_coverage(coverage)
        self.predict_change = predict_change
        self.features = features or FeatureEngineer()
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []
        self.best_params: dict[str, Any] = {}
        self.cv_rmse: float | None = None
        self.residual_bounds: tuple[float, float] | None = None

    def training_frame(self, data: PointInTimeData) -> tuple[pd.DataFrame, pd.Series]:
        """Feature rows and their labels, one per published target period.

        With `predict_change` the label is the change from the latest target
        value known at the row's as-of date; rows with no such value are dropped.
        """
        labels = data.target_history().dropna()
        as_of = [
            add_months(period.date(), -self.horizon) - timedelta(days=1)
            for period in labels.index
        ]
        X = self.features.build_matrix(data, as_of)
        y = pd.Series(labels.to_numpy(), index=X.index, name=labels.name)
        if self.predict_change:
            base = pd.Series(
                [self._latest_target(data, d) for d in as_of], index=X.index
            )
            known = base.notna()
            X, y = X[known], (y - base.to_numpy())[known]
        return X, y

    @staticmethod
    def _latest_target(data: PointInTimeData, as_of: date) -> float:
        """The latest target value known at `as_of` (on or before data's cutoff)."""
        history = PointInTimeData(data.store, data.target, as_of).target_history()
        history = history.dropna()
        return float(history.iloc[-1]) if len(history) else float("nan")

    def fit(self, data: PointInTimeData) -> None:
        X, y = self.training_frame(data)
        if len(X) < MIN_ROWS:
            raise ValueError(
                f"need at least {MIN_ROWS} labelled months of {data.target.name} "
                f"to fit, got {len(X)}"
            )
        X, _ = drop_sparse(X, self.max_missing)
        X = X.loc[:, X.notna().any()]

        best: tuple[float, dict[str, Any], int, np.ndarray] | None = None
        for values in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid, values))
            rmse, iterations, residuals = self._cross_validate(X, y, params)
            if best is None or rmse < best[0]:
                best = (rmse, params, iterations, residuals)
        assert best is not None, "param_grid is empty"
        rmse, params, iterations, residuals = best

        self.best_params = {**params, "n_estimators": iterations}
        self.model = lgb.LGBMRegressor(**self.best_params, **FIXED_PARAMS)
        self.model.fit(X, y)
        self.feature_names = list(X.columns)
        self.cv_rmse = rmse
        tail = (1 - self.coverage) / 2
        low, high = np.quantile(residuals, [tail, 1 - tail])
        self.residual_bounds = (float(low), float(high))

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]
    ) -> tuple[float, int, np.ndarray]:
        """Mean validation RMSE, mean best iteration, and out-of-fold residuals."""
        splits = TimeSeriesSplit(n_splits=self.n_splits, gap=self.horizon)
        rmses, iterations, residuals = [], [], []
        for train, valid in splits.split(X):
            model = lgb.LGBMRegressor(
                **params, n_estimators=self.max_estimators, **FIXED_PARAMS
            )
            model.fit(
                X.iloc[train],
                y.iloc[train],
                eval_X=X.iloc[valid],
                eval_y=y.iloc[valid],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                ],
            )
            best = model.best_iteration_ or self.max_estimators
            error = y.iloc[valid].to_numpy() - model.predict(
                X.iloc[valid], num_iteration=best
            )
            rmses.append(float(np.sqrt(np.mean(error**2))))
            iterations.append(best)
            residuals.append(error)
        return (
            float(np.mean(rmses)),
            max(1, round(float(np.mean(iterations)))),
            np.concatenate(residuals),
        )

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.model is None or self.residual_bounds is None:
            raise NotFittedError("call fit() first")
        steps = months_between(origin(data.cutoff), target_period)
        if steps != self.horizon:
            raise ValueError(
                f"this model forecasts {self.horizon} months ahead; "
                f"{target_period} is {steps} months after {origin(data.cutoff)}"
            )
        row = self.features.build_matrix(data, [data.cutoff])
        point = float(self.model.predict(row[self.feature_names])[0])
        if self.predict_change:
            latest = data.target_history().dropna()
            if latest.empty:
                raise ValueError(f"no {data.target.name} data known by {data.cutoff}")
            point += float(latest.iloc[-1])
        low, high = self.residual_bounds
        return Forecast(point, point + low, point + high)

    def feature_importance(self) -> pd.Series:
        """Total split gain per feature, largest first."""
        if self.model is None:
            raise NotFittedError("call fit() first")
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.feature_names, name="gain").sort_values(
            ascending=False
        )

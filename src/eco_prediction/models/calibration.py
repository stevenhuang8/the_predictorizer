"""Calibrating forecast intervals and probabilities on past errors.

Usage:
    from eco_prediction.models.calibration import (
        IntervalCalibrator,
        ProbabilityCalibrator,
        calibrate_intervals,
    )

    # Backtest: refit the interval on the errors resolved before each refit.
    results = bt.run(model, "cpi_yoy", horizon_months=3)
    calibrated = calibrate_intervals(results, coverage=0.8)
    summarize(calibrated)["interval_coverage"]  # ~0.8

    # By hand: split conformal intervals from validation residuals.
    cal = IntervalCalibrator(coverage=0.8)
    cal.fit(actuals - predictions)
    cal.predict_interval(2.4)  # (2.4 - q, 2.4 + q)

    # Probabilities, binary or categorical.
    pc = ProbabilityCalibrator("isotonic")
    pc.fit_categorical(past_probabilities, past_outcomes)
    pc.calibrate_categorical({"cut": 0.2, "hold": 0.7, "hike": 0.1})

`IntervalCalibrator` is split conformal prediction: the interval half-width is
a quantile of past absolute errors, with the finite-sample correction that
guarantees at least `coverage` when future errors look like past ones
(exchangeability; economic regimes break that, so coverage is approximate).
`method="signed"` uses the lower and upper quantiles of the signed errors
instead, which also corrects a biased point forecast.

`ProbabilityCalibrator` maps raw probabilities to observed frequencies with
isotonic regression or Platt scaling (a logistic fit on the log-odds).
Categorical forecasts are calibrated one-vs-rest with one calibrator pooled
over all categories, then renormalized to sum to 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from eco_prediction.backtest import metrics
from eco_prediction.backtest.harness import Forecast, score_prediction
from eco_prediction.models.baselines import NotFittedError

IntervalMethod = Literal["absolute", "signed"]
ProbabilityMethod = Literal["isotonic", "sigmoid"]

# Probabilities are clipped this far from 0 and 1 before taking log-odds.
_LOGIT_EPS = 1e-6
# Slack for float error in rank arithmetic, e.g. 1 - 0.8 = 0.19999999999999996.
_RANK_EPS = 1e-9


def min_residuals(coverage: float, method: IntervalMethod = "absolute") -> int:
    """Fewest residuals for which a `coverage` interval is finite."""
    tail = 1 - coverage if method == "absolute" else (1 - coverage) / 2
    return math.ceil(1 / tail - _RANK_EPS) - 1


def _conformal_quantile(values: np.ndarray, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest value: split conformal's quantile."""
    rank = math.ceil((len(values) + 1) * level - _RANK_EPS)
    return float(np.sort(values)[rank - 1])


class IntervalCalibrator:
    """Split conformal intervals around a point forecast."""

    def __init__(
        self, coverage: float = 0.8, method: IntervalMethod = "absolute"
    ) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be between 0 and 1")
        if method not in ("absolute", "signed"):
            raise ValueError("method must be 'absolute' or 'signed'")
        self.coverage = coverage
        self.method = method
        self.lower_offset: float | None = None
        self.upper_offset: float | None = None
        self.n_residuals = 0

    def fit(self, residuals: Iterable[float]) -> IntervalCalibrator:
        """Fit on residuals = actual - prediction from held-out forecasts."""
        res = np.asarray(list(residuals), dtype=float)
        if np.isnan(res).any():
            raise ValueError("NaN in residuals; drop unresolved forecasts first")
        needed = min_residuals(self.coverage, self.method)
        if len(res) < needed:
            raise ValueError(
                f"{len(res)} residuals; a {self.coverage:.0%} interval needs {needed}"
            )
        if self.method == "absolute":
            q = _conformal_quantile(np.abs(res), self.coverage)
            self.lower_offset, self.upper_offset = -q, q
        else:
            tail = (1 - self.coverage) / 2
            self.lower_offset = -_conformal_quantile(-res, 1 - tail)
            self.upper_offset = _conformal_quantile(res, 1 - tail)
        self.n_residuals = len(res)
        return self

    def predict_interval(self, point_prediction: float) -> tuple[float, float]:
        if self.lower_offset is None or self.upper_offset is None:
            raise NotFittedError("call fit() first")
        return (
            point_prediction + self.lower_offset,
            point_prediction + self.upper_offset,
        )

    def apply(self, forecast: Forecast) -> Forecast:
        """The forecast with its interval replaced by the calibrated one."""
        lower, upper = self.predict_interval(forecast.point)
        return replace(forecast, lower=lower, upper=upper)


class ProbabilityCalibrator:
    """Isotonic or Platt (sigmoid) recalibration of event probabilities."""

    def __init__(self, method: ProbabilityMethod = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError("method must be 'isotonic' or 'sigmoid'")
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None

    def fit(
        self, predicted_probs: Iterable[float], actual_outcomes: Iterable[float]
    ) -> ProbabilityCalibrator:
        """Fit on past probabilities and whether each event happened (1/0)."""
        probs, hits = _binary_pairs(predicted_probs, actual_outcomes)
        if len(np.unique(hits)) < 2:
            raise ValueError("outcomes are all the same; nothing to calibrate against")
        if self.method == "isotonic":
            model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            model.fit(probs, hits)
        else:
            model = LogisticRegression(C=1e6)  # effectively unregularized Platt
            model.fit(_logit(probs), hits)
        self._model = model
        return self

    def calibrate(self, probs: Iterable[float]) -> np.ndarray:
        if self._model is None:
            raise NotFittedError("call fit() first")
        p = np.asarray(list(probs), dtype=float)
        if ((p < 0) | (p > 1)).any() or np.isnan(p).any():
            raise ValueError("probabilities must be in [0, 1]")
        if isinstance(self._model, IsotonicRegression):
            return np.asarray(self._model.predict(p), dtype=float)
        return np.asarray(self._model.predict_proba(_logit(p))[:, 1], dtype=float)

    def fit_categorical(
        self, probabilities: Sequence[Mapping[str, float]], outcomes: Sequence[str]
    ) -> ProbabilityCalibrator:
        """Fit on categorical forecasts, e.g. {"cut": .2, "hold": .7, "hike": .1}."""
        probs, hits = metrics.one_vs_rest(probabilities, outcomes)
        return self.fit(probs, hits)

    def calibrate_categorical(
        self, probabilities: Mapping[str, float]
    ) -> dict[str, float]:
        """Calibrate each category's probability, then renormalize to sum to 1.

        If calibration sends every category to 0, the input is returned as is.
        """
        categories = list(probabilities)
        calibrated = self.calibrate([probabilities[c] for c in categories])
        total = calibrated.sum()
        if total <= 0:
            return dict(probabilities)
        return {c: float(p / total) for c, p in zip(categories, calibrated)}


def _binary_pairs(
    probs: Iterable[float], outcomes: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) != len(y):
        raise ValueError(f"{len(p)} probabilities but {len(y)} outcomes")
    if len(p) == 0:
        raise ValueError("nothing to fit")
    if np.isnan(p).any() or np.isnan(y).any():
        raise ValueError("NaN in input; drop unresolved forecasts first")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrate_intervals(
    results: pd.DataFrame,
    coverage: float = 0.8,
    method: IntervalMethod = "absolute",
    *,
    max_residuals: int | None = None,
) -> pd.DataFrame:
    """Replace a backtest's intervals with conformal ones, refit at every refit.

    `results` is a `WalkForwardBacktest.run` frame. For each training fold
    (rows sharing `trained_as_of`), the calibrator is fit on the errors of
    earlier forecasts whose actual was published by that fold's cutoff, so
    fold N's errors calibrate fold N + 1 with no lookahead. `max_residuals`
    keeps only the most recent errors (a rolling window); None uses them all.

    Folds with too few resolved errors get no interval (NaN bounds, NA
    `in_interval`), so `summarize` scores coverage on calibrated rows only.
    The model's own bounds are kept in `raw_lower` / `raw_upper`, and
    `n_calibration` is the number of errors each row's interval was fit on.
    """
    if max_residuals is not None and max_residuals < min_residuals(coverage, method):
        raise ValueError("max_residuals is too small for the requested coverage")
    out = results.copy()
    out["raw_lower"] = results["lower"]
    out["raw_upper"] = results["upper"]
    out["lower"] = np.nan
    out["upper"] = np.nan
    out["in_interval"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["n_calibration"] = 0

    resolved = results[results["actual"].notna()].sort_values("forecast_date")
    for cutoff in sorted(results["trained_as_of"].dropna().unique()):
        known = resolved[resolved["actual_as_of"] <= cutoff]
        if max_residuals is not None:
            known = known.tail(max_residuals)
        if len(known) < min_residuals(coverage, method):
            continue
        cal = IntervalCalibrator(coverage, method).fit(
            known["actual"] - known["prediction"]
        )
        fold = results.index[results["trained_as_of"] == cutoff]
        for i in fold:
            row = results.loc[i]
            forecast = cal.apply(Forecast(float(row["prediction"])))
            actual = None if pd.isna(row["actual"]) else float(row["actual"])
            out.loc[i, ["lower", "upper"]] = [forecast.lower, forecast.upper]
            out.loc[i, "in_interval"] = score_prediction(forecast, actual)[
                "in_interval"
            ]
            out.loc[i, "n_calibration"] = cal.n_residuals
    return out

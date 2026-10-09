"""Cut/hold/hike probabilities for scheduled FOMC meetings.

Usage:
    from eco_prediction.models.fomc_model import FOMCForecaster

    store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)
    model = FOMCForecaster(lead_days=7)
    model.fit(store, cutoff=date(2023, 12, 31))
    model.predict_proba(store, date(2024, 9, 18))   # {'cut': .., 'hold': .., 'hike': ..}

    bt = FOMCBacktest(date(2015, 1, 1), date(2024, 12, 31), store=store)
    summarize_fomc(bt.run(model))

A forecast for a meeting is made `lead_days` before its decision day, with the
data published by the day before (the harness convention): `forecast_cutoff`.
One model serves one lead time, like `LightGBMModel` and its horizon.

A training row is one past meeting whose decision was known by the fit
cutoff, with the features as known at that meeting's forecast cutoff:
- the `FeatureEngineer` macro vector (CPI, unemployment, claims, the yield
  curve, ...), minus its calendar dummies and linear trend, which let trees
  learn the era rather than the economy;
- the fed funds target and its change over 3, 6 and 12 months;
- the last two decisions (-1 cut, 0 hold, 1 hike) and the meetings since the
  last change;
- the 3-month bill and 2-year Treasury yields minus the target, and their
  change over the last month. These carry what markets expect the Fed to do,
  the free stand-in for fed funds futures.

`fit` tunes a small LightGBM multiclass grid by multi-class log loss on
`TimeSeriesSplit` folds, with early stopping, then refits on every row. With
`calibrate=True` a `ProbabilityCalibrator` is fit on the winning candidate's
out-of-fold probabilities and applied to every forecast.

Three baselines share the interface: `AlwaysHold`, `Climatology` (decision
frequencies since `train_start`), and `Persistence` (frequencies of each
decision given the previous one, a Markov chain). All probabilities are
dicts over ("cut", "hold", "hike") summing to 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from itertools import pairwise, product
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from eco_prediction.backtest.data import VintageStore
from eco_prediction.data.fomc import (
    FOMC_MEETINGS,
    OUTCOMES,
    Outcome,
    decision,
    fed_funds_target,
    value_at,
)
from eco_prediction.features.engineering import FeatureEngineer, drop_sparse
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.calibration import ProbabilityCalibrator

DEFAULT_TRAIN_START = date(1994, 1, 1)
DEFAULT_GRID: dict[str, list[Any]] = {
    "max_depth": [2, 3],
    "min_data_in_leaf": [5, 10, 20],
    "learning_rate": [0.05],
}
FIXED_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": len(OUTCOMES),
    "metric": "multi_logloss",
    "seed": 0,
    "deterministic": True,
    "num_threads": 1,
    "verbose": -1,
}
MIN_ROWS = 40
DECISION_CODE = {"cut": -1.0, "hold": 0.0, "hike": 1.0}
# Macro features left out: calendar dummies and the linear trend.
EXCLUDED_FEATURES = ("trend_months", *(f"month_{m}" for m in range(1, 13)))
MARKET_SERIES = {"tbill": "DTB3", "t2y": "DGS2"}
MAX_STALE_DAYS = 14
# Meetings looked back over for the decision features.
DECISION_LOOKBACK = 24

Probabilities = dict[str, float]


def forecast_cutoff(meeting: date, lead_days: int) -> date:
    """Last day of data a forecast made `lead_days` before `meeting` can use."""
    return meeting - timedelta(days=lead_days + 1)


def past_decisions(
    store: VintageStore, meeting: date, as_of: date, target: pd.Series | None = None
) -> list[Outcome]:
    """Decisions of the meetings before `meeting` known on as_of, oldest first."""
    if target is None:
        target = fed_funds_target(store, as_of)
    earlier = [m for m in FOMC_MEETINGS if m < meeting][-DECISION_LOOKBACK:]
    outcomes = [decision(store, m, as_of, target) for m in earlier]
    return [o for o in outcomes if o is not None]


def rate_features(store: VintageStore, meeting: date, as_of: date) -> dict[str, float]:
    """Policy-rate and market features for `meeting` from data known on as_of."""
    nan = float("nan")
    target = fed_funds_target(store, as_of)
    at = pd.Timestamp(as_of)
    current = value_at(target, at)
    features = {"fed_target": current}
    for months in (3, 6, 12):
        before = value_at(target, at - pd.DateOffset(months=months))
        features[f"fed_change_{months}m"] = current - before

    decisions = past_decisions(store, meeting, as_of, target)
    codes = [DECISION_CODE[d] for d in decisions]
    features["last_decision"] = codes[-1] if codes else nan
    features["last_decision2"] = codes[-2] if len(codes) > 1 else nan
    since = next((i for i, c in enumerate(reversed(codes)) if c != 0), len(codes))
    features["meetings_since_change"] = float(since) if codes else nan

    for name, series_id in MARKET_SERIES.items():
        known = store.known(series_id, as_of)
        values = pd.Series(
            known["value"].to_numpy(), index=pd.DatetimeIndex(known["observed_at"])
        ).dropna()
        values = values[values.index <= at]
        if values.empty or (at - values.index[-1]).days > MAX_STALE_DAYS:
            features[f"{name}_spread"] = nan
            features[f"{name}_change_1m"] = nan
            continue
        latest = float(values.iloc[-1])
        month_ago = value_at(values, at - pd.DateOffset(months=1))
        features[f"{name}_spread"] = latest - current
        features[f"{name}_change_1m"] = latest - month_ago
    return features


def _to_probabilities(row: np.ndarray) -> Probabilities:
    return {o: float(p) for o, p in zip(OUTCOMES, row)}


class FOMCModel(Protocol):
    lead_days: int

    def fit(self, store: VintageStore, cutoff: date) -> None: ...

    def predict_proba(self, store: VintageStore, meeting: date) -> Probabilities: ...


class _DecisionHistory:
    """Shared by the baselines: labelled meetings in [train_start, cutoff]."""

    def __init__(self, lead_days: int, train_start: date) -> None:
        self.lead_days = lead_days
        self.train_start = train_start

    def labelled(self, store: VintageStore, cutoff: date) -> list[Outcome]:
        target = fed_funds_target(store, cutoff)
        meetings = [m for m in FOMC_MEETINGS if self.train_start <= m <= cutoff]
        outcomes = [decision(store, m, cutoff, target) for m in meetings]
        return [o for o in outcomes if o is not None]


class AlwaysHold(_DecisionHistory):
    """All probability on 'hold'."""

    def __init__(
        self, lead_days: int = 7, train_start: date = DEFAULT_TRAIN_START
    ) -> None:
        super().__init__(lead_days, train_start)

    def fit(self, store: VintageStore, cutoff: date) -> None:
        pass

    def predict_proba(self, store: VintageStore, meeting: date) -> Probabilities:
        return {"cut": 0.0, "hold": 1.0, "hike": 0.0}


class Climatology(_DecisionHistory):
    """How often each decision happened in training, add-one smoothed."""

    def __init__(
        self, lead_days: int = 7, train_start: date = DEFAULT_TRAIN_START
    ) -> None:
        super().__init__(lead_days, train_start)
        self.probabilities: Probabilities | None = None

    def fit(self, store: VintageStore, cutoff: date) -> None:
        outcomes = self.labelled(store, cutoff)
        counts = np.array([outcomes.count(o) + 1.0 for o in OUTCOMES])
        self.probabilities = _to_probabilities(counts / counts.sum())

    def predict_proba(self, store: VintageStore, meeting: date) -> Probabilities:
        if self.probabilities is None:
            raise NotFittedError("call fit() first")
        return dict(self.probabilities)


class Persistence(_DecisionHistory):
    """P(decision | previous decision) from training, add-one smoothed."""

    def __init__(
        self, lead_days: int = 7, train_start: date = DEFAULT_TRAIN_START
    ) -> None:
        super().__init__(lead_days, train_start)
        self.transitions: dict[str, Probabilities] | None = None

    def fit(self, store: VintageStore, cutoff: date) -> None:
        outcomes = self.labelled(store, cutoff)
        counts = {prev: np.ones(len(OUTCOMES)) for prev in OUTCOMES}
        for prev, nxt in pairwise(outcomes):
            counts[prev][OUTCOMES.index(nxt)] += 1
        self.transitions = {
            prev: _to_probabilities(c / c.sum()) for prev, c in counts.items()
        }

    def predict_proba(self, store: VintageStore, meeting: date) -> Probabilities:
        if self.transitions is None:
            raise NotFittedError("call fit() first")
        as_of = forecast_cutoff(meeting, self.lead_days)
        decisions = past_decisions(store, meeting, as_of)
        previous = decisions[-1] if decisions else "hold"
        return dict(self.transitions[previous])


class FOMCForecaster:
    """LightGBM multiclass on macro, rate and market features; see the module docstring."""

    def __init__(
        self,
        lead_days: int = 7,
        *,
        train_start: date = DEFAULT_TRAIN_START,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        max_estimators: int = 300,
        early_stopping_rounds: int = 30,
        max_missing: float = 0.5,
        calibrate: bool = False,
        features: FeatureEngineer | None = None,
    ) -> None:
        if lead_days < 0:
            raise ValueError("lead_days must be at least 0")
        self.lead_days = lead_days
        self.train_start = train_start
        self.param_grid = dict(param_grid or DEFAULT_GRID)
        self.n_splits = n_splits
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.max_missing = max_missing
        self.calibrate = calibrate
        self.features = features or FeatureEngineer()
        self.booster: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.best_params: dict[str, Any] = {}
        self.cv_log_loss: float | None = None
        self.calibrator: ProbabilityCalibrator | None = None

    def feature_row(self, store: VintageStore, meeting: date) -> dict[str, float]:
        """Every feature for `meeting`, from data known at its forecast cutoff."""
        as_of = forecast_cutoff(meeting, self.lead_days)
        macro = self.features.build_features_as_of(store, as_of)
        row = {
            k: float("nan") if v is None else v
            for k, v in macro.items()
            if k not in EXCLUDED_FEATURES
        }
        return {**row, **rate_features(store, meeting, as_of)}

    def training_frame(
        self, store: VintageStore, cutoff: date
    ) -> tuple[pd.DataFrame, pd.Series]:
        """One row per meeting in [train_start, cutoff] decided by the cutoff.

        Labels are outcome indexes into OUTCOMES (0 cut, 1 hold, 2 hike).
        """
        target = fed_funds_target(store, cutoff)
        rows, labels, index = [], [], []
        for meeting in FOMC_MEETINGS:
            if not self.train_start <= meeting <= cutoff:
                continue
            outcome = decision(store, meeting, cutoff, target)
            if outcome is None:
                continue
            rows.append(self.feature_row(store, meeting))
            labels.append(OUTCOMES.index(outcome))
            index.append(pd.Timestamp(meeting))
        meetings = pd.DatetimeIndex(index, name="meeting")
        X = pd.DataFrame(rows, index=meetings, dtype="float64")
        return X, pd.Series(labels, index=meetings, name="outcome")

    def fit(self, store: VintageStore, cutoff: date) -> None:
        X, y = self.training_frame(store, cutoff)
        if len(X) < MIN_ROWS:
            raise ValueError(
                f"need at least {MIN_ROWS} decided meetings to fit, got {len(X)}"
            )
        X, _ = drop_sparse(X, self.max_missing)
        X = X.loc[:, X.notna().any()]

        best: tuple[float, dict[str, Any], int, np.ndarray, np.ndarray] | None = None
        for values in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid, values))
            loss, iterations, oof, oof_y = self._cross_validate(X, y, params)
            if best is None or loss < best[0]:
                best = (loss, params, iterations, oof, oof_y)
        assert best is not None, "param_grid is empty"
        loss, params, iterations, oof, oof_y = best

        self.best_params = {**params, "num_iterations": iterations}
        self.booster = lgb.train(
            {**params, **FIXED_PARAMS},
            lgb.Dataset(X, y),
            num_boost_round=iterations,
        )
        self.feature_names = list(X.columns)
        self.cv_log_loss = loss
        self.calibrator = None
        if self.calibrate:
            outcomes = [OUTCOMES[i] for i in oof_y]
            probs = [_to_probabilities(row) for row in oof]
            if len(set(outcomes)) > 1:
                self.calibrator = ProbabilityCalibrator().fit_categorical(
                    probs, outcomes
                )

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]
    ) -> tuple[float, int, np.ndarray, np.ndarray]:
        """Mean validation log loss, mean best iteration, out-of-fold probabilities.

        A one-row gap keeps the meeting just before a validation fold, whose
        decision is announced after the fold's first forecast, out of training.
        """
        splits = TimeSeriesSplit(n_splits=self.n_splits, gap=1)
        losses, iterations, probs, labels = [], [], [], []
        for train, valid in splits.split(X):
            train_set = lgb.Dataset(X.iloc[train], y.iloc[train])
            valid_set = lgb.Dataset(X.iloc[valid], y.iloc[valid], reference=train_set)
            booster = lgb.train(
                {**params, **FIXED_PARAMS},
                train_set,
                num_boost_round=self.max_estimators,
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                ],
            )
            best = booster.best_iteration or self.max_estimators
            p = np.asarray(booster.predict(X.iloc[valid], num_iteration=best))
            truth = y.iloc[valid].to_numpy()
            losses.append(_log_loss(p, truth))
            iterations.append(best)
            probs.append(p)
            labels.append(truth)
        return (
            float(np.mean(losses)),
            max(1, round(float(np.mean(iterations)))),
            np.concatenate(probs),
            np.concatenate(labels),
        )

    def prediction_inputs(self, store: VintageStore, meeting: date) -> pd.DataFrame:
        if self.booster is None:
            raise NotFittedError("call fit() first")
        row = self.feature_row(store, meeting)
        return pd.DataFrame([row], dtype="float64").reindex(columns=self.feature_names)

    def predict_proba(self, store: VintageStore, meeting: date) -> Probabilities:
        X = self.prediction_inputs(store, meeting)
        assert self.booster is not None
        probs = _to_probabilities(np.asarray(self.booster.predict(X))[0])
        if self.calibrator is not None:
            probs = self.calibrator.calibrate_categorical(probs)
        return probs

    def feature_importance(self) -> pd.Series:
        """Total split gain per feature, largest first."""
        if self.booster is None:
            raise NotFittedError("call fit() first")
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.feature_names, name="gain").sort_values(
            ascending=False
        )


def _log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-15, 1))))

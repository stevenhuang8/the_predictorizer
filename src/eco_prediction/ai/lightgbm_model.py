"""LightGBM on the AI features: numeric forecasts and threshold probabilities
(AI task 15).

    from eco_prediction.ai.lightgbm_model import AILightGBMModel, AIThresholdClassifier

    bt.run(AILightGBMModel(3), AI_TARGETS["ARXIV_AI_SUBMISSIONS"], horizon_months=3)
    bt.run(AILightGBMModel(3, pool=True), AI_TARGETS["EPOCH_GPQA_SOTA"], horizon_months=3)

    clf = AIThresholdClassifier().fit(BinaryData.from_db(conn), date(2026, 10, 14))
    evaluate_binary(clf, binary_questions(conn), data)

**Numeric** (`AILightGBMModel`, the harness's `Forecaster`): the economy's
`LightGBMModel` (`models/lightgbm_model.py`, unchanged) run on the
`AIFeatureEngineer` features with the target's `target_*` aliases. Its
direct-forecast rows, change labels, time-ordered grid search with early
stopping, and residual intervals all carry over. What changes is the scale:
the target is modelled on Task 14's scale for its kind (`scale_for`), so the
label is a change on that scale:
- count: the log change (a growth rate);
- percent: the change in log headroom (the share of the gap to 100 closed);
- log10: the change in log10 FLOP.
The forecast and interval are transformed back, and running maxima are
clipped to [last value, 100 if a percentage], as in the baselines.

One difference in the rows: they are dated on the fit cutoff's day of the
month. At the harness's month-end cutoffs that is the economy's rule, but a
live run mid-month already knows the previous month, and month-end rows
would teach it one month too many of growth (`row_offset`).

**Trending features are left out by default** (`trending=False`, both
models): the levels, logs and lags of the counts and of compute, their log
ratios, and `trend_months` (`is_trending`). They grow with time, so trees
use them as a calendar ("after 2020, growth was slow") and then extrapolate
that past the training period. What's left describes the moment without
the date: growth rates, changes, headroom, months since a change, the
cross-series correlations and the calendar month.

Every AI series' features are in every row, so cross-series signals (HF's
growth for arXiv) are available to the trees. `pool=True` also trains on
every series of the target's kind, one row per series and month, with
`series_<prefix>` indicator columns. The benchmark series are too short to
fit alone (`MIN_ROWS` is 36 labelled months), but the three together are not.
Pooled cross-validation splits by date, so all series' rows for a month fall
in the same fold.

`SHAPExplainer` (`models/explainability.py`) works unchanged; its
contributions are on the model's scale (log change for counts).

**Binary** (`AIThresholdClassifier`, `predict_probability` like the
baselines): a LightGBM classifier (`objective='binary'`) per series kind, for
the threshold questions ('SERIES>=T'). History has almost no resolved
threshold questions, so training rows are the questions that could have been
asked: for each series of the kind, past base month b, horizon h and the
generator's steps (+10/25/50%, 25/50/75% of the gap, 2x/3x/10x FLOP), was
the value h + 1 months later at or above the step? A row's features are the
AI features as of b's first release, with the step restated relative to the
base (`step`: log ratio, gap share or log10 difference) and `horizon_months`.
It is tuned like the regressor, on log loss, with folds that drop training
rows whose outcome was published after the validation period starts. The raw
probabilities are Platt-scaled (`ProbabilityCalibrator('sigmoid')`) on the
winning candidate's out-of-fold predictions, when there are enough of both
outcomes, and kept within [0.01, 0.99]: a tree that has never seen an
outcome happen will otherwise say it can't.

The classifier is fit once at a cutoff; asking it about a question whose
cutoff is earlier raises `LookaheadError`. Curated (Polymarket) questions
raise `NoForecast`: the market price is the baseline to beat there, and the
LLM forecaster (tasks 18-19) is meant for them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from itertools import product
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from eco_prediction.ai.baselines import (
    BinaryData,
    BinaryQuestion,
    NoForecast,
    _bounded,
    cleared,
    smoothed_rate,
)
from eco_prediction.ai.features import COUNT_SERIES, SERIES_PREFIX, AIFeatureEngineer
from eco_prediction.ai.questions import COUNT_STEPS, FLOP_STEPS, GAP_STEPS, HORIZONS
from eco_prediction.ai.resolutions import parse_threshold_key
from eco_prediction.ai.statistical_models import Scale, scale_for
from eco_prediction.ai.targets import AI_TARGETS, KINDS, Kind, kind_of
from eco_prediction.backtest.data import LookaheadError, PointInTimeData, Target
from eco_prediction.backtest.harness import Forecast, add_months
from eco_prediction.features.engineering import FeatureEngineer, drop_sparse
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.calibration import ProbabilityCalibrator
from eco_prediction.models.lightgbm_model import (
    DEFAULT_GRID,
    FIXED_PARAMS,
    MIN_ROWS,
    LightGBMModel,
    origin,
)

MIN_CALIBRATION = 30  # out-of-fold predictions needed to Platt-scale
PROBABILITY_BOUND = 0.01  # probabilities are kept in [0.01, 0.99]

__all__ = [
    "AILightGBMModel",
    "AIThresholdClassifier",
    "is_trending",
    "pool_for",
    "relative_step",
    "row_offset",
]


def pool_for(series_id: str) -> list[str]:
    """Every AI numeric target of the same kind as `series_id`, it first."""
    kind = kind_of(series_id)
    return [series_id] + [s for s, k in KINDS.items() if k == kind and s != series_id]


def _indicators(pool: Sequence[str], series_id: str) -> dict[str, float]:
    return {f"series_{SERIES_PREFIX[s]}": float(s == series_id) for s in pool}


def _scaled_target(target: Target, scale: Scale) -> Target:
    return Target(
        target.name,
        target.series_id,
        lambda s: scale.forward(target.transform(s)),
        f"{target.description} on {scale.name}",
    )


class _ScaledData(PointInTimeData):
    """A view of another series, or the same one, on a model scale. Reads go
    through `parent`, so the harness's look-ahead check sees them."""

    def __init__(self, parent: PointInTimeData, target: Target) -> None:
        super().__init__(parent.store, target, parent.cutoff, parent.start)
        self.parent = parent

    def _raw(self, series_id: str) -> pd.Series:
        values = self.parent._raw(series_id)
        self.max_as_of_seen = self.parent.max_as_of_seen
        return values


def is_trending(name: str, target_kind: Kind | None = None) -> bool:
    """Whether a feature drifts with time rather than describing the moment:
    the levels, logs and lags of the counts and of compute, their log ratios,
    and `trend_months`. With a count or log10 target, its aliases too."""
    if name == "trend_months" or name.startswith("log_ratio_"):
        return True
    prefixes = [*COUNT_SERIES, "flops"]
    if target_kind in ("count", "log10"):
        prefixes.append("target")
    return any(re.fullmatch(rf"{p}(_log|_lag\d+)?", name) for p in prefixes)


def _stationary(features: Mapping[str, Any], target_kind: Kind) -> dict[str, Any]:
    return {k: v for k, v in features.items() if not is_trending(k, target_kind)}


class _TargetFeatures:
    """`AIFeatureEngineer` rows with the view's target aliases (and, pooled,
    series indicators), in the `FeatureEngineer.build_matrix` shape."""

    def __init__(
        self, engineer: AIFeatureEngineer, pool: Sequence[str], trending: bool
    ) -> None:
        self.engineer = engineer
        self.pool = list(pool)
        self.trending = trending

    def build_matrix(
        self, data: PointInTimeData, as_of_dates: Iterable[date]
    ) -> pd.DataFrame:
        series_id = data.target.series_id
        X = self.engineer.build_matrix(data, as_of_dates, series_id)
        if not self.trending:
            kind = kind_of(series_id)
            X = X[[c for c in X.columns if not is_trending(c, kind)]].copy()
        if len(self.pool) > 1:
            for name, value in _indicators(self.pool, series_id).items():
                X[name] = value
        return X


def row_offset(cutoff: date) -> timedelta:
    """How far into its forecast month a forecast made the day after `cutoff`
    is (0 at a month end), capped so it stays inside any month."""
    return min(cutoff + timedelta(days=1) - origin(cutoff), timedelta(days=27))


def _date_folds(
    index: pd.Index, n_splits: int, gap: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`TimeSeriesSplit` over the distinct dates in `index`, as row positions."""
    dates = np.asarray(index)
    unique = np.unique(dates)
    folds = []
    for train, valid in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(unique):
        folds.append(
            (
                np.flatnonzero(np.isin(dates, unique[train])),
                np.flatnonzero(np.isin(dates, unique[valid])),
            )
        )
    return folds


class AILightGBMModel(LightGBMModel):
    """The economy's LightGBM on an AI series' scale; see the module docstring."""

    def __init__(
        self,
        horizon_months: int,
        *,
        pool: bool = False,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        max_estimators: int = 500,
        early_stopping_rounds: int = 50,
        max_missing: float = 0.5,
        coverage: float = 0.8,
        trending: bool = False,
        features: AIFeatureEngineer | None = None,
    ) -> None:
        super().__init__(
            horizon_months,
            param_grid=param_grid,
            n_splits=n_splits,
            max_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds,
            max_missing=max_missing,
            coverage=coverage,
            predict_change=True,
        )
        self.pool = pool
        self.trending = trending
        self.engineer = features or AIFeatureEngineer()
        self.series: list[str] = []
        self.scales: dict[str, Scale] = {}

    def fit(self, data: PointInTimeData) -> None:
        target = data.target.series_id
        self.series = pool_for(target) if self.pool else [target]
        self.features = cast(
            FeatureEngineer, _TargetFeatures(self.engineer, self.series, self.trending)
        )
        self.scales = {}
        for series_id in self.series:
            history = _ScaledData(data, AI_TARGETS[series_id]).target_history()
            self.scales[series_id] = scale_for(kind_of(series_id), history)
        super().fit(data)

    def _scaled(self, data: PointInTimeData, series_id: str) -> PointInTimeData:
        if series_id not in self.scales:
            raise ValueError(f"this model wasn't fit on {series_id}")
        return _ScaledData(
            data, _scaled_target(AI_TARGETS[series_id], self.scales[series_id])
        )

    def training_frame(self, data: PointInTimeData) -> tuple[pd.DataFrame, pd.Series]:
        """Rows of every pooled series, ordered by as-of date; labels on scale."""
        frames = [self._series_frame(self._scaled(data, s)) for s in self.series]
        X = pd.concat([X for X, _ in frames])
        y = pd.concat([y for _, y in frames]).rename(data.target.name)
        order = np.argsort(X.index.to_numpy(), kind="stable")
        return X.iloc[order], y.iloc[order]

    def _series_frame(self, data: PointInTimeData) -> tuple[pd.DataFrame, pd.Series]:
        """The economy's training rows, dated on the cutoff's day of the month.

        A row for period P is as of the day a forecast of P made on the same
        day of the month as `data.cutoff` would use, `horizon` months earlier.
        At the harness's month-end cutoffs that is the economy's rule (the
        day before P's month, `horizon` months back). Mid-month, after the
        previous month is published, rows know that month too, so a live
        run's label spans the same number of months as its forecast.
        """
        offset = row_offset(data.cutoff)
        labels = data.target_history().dropna()
        as_of = [
            add_months(period.date(), -self.horizon) + offset - timedelta(days=1)
            for period in labels.index
        ]
        X = self.features.build_matrix(data, as_of)
        y = pd.Series(labels.to_numpy(), index=X.index, name=labels.name)
        base = pd.Series([self._latest_target(data, d) for d in as_of], index=X.index)
        known = base.notna()
        return X[known], (y - base.to_numpy())[known]

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]
    ) -> tuple[float, int, np.ndarray]:
        """As the economy's, with folds split by date (one row per date unpooled)."""
        rmses, iterations, residuals = [], [], []
        for train, valid in _date_folds(X.index, self.n_splits, self.horizon):
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

    def prediction_inputs(
        self, data: PointInTimeData, target_period: date
    ) -> tuple[pd.DataFrame, float]:
        """The feature row, and the latest target value on the model's scale."""
        return super().prediction_inputs(
            self._scaled(data, data.target.series_id), target_period
        )

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        series_id = data.target.series_id
        f = super().predict(data, target_period)  # on the model's scale
        scale = self.scales[series_id]
        point = scale.inverse(f.point)
        lower = None if f.lower is None else scale.inverse(f.lower)
        upper = None if f.upper is None else scale.inverse(f.upper)
        if scale.decreasing:
            lower, upper = upper, lower
        last = float(data.target_history().dropna().iloc[-1])
        return _bounded(kind_of(series_id), last, Forecast(point, lower, upper))


# --- threshold questions ----------------------------------------------------------


def relative_step(kind: Kind, base: float, threshold: float) -> float:
    """A threshold as a step from its base: the log ratio for counts, the share
    of the gap to 100 for percentages, the log10 difference for compute."""
    if kind == "count":
        return math.log(threshold / base)
    if kind == "percent":
        gap = 100 - base
        return (threshold - base) / gap if gap > 0 else 1.0
    return threshold - base


def _thresholds(kind: Kind, base: float) -> list[float]:
    """The generator's steps up from base (ai/questions.py), unrounded."""
    if kind == "count":
        return [base * mult for mult, _ in COUNT_STEPS]
    if kind == "percent":
        return [base + frac * (100 - base) for frac, _ in GAP_STEPS]
    return [base + math.log10(mult) for mult, _ in FLOP_STEPS]


def _base_month(question: BinaryQuestion) -> pd.Timestamp:
    """The month a threshold was set from: the one before the asking month."""
    assert question.horizon_months is not None
    target = pd.Timestamp(question.target_date).to_period("M").to_timestamp()
    return target - pd.DateOffset(months=question.horizon_months + 1)


class _KindModel:
    """A fitted classifier for one kind, and its calibrator if there is one."""

    def __init__(
        self,
        model: lgb.LGBMClassifier | None,
        feature_names: list[str],
        calibrator: ProbabilityCalibrator | None,
        constant: float | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.calibrator = calibrator
        self.constant = constant

    def probability(self, row: pd.DataFrame) -> float:
        if self.model is None:
            assert self.constant is not None
            return self.constant
        p = float(np.asarray(self.model.predict_proba(row[self.feature_names]))[0, 1])
        if self.calibrator is not None:
            p = float(self.calibrator.calibrate([p])[0])
        return min(max(p, PROBABILITY_BOUND), 1 - PROBABILITY_BOUND)


class AIThresholdClassifier:
    """LightGBM probabilities for threshold questions; see the module docstring."""

    def __init__(
        self,
        *,
        horizons: Sequence[int] = HORIZONS,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        max_estimators: int = 500,
        early_stopping_rounds: int = 50,
        max_missing: float = 0.5,
        calibrate: bool = True,
        trending: bool = False,
        features: AIFeatureEngineer | None = None,
    ) -> None:
        self.horizons = tuple(horizons)
        self.param_grid = dict(param_grid or DEFAULT_GRID)
        self.n_splits = n_splits
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.max_missing = max_missing
        self.calibrate = calibrate
        self.trending = trending
        self.engineer = features or AIFeatureEngineer()
        self.cutoff: date | None = None
        self.models: dict[Kind, _KindModel] = {}
        self.best_params: dict[Kind, dict[str, Any]] = {}
        self.cv_log_loss: dict[Kind, float] = {}
        self.skipped: dict[Kind, str] = {}

    # --- training rows ----------------------------------------------------------

    def _row(
        self,
        data: BinaryData,
        series_id: str,
        as_of: date,
        step: float,
        horizon: int,
        pool: Sequence[str],
    ) -> dict[str, float | None]:
        features = self.engineer.build_features_as_of(data.store, as_of, series_id)
        if not self.trending:
            features = _stationary(features, kind_of(series_id))
        return {
            **features,
            **(_indicators(pool, series_id) if len(pool) > 1 else {}),
            "step": step,
            "horizon_months": float(horizon),
        }

    def training_frame(
        self, data: BinaryData, cutoff: date, kind: Kind
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Rows (indexed by feature as-of date), labels, and when each label
        was published. Only outcomes published by `cutoff` are used."""
        pool = [s for s, k in KINDS.items() if k == kind]
        rows, labels, published = [], [], []
        as_ofs: list[date] = []
        for series_id in pool:
            series = data.series(series_id, cutoff)
            for month, base in series.dropna().items():
                base_month = cast(pd.Timestamp, month)
                as_of = data.store.first_release(series_id, base_month.date())
                if as_of is None or as_of > cutoff:
                    continue
                for h in self.horizons:
                    later_month = base_month + pd.DateOffset(months=h + 1)
                    later = series.get(later_month)
                    if later is None or pd.isna(later):
                        continue
                    released = data.store.first_release(series_id, later_month.date())
                    assert released is not None
                    for threshold in _thresholds(kind, float(base)):
                        hit = cleared(kind, float(base), threshold)(float(base), later)
                        step = relative_step(kind, float(base), threshold)
                        rows.append(self._row(data, series_id, as_of, step, h, pool))
                        labels.append(int(hit))
                        published.append(pd.Timestamp(released))
                        as_ofs.append(as_of)
        index = pd.DatetimeIndex(as_ofs, name="as_of")
        X = pd.DataFrame(rows, index=index, dtype="float64")
        order = np.argsort(index.to_numpy(), kind="stable")
        y = pd.Series(labels, index=index, name="cleared")
        p = pd.Series(published, index=index, name="published")
        return X.iloc[order], y.iloc[order], p.iloc[order]

    # --- fitting ---------------------------------------------------------------

    def fit(self, data: BinaryData, cutoff: date) -> AIThresholdClassifier:
        """Fit one classifier per kind on what was known by the end of `cutoff`.

        A kind with fewer than `MIN_ROWS` rows is skipped (see `skipped`); one
        whose outcomes all agree gets their smoothed rate as a constant.
        """
        self.cutoff = cutoff
        self.models, self.best_params, self.cv_log_loss, self.skipped = {}, {}, {}, {}
        for kind in sorted(set(KINDS.values())):
            X, y, published = self.training_frame(data, cutoff, kind)
            if len(X) < MIN_ROWS:
                self.skipped[kind] = f"{len(X)} training rows, need {MIN_ROWS}"
                continue
            if y.nunique() < 2:
                rate = smoothed_rate(int(y.sum()), len(y))
                self.models[kind] = _KindModel(None, [], None, constant=rate)
                continue
            self.models[kind] = self._fit_kind(kind, X, y, published)
        return self

    def _fit_kind(
        self, kind: Kind, X: pd.DataFrame, y: pd.Series, published: pd.Series
    ) -> _KindModel:
        X, _ = drop_sparse(X, self.max_missing)
        X = X.loc[:, X.notna().any()]
        folds = self._folds(X.index, published, y)

        best: tuple[float, dict[str, Any], int, np.ndarray, np.ndarray] | None = None
        for values in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid, values))
            loss, iterations, oof_p, oof_y = self._cross_validate(X, y, folds, params)
            if best is None or loss < best[0]:
                best = (loss, params, iterations, oof_p, oof_y)

        calibrator = None
        if best is None:  # no usable fold: fit with the defaults, uncalibrated
            params, iterations = {}, 100
        else:
            loss, params, iterations, oof_p, oof_y = best
            self.cv_log_loss[kind] = loss
            if (
                self.calibrate
                and len(oof_y) >= MIN_CALIBRATION
                and len(np.unique(oof_y)) == 2
            ):
                calibrator = ProbabilityCalibrator("sigmoid").fit(oof_p, oof_y)
        self.best_params[kind] = {**params, "n_estimators": iterations}
        model = lgb.LGBMClassifier(
            objective="binary", **self.best_params[kind], **FIXED_PARAMS
        )
        model.fit(X, y)
        return _KindModel(model, list(X.columns), calibrator)

    def _folds(
        self, index: pd.Index, published: pd.Series, y: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Date folds, keeping only training rows whose outcome was published
        before the validation period starts; folds lacking both outcomes in
        training are dropped."""
        dates = index.to_numpy()
        out = []
        for train, valid in _date_folds(index, self.n_splits, 0):
            start = dates[valid].min()
            train = train[published.to_numpy()[train] < start]
            if len(train) and y.iloc[train].nunique() == 2:
                out.append((train, valid))
        return out

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: list[tuple[np.ndarray, np.ndarray]],
        params: dict[str, Any],
    ) -> tuple[float, int, np.ndarray, np.ndarray]:
        """Mean validation log loss, mean best iteration, out-of-fold (p, y)."""
        losses, iterations, probs, outcomes = [], [], [], []
        for train, valid in folds:
            model = lgb.LGBMClassifier(
                objective="binary",
                **params,
                n_estimators=self.max_estimators,
                **FIXED_PARAMS,
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
            p = np.asarray(model.predict_proba(X.iloc[valid], num_iteration=best))[:, 1]
            losses.append(log_loss(y.iloc[valid], p, labels=[0, 1]))
            iterations.append(best)
            probs.append(p)
            outcomes.append(y.iloc[valid].to_numpy())
        if not folds:
            return math.inf, 100, np.array([]), np.array([])
        return (
            float(np.mean(losses)),
            max(1, round(float(np.mean(iterations)))),
            np.concatenate(probs),
            np.concatenate(outcomes),
        )

    # --- predicting -------------------------------------------------------------

    def prediction_row(
        self, question: BinaryQuestion, data: BinaryData
    ) -> pd.DataFrame:
        """The feature row the question's kind model sees."""
        if self.cutoff is None:
            raise NotFittedError("call fit() first")
        if question.curated or question.horizon_months is None:
            raise NoForecast("not a threshold question")
        if question.cutoff < self.cutoff:
            raise LookaheadError(
                f"fit with data to {self.cutoff}, after the question's cutoff "
                f"{question.cutoff}"
            )
        series_id, threshold = parse_threshold_key(question.question_key)
        kind = kind_of(series_id)
        base = data.series(series_id, question.cutoff).get(_base_month(question))
        if base is None or pd.isna(base):
            raise NoForecast(f"{series_id}: base month not known by {question.cutoff}")
        pool = [s for s, k in KINDS.items() if k == kind]
        row = self._row(
            data,
            series_id,
            question.cutoff,
            relative_step(kind, float(base), threshold),
            question.horizon_months,
            pool,
        )
        return pd.DataFrame(
            [row], index=pd.DatetimeIndex([question.cutoff]), dtype="float64"
        )

    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]:
        row = self.prediction_row(question, data)
        series_id, _ = parse_threshold_key(question.question_key)
        kind = kind_of(series_id)
        if kind not in self.models:
            raise NoForecast(f"no {kind} model: {self.skipped.get(kind, 'not fit')}")
        p = self.models[kind].probability(row)
        return {"yes": p, "no": 1 - p}

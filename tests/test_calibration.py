"""Tests for interval and probability calibration.

Synthetic miscalibrated forecasts check the property calibration exists for:
after it, 80% intervals cover about 80% and the reliability curve lies
closer to y = x.
"""

from __future__ import annotations

from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import (
    Forecast,
    WalkForwardBacktest,
    add_months,
    summarize,
)
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.calibration import (
    IntervalCalibrator,
    ProbabilityCalibrator,
    calibrate_intervals,
    min_residuals,
)

RNG_SEED = 18


# Intervals


def test_conformal_quantile_hand_computed() -> None:
    # n = 9 at 80%: rank ceil(10 * 0.8) = 8, the 8th smallest |residual|.
    cal = IntervalCalibrator(0.8).fit([1, -2, 3, -4, 5, -6, 7, -8, 9])
    assert cal.predict_interval(10.0) == (2.0, 18.0)
    assert cal.n_residuals == 9


def test_signed_method_corrects_bias() -> None:
    rng = np.random.default_rng(RNG_SEED)
    residuals = rng.normal(2.0, 1.0, 2_000)  # forecasts run 2 low
    lower, upper = IntervalCalibrator(0.8, "signed").fit(residuals).predict_interval(0)
    z = NormalDist().inv_cdf(0.9)
    assert lower == pytest.approx(2.0 - z, abs=0.1)
    assert upper == pytest.approx(2.0 + z, abs=0.1)


@pytest.mark.parametrize("method", ["absolute", "signed"])
def test_calibrated_80_percent_intervals_cover_about_80_percent(method: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    validation = rng.standard_t(4, 1_000) * 3  # heavy tails, unknown scale
    test = rng.standard_t(4, 10_000) * 3
    cal = IntervalCalibrator(0.8, method).fit(validation)  # type: ignore[arg-type]
    intervals = [cal.predict_interval(0.0)] * len(test)
    assert metrics.interval_coverage(intervals, test) == pytest.approx(0.8, abs=0.03)


def test_apply_replaces_interval_and_keeps_point() -> None:
    cal = IntervalCalibrator(0.8).fit([1.0] * 10)
    assert cal.apply(Forecast(3.0, 2.9, 3.1)) == Forecast(3.0, 2.0, 4.0)


@pytest.mark.parametrize(
    ("coverage", "method", "needed"),
    [(0.8, "absolute", 4), (0.9, "absolute", 9), (0.8, "signed", 9)],
)
def test_min_residuals(coverage: float, method: str, needed: int) -> None:
    assert min_residuals(coverage, method) == needed  # type: ignore[arg-type]
    IntervalCalibrator(coverage, method).fit(range(needed))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs"):
        IntervalCalibrator(coverage, method).fit(range(needed - 1))  # type: ignore[arg-type]


def test_interval_calibrator_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="coverage"):
        IntervalCalibrator(1.0)
    with pytest.raises(ValueError, match="method"):
        IntervalCalibrator(0.8, "quantile")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NaN"):
        IntervalCalibrator(0.8).fit([1.0] * 5 + [float("nan")])
    with pytest.raises(NotFittedError):
        IntervalCalibrator(0.8).predict_interval(0.0)


# Probabilities


def overconfident(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Events happen with probability p, but the forecast pushes p toward 0/1."""
    true_p = rng.uniform(0.05, 0.95, n)
    outcomes = (rng.uniform(size=n) < true_p).astype(float)
    logit = np.log(true_p / (1 - true_p))
    forecast = 1 / (1 + np.exp(-2.5 * logit))
    return forecast, outcomes


def calibration_error(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Count-weighted mean gap between the reliability curve and y = x."""
    curve = metrics.calibration_curve(probs, outcomes)
    gap = (curve["mean_predicted"] - curve["observed_frequency"]).abs()
    return float((gap * curve["count"]).sum() / curve["count"].sum())


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_moves_curve_toward_diagonal(method: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    train_p, train_y = overconfident(5_000, rng)
    test_p, test_y = overconfident(5_000, rng)
    cal = ProbabilityCalibrator(method).fit(train_p, train_y)  # type: ignore[arg-type]
    calibrated = cal.calibrate(test_p)

    before = calibration_error(test_p, test_y)
    after = calibration_error(calibrated, test_y)
    assert before > 0.08
    assert after < 0.03
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_sigmoid_undoes_a_logit_scaling_exactly() -> None:
    # Overconfidence by a factor on the log-odds is exactly what Platt fits.
    rng = np.random.default_rng(RNG_SEED)
    train_p, train_y = overconfident(20_000, rng)
    cal = ProbabilityCalibrator("sigmoid").fit(train_p, train_y)
    true_p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    logit = np.log(true_p / (1 - true_p))
    raw = 1 / (1 + np.exp(-2.5 * logit))
    assert cal.calibrate(raw) == pytest.approx(true_p, abs=0.03)


def test_categorical_calibration_improves_brier_and_sums_to_one() -> None:
    rng = np.random.default_rng(RNG_SEED)
    categories = ["cut", "hold", "hike"]

    def draws(n: int) -> tuple[list[dict[str, float]], list[str]]:
        forecasts, outcomes = [], []
        for _ in range(n):
            true = rng.dirichlet([1, 3, 1])
            sharp = true**3 / (true**3).sum()  # overconfident
            forecasts.append(dict(zip(categories, sharp.tolist())))
            outcomes.append(str(rng.choice(categories, p=true)))
        return forecasts, outcomes

    train, train_outcomes = draws(3_000)
    test, test_outcomes = draws(3_000)
    cal = ProbabilityCalibrator().fit_categorical(train, train_outcomes)
    calibrated = [cal.calibrate_categorical(f) for f in test]

    assert all(sum(f.values()) == pytest.approx(1) for f in calibrated)
    assert all(list(f) == categories for f in calibrated)
    assert metrics.mean_brier_score(calibrated, test_outcomes) < (
        metrics.mean_brier_score(test, test_outcomes) - 0.02
    )


def test_probability_calibrator_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="method"):
        ProbabilityCalibrator("platt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="all the same"):
        ProbabilityCalibrator().fit([0.2, 0.8], [1, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        ProbabilityCalibrator().fit([0.2, 0.8], [0, 2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ProbabilityCalibrator().fit([0.2, 1.8], [0, 1])
    with pytest.raises(ValueError, match="2 probabilities but 1"):
        ProbabilityCalibrator().fit([0.2, 0.8], [0])
    with pytest.raises(NotFittedError):
        ProbabilityCalibrator().calibrate([0.5])


# Backtest integration


def noise_store() -> VintageStore:
    """Unemployment is 5 + N(0, 1) noise, published on the 10th of the next month."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    period = date(2000, 1, 1)
    while period <= date(2023, 12, 1):
        published = add_months(period, 1).replace(day=10)
        rows.append((period, published, 5.0 + float(rng.normal())))
        period = add_months(period, 1)
    return VintageStore(
        {"UNRATE": pd.DataFrame(rows, columns=["observed_at", "as_of", "value"])}
    )


class OverconfidentModel:
    """Predicts 5 with a +/- 0.1 interval: right on average, far too narrow."""

    def fit(self, data: PointInTimeData) -> None:
        pass

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        return Forecast(5.0, 4.9, 5.1)


@pytest.fixture(scope="module")
def backtest_results() -> pd.DataFrame:
    bt = WalkForwardBacktest(
        date(2008, 1, 1), date(2023, 6, 1), "annually", store=noise_store()
    )
    return bt.run(OverconfidentModel(), "unemployment", horizon_months=3)


def test_calibrate_intervals_fixes_backtest_coverage(
    backtest_results: pd.DataFrame,
) -> None:
    calibrated = calibrate_intervals(backtest_results, coverage=0.8)

    raw_coverage = summarize(backtest_results)["interval_coverage"]
    coverage = summarize(calibrated)["interval_coverage"]
    assert raw_coverage is not None and raw_coverage < 0.15
    assert coverage == pytest.approx(0.8, abs=0.08)
    # Point forecasts and the model's own bounds are untouched.
    pd.testing.assert_series_equal(
        calibrated["prediction"], backtest_results["prediction"]
    )
    assert (calibrated["raw_lower"] == 4.9).all()
    half_width = (calibrated["upper"] - calibrated["prediction"]).dropna()
    z = NormalDist().inv_cdf(0.9)
    assert half_width.median() == pytest.approx(z, abs=0.25)


def test_calibrate_intervals_uses_only_errors_published_by_each_refit(
    backtest_results: pd.DataFrame,
) -> None:
    calibrated = calibrate_intervals(backtest_results, coverage=0.8)

    # The first fold has nothing earlier to learn from.
    first = calibrated["trained_as_of"] == calibrated["trained_as_of"].iloc[0]
    assert calibrated.loc[first, "lower"].isna().all()
    assert calibrated.loc[first, "in_interval"].isna().all()
    assert (calibrated.loc[first, "n_calibration"] == 0).all()

    # Later folds: n_calibration is exactly the forecasts resolved by the refit.
    for cutoff, fold in calibrated[~first].groupby("trained_as_of"):
        resolved = backtest_results["actual_as_of"].dropna() <= cutoff
        assert (fold["n_calibration"] == resolved.sum()).all()
    # 2009 refit (cutoff 2008-12-31): the 8 forecasts made Jan-Aug 2008 target
    # Apr-Nov, published by Dec 10; September's targets Dec, published in Jan.
    second = calibrated[calibrated["trained_as_of"] == date(2008, 12, 31)]
    assert (second["n_calibration"] == 8).all()


def test_calibrate_intervals_rolling_window(backtest_results: pd.DataFrame) -> None:
    calibrated = calibrate_intervals(backtest_results, max_residuals=24)
    assert calibrated["n_calibration"].max() == 24
    with pytest.raises(ValueError, match="too small"):
        calibrate_intervals(backtest_results, max_residuals=3)

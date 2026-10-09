"""Tests for forecast scoring metrics.

Hand-computed cases check the formulas; synthetic draws check the properties
the metrics exist to measure (calibrated forecasts lie on y = x, 80% normal
intervals cover about 80%).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import matplotlib
import numpy as np
import pytest

from eco_prediction.backtest import metrics

matplotlib.use("Agg")

RNG_SEED = 13


# Point forecasts


def test_point_metrics_hand_computed() -> None:
    pred = [1.0, 2.0, 3.0, 4.0]
    act = [1.0, 1.0, 5.0, 4.0]  # errors 0, +1, -2, 0
    assert metrics.rmse(pred, act) == pytest.approx(math.sqrt(5 / 4))
    assert metrics.mae(pred, act) == pytest.approx(0.75)
    assert metrics.bias(pred, act) == pytest.approx(-0.25)


def test_perfect_point_forecasts_score_zero() -> None:
    values = [3.1, 2.9, 4.0]
    assert metrics.rmse(values, values) == 0
    assert metrics.mae(values, values) == 0


@pytest.mark.parametrize(
    ("pred", "act"),
    [([], []), ([1.0, 2.0], [1.0]), ([1.0, float("nan")], [1.0, 2.0])],
    ids=["empty", "mismatched", "nan"],
)
def test_point_metrics_reject_bad_input(pred: list[float], act: list[float]) -> None:
    with pytest.raises(ValueError):
        metrics.rmse(pred, act)


# Intervals


def test_interval_coverage_hand_computed_with_inclusive_bounds() -> None:
    intervals = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    actuals = [0.0, 1.0, 0.5, 1.5]  # both bounds count as covered
    assert metrics.interval_coverage(intervals, actuals) == 0.75


def test_80_percent_intervals_cover_about_80_percent() -> None:
    rng = np.random.default_rng(RNG_SEED)
    actuals = rng.normal(0, 1, 10_000)
    z = NormalDist().inv_cdf(0.9)
    intervals = [(-z, z)] * len(actuals)
    assert metrics.interval_coverage(intervals, actuals) == pytest.approx(
        0.8, abs=0.015
    )


def test_interval_coverage_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="lower > upper"):
        metrics.interval_coverage([(1.0, 0.0)], [0.5])


def test_interval_score_hand_computed() -> None:
    # 80% intervals, so a miss costs 2 / 0.2 = 10 per unit outside.
    intervals = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    actuals = [0.5, 1.5, -0.2]  # inside, 0.5 above, 0.2 below
    scores = [1.0, 1.0 + 10 * 0.5, 1.0 + 10 * 0.2]
    assert metrics.interval_score(intervals, actuals, 0.8) == pytest.approx(
        np.mean(scores)
    )


def test_interval_score_is_lowest_for_the_true_quantiles() -> None:
    # Too narrow and too wide both score worse than the correct 80% interval,
    # which is what coverage alone can't tell apart.
    rng = np.random.default_rng(RNG_SEED)
    actuals = rng.normal(0, 1, 20_000)
    z = NormalDist().inv_cdf(0.9)

    def score(half_width: float) -> float:
        intervals = [(-half_width, half_width)] * len(actuals)
        return metrics.interval_score(intervals, actuals, 0.8)

    assert score(z) < score(0.5 * z)
    assert score(z) < score(2.0 * z)


def test_interval_score_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="coverage"):
        metrics.interval_score([(0.0, 1.0)], [0.5], 1.0)
    with pytest.raises(ValueError, match="lower > upper"):
        metrics.interval_score([(1.0, 0.0)], [0.5], 0.8)


# Diebold-Mariano


def test_diebold_mariano_detects_a_better_forecast() -> None:
    rng = np.random.default_rng(RNG_SEED)
    good = rng.normal(0, 1.0, 300)
    bad = rng.normal(0, 1.5, 300)
    statistic, p = metrics.diebold_mariano(good, bad, horizon=1)
    assert statistic < 0 and p < 0.001
    flipped, p_flipped = metrics.diebold_mariano(bad, good, horizon=1)
    assert flipped == pytest.approx(-statistic) and p_flipped == pytest.approx(p)


def test_diebold_mariano_does_not_reject_equal_forecasts() -> None:
    rng = np.random.default_rng(RNG_SEED)
    rejections = 0
    for _ in range(200):
        a, b = rng.normal(0, 1, (2, 60))
        rejections += metrics.diebold_mariano(a, b, horizon=3)[1] < 0.05
    assert rejections / 200 < 0.10  # nominal 5%; small-sample slack


def test_diebold_mariano_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        metrics.diebold_mariano([1.0] * 5, [2.0] * 5, horizon=3)
    with pytest.raises(ValueError, match="no variance"):
        metrics.diebold_mariano([1.0] * 10, [2.0] * 10)


# Brier


def test_brier_score_hand_computed() -> None:
    probs = {"cut": 0.2, "hold": 0.7, "hike": 0.1}
    # 0.2^2 + 0.3^2 + 0.1^2
    assert metrics.brier_score(probs, "hold") == pytest.approx(0.14)
    # 0.8^2 + 0.7^2 + 0.1^2
    assert metrics.brier_score(probs, "cut") == pytest.approx(1.14)


def test_brier_score_bounds() -> None:
    assert metrics.brier_score({"cut": 0, "hold": 1, "hike": 0}, "hold") == 0
    assert metrics.brier_score({"cut": 0, "hold": 1, "hike": 0}, "cut") == 2


def test_mean_brier_score() -> None:
    probs: list[dict[str, float]] = [
        {"cut": 0, "hold": 1, "hike": 0},
        {"cut": 0.2, "hold": 0.7, "hike": 0.1},
    ]
    assert metrics.mean_brier_score(probs, ["hold", "hold"]) == pytest.approx(0.07)


@pytest.mark.parametrize(
    ("probs", "outcome", "message"),
    [
        ({"cut": 0.5, "hold": 0.6}, "hold", "sum to"),
        ({"cut": -0.1, "hold": 1.1}, "hold", r"in \[0, 1\]"),
        ({"cut": 0.5, "hold": 0.5}, "hike", "not one of"),
        ({}, "hold", "no categories"),
    ],
)
def test_brier_score_rejects_bad_forecasts(
    probs: dict[str, float], outcome: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metrics.brier_score(probs, outcome)


# Calibration


def test_calibrated_forecasts_lie_on_the_diagonal() -> None:
    rng = np.random.default_rng(RNG_SEED)
    probs = rng.uniform(0, 1, 50_000)
    outcomes = (rng.uniform(0, 1, len(probs)) < probs).astype(float)
    curve = metrics.calibration_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 10
    np.testing.assert_allclose(
        curve["observed_frequency"], curve["mean_predicted"], atol=0.02
    )
    assert curve["count"].sum() == len(probs)


def test_overconfident_forecasts_fall_off_the_diagonal() -> None:
    rng = np.random.default_rng(RNG_SEED)
    true_probs = rng.uniform(0.3, 0.7, 20_000)
    outcomes = (rng.uniform(0, 1, len(true_probs)) < true_probs).astype(float)
    stated = np.clip(0.5 + 2 * (true_probs - 0.5), 0, 1)  # pushed toward 0 and 1
    curve = metrics.calibration_curve(stated, outcomes, n_bins=5)
    top = curve.iloc[-1]
    assert top["observed_frequency"] < top["mean_predicted"] - 0.1


def test_calibration_bins_and_empty_bins() -> None:
    curve = metrics.calibration_curve([0.0, 0.05, 0.5, 1.0], [0, 0, 1, 1], n_bins=10)
    # bins [0, 0.1), [0.5, 0.6) and [0.9, 1.0] only; 1.0 lands in the last bin
    assert curve["bin_lower"].tolist() == pytest.approx([0.0, 0.5, 0.9])
    assert curve["bin_upper"].tolist() == pytest.approx([0.1, 0.6, 1.0])
    assert curve["count"].tolist() == [2, 1, 1]
    assert curve["mean_predicted"].tolist() == pytest.approx([0.025, 0.5, 1.0])
    assert curve["observed_frequency"].tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    ("probs", "outcomes", "kwargs"),
    [([1.2], [1], {}), ([0.5], [2], {}), ([0.5], [1], {"n_bins": 0})],
    ids=["prob>1", "outcome-not-binary", "no-bins"],
)
def test_calibration_rejects_bad_input(
    probs: list[float], outcomes: list[float], kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        metrics.calibration_curve(probs, outcomes, **kwargs)


def test_one_vs_rest_flattens_categorical_forecasts() -> None:
    probs, hits = metrics.one_vs_rest(
        [
            {"cut": 0.2, "hold": 0.7, "hike": 0.1},
            {"cut": 0.0, "hold": 0.4, "hike": 0.6},
        ],
        ["hold", "hike"],
    )
    assert probs.tolist() == [0.2, 0.7, 0.1, 0.0, 0.4, 0.6]
    assert hits.tolist() == [0, 1, 0, 0, 0, 1]


def test_plot_calibration_curve_overlays_models() -> None:
    curve = metrics.calibration_curve([0.1, 0.5, 0.9], [0, 1, 1], n_bins=5)
    ax = metrics.plot_calibration_curve(curve, label="a")
    same = metrics.plot_calibration_curve(curve, ax=ax, label="b")
    assert same is ax
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["perfect", "a", "b"]
    assert ax.get_xlabel() == "predicted probability"

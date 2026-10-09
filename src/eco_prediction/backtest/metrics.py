"""Scores for point, interval and probability forecasts.

Usage:
    from eco_prediction.backtest import metrics

    metrics.rmse(results["prediction"], results["actual"])
    metrics.interval_coverage(zip(results["lower"], results["upper"]), results["actual"])
    metrics.interval_score(zip(results["lower"], results["upper"]), results["actual"], 0.8)
    metrics.diebold_mariano(model_errors, random_walk_errors, horizon=3)  # (stat, p)
    metrics.brier_score({"cut": 0.2, "hold": 0.7, "hike": 0.1}, "hold")  # 0.14

    probs, hits = metrics.one_vs_rest(fomc_probabilities, fomc_outcomes)
    curve = metrics.calibration_curve(probs, hits)
    metrics.plot_calibration_curve(curve, label="fomc model")

The functions take plain sequences, not harness results, so they also score
live forecasts. They raise ValueError on empty or mismatched input and on NaN:
filter out unresolved forecasts first (`summarize` keeps rows with an actual).

`brier_score` is the multi-category form, the sum of squared differences over
every category, so it runs from 0 (all probability on the outcome) to 2 (all
on a wrong category). A binary event is two categories, e.g.
{"yes": p, "no": 1 - p}.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Tolerance for class probabilities summing to 1.
PROBABILITY_TOLERANCE = 1e-6


def _paired(
    predictions: Iterable[float], actuals: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(list(predictions), dtype=float)
    act = np.asarray(list(actuals), dtype=float)
    if len(pred) != len(act):
        raise ValueError(f"{len(pred)} predictions but {len(act)} actuals")
    if len(pred) == 0:
        raise ValueError("nothing to score")
    if np.isnan(pred).any() or np.isnan(act).any():
        raise ValueError("NaN in input; drop unresolved forecasts first")
    return pred, act


def rmse(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    pred, act = _paired(predictions, actuals)
    return math.sqrt(np.mean((pred - act) ** 2))


def mae(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    pred, act = _paired(predictions, actuals)
    return float(np.mean(np.abs(pred - act)))


def bias(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    """Mean of prediction - actual; positive means forecasts run high."""
    pred, act = _paired(predictions, actuals)
    return float(np.mean(pred - act))


def interval_coverage(
    intervals: Iterable[tuple[float, float]], actuals: Iterable[float]
) -> float:
    """Fraction of actuals inside their [lower, upper] interval, bounds included."""
    bounds = np.asarray(list(intervals), dtype=float).reshape(-1, 2)
    lower, upper = bounds[:, 0], bounds[:, 1]
    _, act = _paired(lower, actuals)
    if np.isnan(upper).any():
        raise ValueError("NaN in input; drop unresolved forecasts first")
    if (lower > upper).any():
        raise ValueError("an interval has lower > upper")
    return float(np.mean((lower <= act) & (act <= upper)))


def interval_score(
    intervals: Iterable[tuple[float, float]],
    actuals: Iterable[float],
    coverage: float,
) -> float:
    """Mean interval (Winkler) score of central `coverage` intervals; lower is better.

    Width, plus 2 / (1 - coverage) times the distance by which the actual falls
    outside. It rewards narrow intervals and punishes misses, so unlike
    coverage it can't be gamed by widening. Gneiting & Raftery (2007).
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    bounds = np.asarray(list(intervals), dtype=float).reshape(-1, 2)
    lower, upper = bounds[:, 0], bounds[:, 1]
    _, act = _paired(lower, actuals)
    if np.isnan(upper).any():
        raise ValueError("NaN in input; drop unresolved forecasts first")
    if (lower > upper).any():
        raise ValueError("an interval has lower > upper")
    penalty = 2 / (1 - coverage)
    below = np.maximum(lower - act, 0)
    above = np.maximum(act - upper, 0)
    return float(np.mean(upper - lower + penalty * (below + above)))


def diebold_mariano(
    errors_a: Iterable[float], errors_b: Iterable[float], horizon: int = 1
) -> tuple[float, float]:
    """Test of equal squared-error accuracy: (statistic, two-sided p-value).

    Negative statistics mean `errors_a` are smaller. The loss differential's
    variance allows for autocorrelation up to lag horizon - 1 (forecasts
    `horizon` steps ahead overlap), and the Harvey, Leybourne & Newbold (1997)
    small-sample correction is applied, with a t distribution on n - 1
    degrees of freedom.
    """
    from scipy import stats  # type: ignore[import-untyped]

    a, b = _paired(errors_a, errors_b)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    d = a**2 - b**2
    n = len(d)
    if n < 2 * horizon:
        raise ValueError(f"need at least {2 * horizon} forecasts, got {n}")
    centred = d - d.mean()
    variance = np.mean(centred**2)
    for lag in range(1, horizon):
        variance += 2 * np.mean(centred[lag:] * centred[:-lag])
    if variance <= 0:
        raise ValueError("loss differential has no variance")
    statistic = d.mean() / math.sqrt(variance / n)
    statistic *= math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    p_value = 2 * float(stats.t.sf(abs(statistic), df=n - 1))
    return float(statistic), p_value


def _check_probabilities(probabilities: Mapping[str, float]) -> None:
    if not probabilities:
        raise ValueError("no categories")
    if any(not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError(f"probabilities must be in [0, 1]: {dict(probabilities)}")
    total = sum(probabilities.values())
    if abs(total - 1) > PROBABILITY_TOLERANCE:
        raise ValueError(f"probabilities sum to {total}, not 1")


def brier_score(probabilities: Mapping[str, float], outcome: str) -> float:
    """Squared error of one categorical forecast, summed over categories."""
    _check_probabilities(probabilities)
    if outcome not in probabilities:
        raise ValueError(f"outcome {outcome!r} is not one of {list(probabilities)}")
    return sum(
        (p - (1.0 if category == outcome else 0.0)) ** 2
        for category, p in probabilities.items()
    )


def mean_brier_score(
    probabilities: Sequence[Mapping[str, float]], outcomes: Sequence[str]
) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError(f"{len(probabilities)} forecasts but {len(outcomes)} outcomes")
    if not probabilities:
        raise ValueError("nothing to score")
    return float(np.mean([brier_score(p, o) for p, o in zip(probabilities, outcomes)]))


def one_vs_rest(
    probabilities: Sequence[Mapping[str, float]], outcomes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten categorical forecasts into (probability, happened) pairs.

    Each category of each forecast becomes one binary prediction, so a set of
    cut/hold/hike forecasts can go through `calibration_curve`.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError(f"{len(probabilities)} forecasts but {len(outcomes)} outcomes")
    probs: list[float] = []
    hits: list[float] = []
    for forecast, outcome in zip(probabilities, outcomes):
        _check_probabilities(forecast)
        if outcome not in forecast:
            raise ValueError(f"outcome {outcome!r} is not one of {list(forecast)}")
        for category, p in forecast.items():
            probs.append(p)
            hits.append(1.0 if category == outcome else 0.0)
    return np.asarray(probs), np.asarray(hits)


def calibration_curve(
    probabilities: Iterable[float], outcomes: Iterable[float], n_bins: int = 10
) -> pd.DataFrame:
    """Predicted probability against observed frequency, in equal-width bins.

    `outcomes` are 1 if the event happened, else 0. One row per non-empty bin:
    bin_lower, bin_upper, mean_predicted, observed_frequency, count. A
    calibrated forecaster has mean_predicted close to observed_frequency.
    A probability equal to a bin edge goes in the higher bin (1.0 in the last).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    probs, hits = _paired(probabilities, outcomes)
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(hits, (0, 1)).all():
        raise ValueError("outcomes must be 0 or 1")

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], probs, side="right")
    frame = pd.DataFrame({"bin": bins, "predicted": probs, "observed": hits})
    grouped = frame.groupby("bin").agg(
        mean_predicted=("predicted", "mean"),
        observed_frequency=("observed", "mean"),
        count=("observed", "size"),
    )
    grouped.insert(0, "bin_lower", edges[grouped.index])
    grouped.insert(1, "bin_upper", edges[grouped.index + 1])
    return grouped.reset_index(drop=True)


def plot_calibration_curve(
    curve: pd.DataFrame, ax: Axes | None = None, label: str | None = None
) -> Axes:
    """Reliability diagram of a `calibration_curve` result, with the y = x line.

    Marker area grows with the number of forecasts in the bin. Pass the same
    `ax` again to compare models on one plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_aspect("equal")
    sizes = 20 + 180 * curve["count"] / max(curve["count"].max(), 1)
    (line,) = ax.plot(curve["mean_predicted"], curve["observed_frequency"], label=label)
    ax.scatter(
        curve["mean_predicted"],
        curve["observed_frequency"],
        s=sizes,
        color=line.get_color(),
    )
    if label is not None:
        ax.legend()
    return ax

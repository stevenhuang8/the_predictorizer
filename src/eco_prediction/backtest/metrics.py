"""Scores for point, interval and probability forecasts.

Usage:
    from eco_prediction.backtest import metrics

    metrics.rmse(results["prediction"], results["actual"])
    metrics.interval_coverage(zip(results["lower"], results["upper"]), results["actual"])
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

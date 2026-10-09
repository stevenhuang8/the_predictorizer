"""Why did a forecast miss? Automatic post-mortems for scored forecasts.

Usage:
    uv run python -m eco_prediction.postmortem.classifier            # classify, list misses
    uv run python -m eco_prediction.postmortem.classifier --all      # list every post-mortem
    uv run python -m eco_prediction.postmortem.classifier --set 42 regime_change --notes "COVID"

    from eco_prediction.postmortem.classifier import run_postmortems
    run_postmortems(conn, store, as_of=date.today())    # PostmortemResult(...)

The daily forecast job runs this after resolving and scoring. Every scored
forecast gets one category; automatic ones are refreshed on each run (a later
revision can turn a miss into bad data) and a manual one (`--set`) is never
overwritten.

Numeric forecasts (CPI YoY, unemployment), first match wins:
1. expected_variance - |error| <= EXPECTED_BAND (1.5) x the interval's
   half-width. For an 80% normal interval that is about a 95% band: calibrated
   intervals miss 1 time in 5, so a near miss is not a failure.
2. bad_data - against the target's latest vintage the error would be inside
   that band, and the revision took away at least BAD_DATA_SHARE (25%) of the
   error: the first release the forecast was scored on was later revised
   toward it, materially. (With no interval: at least half the error.)
3. regime_change - the actual move from the last value known at the forecast
   exceeded the REGIME_QUANTILE (99th percentile) of every same-length move in
   the history known then: a move the past gave no precedent for.
4. bad_model - anything else: a normal-sized move, no revision to blame.

FOMC forecasts:
1. expected_variance - the outcome got at least FOMC_EXPECTED_PROB (20%).
2. bad_model - it got less, but the 3-month bill had priced it: bill minus
   target at least PRICED_SPREAD (0.10 pp) above for a hike, below for a cut,
   within it for a hold. The model ignored a strong, available signal.
3. regime_change - it got less and markets hadn't priced it either: a surprise.
(Decisions aren't revised, so bad_data never applies; with no bill data the
call is bad_model.)

`revised_data_impact` = |error vs first release| - |error vs latest value|:
how much of the error later revisions took away. `evidence` stores every
number a call was based on.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import actual_value, months_between
from eco_prediction.db.postmortems import (
    MISS_CATEGORIES,
    ForecastToReview,
    MissCategory,
    forecasts_to_review,
    list_postmortems,
    save_postmortem,
)
from eco_prediction.models.fomc_model import rate_features

log = logging.getLogger("eco_prediction.postmortem")

EXPECTED_BAND = 1.5
# A revision must take away at least this share of the error to be the cause
# (half, when there's no interval band to check against).
BAD_DATA_SHARE = 0.25
REGIME_QUANTILE = 0.99
MIN_HISTORY_MOVES = 36
FOMC_EXPECTED_PROB = 0.2
PRICED_SPREAD = 0.10


@dataclass(frozen=True)
class Classification:
    category: MissCategory
    notes: str
    revised_data_impact: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def classify_numeric(
    *,
    prediction: float,
    lower: float | None,
    upper: float | None,
    actual_first: float,
    actual_latest: float | None,
    last_known: float | None,
    historical_moves: np.ndarray,
) -> Classification:
    """Categorize one numeric forecast; see the module docstring for the rules."""
    latest = actual_first if actual_latest is None else actual_latest
    error_first = prediction - actual_first
    error_latest = prediction - latest
    impact = abs(error_first) - abs(error_latest)
    half_width = None if lower is None or upper is None else (upper - lower) / 2
    band = None if half_width is None else EXPECTED_BAND * half_width
    move = None if last_known is None else actual_first - last_known
    moves = np.abs(historical_moves[~np.isnan(historical_moves)])
    threshold = (
        float(np.quantile(moves, REGIME_QUANTILE))
        if len(moves) >= MIN_HISTORY_MOVES
        else None
    )
    evidence = {
        "error_vs_first_release": error_first,
        "error_vs_latest": error_latest,
        "actual_first_release": actual_first,
        "actual_latest": latest,
        "interval_half_width": half_width,
        "expected_band": band,
        "move_from_last_known": move,
        "regime_threshold": threshold,
        "history_moves": len(moves),
    }

    def result(category: MissCategory, notes: str) -> Classification:
        return Classification(category, notes, impact, evidence)

    if band is not None and abs(error_first) <= band:
        return result(
            "expected_variance",
            f"Error {error_first:+.2f} within {EXPECTED_BAND}x the interval "
            f"half-width ({band:.2f}).",
        )
    revised = latest != actual_first
    if revised and (
        (
            band is not None
            and abs(error_latest) <= band
            and impact >= BAD_DATA_SHARE * abs(error_first)
        )
        or (band is None and impact >= abs(error_first) / 2)
    ):
        return result(
            "bad_data",
            f"First release {actual_first:.2f} revised to {latest:.2f}; against the "
            f"revision the error is {error_latest:+.2f}.",
        )
    if move is not None and threshold is not None and abs(move) > threshold:
        return result(
            "regime_change",
            f"Move of {move:+.2f} from the last known value exceeds the "
            f"{REGIME_QUANTILE:.0%} historical move ({threshold:.2f}).",
        )
    return result(
        "bad_model",
        f"Error {error_first:+.2f} on a move of "
        + ("unknown size" if move is None else f"{move:+.2f}")
        + (
            ""
            if threshold is None
            else f", within historical norms ({REGIME_QUANTILE:.0%}: {threshold:.2f})"
        )
        + ".",
    )


def classify_fomc(
    probabilities: dict[str, float], outcome: str, tbill_spread: float | None
) -> Classification:
    """Categorize one FOMC forecast; see the module docstring for the rules."""
    p = float(probabilities.get(outcome, 0.0))
    spread = None if tbill_spread is None or math.isnan(tbill_spread) else tbill_spread
    evidence = {"p_outcome": p, "outcome": outcome, "tbill_spread": spread}
    if p >= FOMC_EXPECTED_PROB:
        return Classification(
            "expected_variance",
            f"{outcome} had {p:.0%} probability (>= {FOMC_EXPECTED_PROB:.0%}).",
            evidence=evidence,
        )
    if spread is None:
        return Classification(
            "bad_model",
            f"{outcome} had only {p:.0%}; no bill data to tell if markets priced it.",
            evidence=evidence,
        )
    priced = {
        "hike": spread >= PRICED_SPREAD,
        "cut": spread <= -PRICED_SPREAD,
        "hold": abs(spread) < PRICED_SPREAD,
    }[outcome]
    if priced:
        return Classification(
            "bad_model",
            f"{outcome} had only {p:.0%} though the bill spread ({spread:+.2f}) "
            "had priced it.",
            evidence=evidence,
        )
    return Classification(
        "regime_change",
        f"{outcome} had only {p:.0%} and the bill spread ({spread:+.2f}) hadn't "
        "priced it either: a surprise.",
        evidence=evidence,
    )


def numeric_inputs(store: VintageStore, f: ForecastToReview) -> dict[str, Any]:
    """The latest value, last known value and past moves for a numeric forecast."""
    target = TARGETS[f.target]
    latest, _ = actual_value(store, target, f.target_date, "latest")
    cutoff = f.forecast_date - timedelta(days=1)
    history = PointInTimeData(store, target, cutoff).target_history().dropna()
    if history.empty:
        return {
            "actual_latest": latest,
            "last_known": None,
            "historical_moves": np.array([]),
        }
    monthly = history.asfreq("MS")
    steps = max(1, months_between(history.index[-1].date(), f.target_date))
    moves = (monthly - monthly.shift(steps)).to_numpy(dtype=float)
    return {
        "actual_latest": latest,
        "last_known": float(history.iloc[-1]),
        "historical_moves": moves,
    }


def classify(store: VintageStore, f: ForecastToReview) -> Classification:
    if f.target == "fomc_decision":
        assert f.probabilities is not None and f.actual_outcome is not None
        cutoff = f.forecast_date - timedelta(days=1)
        spread = rate_features(store, f.target_date, cutoff)["tbill_spread"]
        return classify_fomc(f.probabilities, f.actual_outcome, spread)
    assert f.prediction is not None and f.actual_value is not None
    return classify_numeric(
        prediction=f.prediction,
        lower=f.interval_lower,
        upper=f.interval_upper,
        actual_first=f.actual_value,
        **numeric_inputs(store, f),
    )


@dataclass
class PostmortemResult:
    classified: int = 0
    by_category: Counter[str] = field(default_factory=Counter)


def run_postmortems(
    conn: Connection, store: VintageStore, as_of: date
) -> PostmortemResult:
    """Classify (or reclassify) every scored forecast not classified by hand."""
    store = store.until(as_of)
    result = PostmortemResult()
    for forecast in forecasts_to_review(conn):
        c = classify(store, forecast)
        save_postmortem(
            conn,
            forecast.forecast_id,
            c.category,
            notes=c.notes,
            revised_data_impact=c.revised_data_impact,
            evidence=c.evidence,
        )
        result.classified += 1
        result.by_category[c.category] += 1
    conn.commit()
    log.info(
        "Post-mortems: %d classified %s",
        result.classified,
        dict(result.by_category) or "",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    from eco_prediction.data.fomc import RELEASE_LAGS
    from eco_prediction.db.connection import ConfigError, connection
    from eco_prediction.scheduler.forecast_job import setup_logging, today

    parser = argparse.ArgumentParser(description="Classify why forecasts missed.")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("FORECAST_ID", "CATEGORY"),
        help=f"classify one forecast by hand; CATEGORY is one of {', '.join(MISS_CATEGORIES)}",
    )
    parser.add_argument("--notes", help="notes for --set")
    parser.add_argument("--all", action="store_true", help="list expected variance too")
    args = parser.parse_args(argv)
    setup_logging(None)

    try:
        with connection() as conn:
            if args.set:
                forecast_id, category = int(args.set[0]), args.set[1]
                if category not in MISS_CATEGORIES:
                    parser.error(
                        f"CATEGORY must be one of {', '.join(MISS_CATEGORIES)}"
                    )
                save_postmortem(
                    conn,
                    forecast_id,
                    category,  # type: ignore[arg-type]
                    notes=args.notes,
                    classified_by="manual",
                )
                conn.commit()
                print(f"Forecast {forecast_id}: {category} (manual)")
                return 0
            store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)
            run_postmortems(conn, store, today())
            frame = list_postmortems(conn)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    if not args.all:
        frame = frame[frame["category"] != "expected_variance"]
    if frame.empty:
        print("No post-mortems to show.")
    else:
        with pd.option_context("display.width", 200, "display.max_colwidth", 90):
            print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

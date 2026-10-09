"""Resolve questions whose answers are out, score their forecasts, report.

Usage:
    uv run python -m eco_prediction.scheduler.resolution_job              # refresh, resolve, score
    uv run python -m eco_prediction.scheduler.resolution_job --no-ingest
    uv run python -m eco_prediction.scheduler.resolution_job --include-backtest

The daily forecast job (`forecast_job`) runs this after its data refresh, so
the one cron line covers both; this module is for running it by hand.

A question resolves once its answer has been published by the run date:
- CPI YoY and unemployment: the target month's **first release**, as in the
  backtest harness (`harness.actual_value`), so live and backtest scores are
  comparable; later revisions don't change a resolution (`actual_as_of`
  records the vintage, for Task 20's post-mortems);
- FOMC: the decision, once the target the day after the meeting is published.

Every forecast of a resolved question is then scored, live and backtest:
`error` (prediction - actual) and `in_interval` for numeric forecasts,
`brier_score` (multi-category, 0-2) for FOMC probabilities. Forecasts added
to an already resolved question are scored on the next run. Runs are safe to
repeat: resolved questions and scored forecasts are skipped.

`score_summary` aggregates the scores by target, horizon and model. Monthly
questions (`horizon_months`) and pre-meeting FOMC questions (`lead_days`) are
separate rows, so week-ahead and months-ahead accuracy are never mixed.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import TARGETS, VintageStore
from eco_prediction.backtest.harness import actual_value
from eco_prediction.data.fomc import (
    RANGE_START,
    RANGE_UPPER_SERIES,
    RELEASE_LAGS,
    TARGET_SERIES,
    decision,
)
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.resolutions import (
    OpenQuestion,
    UnscoredForecast,
    save_resolution,
    save_score,
    scored_forecasts,
    unresolved_questions,
    unscored_forecasts,
)
from eco_prediction.scheduler.forecast_job import (
    DEFAULT_LOG_FILE,
    refresh_data,
    setup_logging,
    today,
    try_lock,
    unlock,
)

log = logging.getLogger("eco_prediction.resolution_job")

DEFAULT_COVERAGE = 0.8  # for interval scores, when a model version doesn't say


@dataclass
class ResolutionResult:
    resolved: list[int] = field(default_factory=list)  # question ids
    still_open: int = 0
    scored: int = 0


@dataclass(frozen=True)
class Answer:
    actual_as_of: date
    value: float | None = None
    outcome: str | None = None


def answer(store: VintageStore, question: OpenQuestion, as_of: date) -> Answer | None:
    """The question's answer as published by as_of; None if not out yet."""
    if question.target == "fomc_decision":
        meeting = question.target_date
        outcome = decision(store, meeting, as_of)
        if outcome is None:
            return None
        day_after = meeting + timedelta(days=1)
        series = RANGE_UPPER_SERIES if day_after >= RANGE_START else TARGET_SERIES
        published = store.first_release(series, day_after)
        assert published is not None  # decision() found it
        return Answer(published, outcome=outcome)
    value, vintage = actual_value(store, TARGETS[question.target], question.target_date)
    if value is None or vintage is None:
        return None
    return Answer(vintage, value=value)


def score(forecast: UnscoredForecast) -> dict[str, float | bool | None]:
    """Score columns for one forecast against its question's resolution."""
    error, in_interval, brier = None, None, None
    if forecast.prediction is not None and forecast.actual_value is not None:
        error = forecast.prediction - forecast.actual_value
        if forecast.interval_lower is not None and forecast.interval_upper is not None:
            in_interval = (
                forecast.interval_lower
                <= forecast.actual_value
                <= forecast.interval_upper
            )
    if forecast.probabilities is not None and forecast.actual_outcome is not None:
        brier = metrics.brier_score(forecast.probabilities, forecast.actual_outcome)
    return {"error": error, "in_interval": in_interval, "brier_score": brier}


def resolve_and_score(
    conn: Connection, store: VintageStore, as_of: date
) -> ResolutionResult:
    """Resolve every question answered by as_of, then score unscored forecasts."""
    store = store.until(as_of)
    result = ResolutionResult()
    for question in unresolved_questions(conn):
        found = answer(store, question, as_of)
        if found is None:
            result.still_open += 1
            continue
        save_resolution(
            conn,
            question.id,
            actual_as_of=found.actual_as_of,
            value=found.value,
            outcome=found.outcome,
        )
        conn.commit()
        result.resolved.append(question.id)
        log.info(
            "Resolved %s %s: %s (published %s)",
            question.target,
            question.target_date,
            found.outcome if found.outcome is not None else f"{found.value:.2f}",
            found.actual_as_of,
        )
    for forecast in unscored_forecasts(conn):
        save_score(conn, forecast.id, **score(forecast))  # type: ignore[arg-type]
        result.scored += 1
    conn.commit()
    log.info(
        "Resolution: %d questions resolved, %d still open, %d forecasts scored",
        len(result.resolved),
        result.still_open,
        result.scored,
    )
    return result


def _interval_score(row: pd.Series) -> float:
    if pd.isna(row["interval_lower"]) or pd.isna(row["interval_upper"]):
        return math.nan
    params = row["parameters"] or {}
    coverage = float(params.get("coverage") or DEFAULT_COVERAGE)
    return metrics.interval_score(
        [(row["interval_lower"], row["interval_upper"])],
        [row["actual_value"]],
        coverage,
    )


def score_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Scores by target, horizon (months or lead days) and model.

    Columns: n, rmse, mae, bias, coverage, interval_score (numeric) and
    brier (FOMC); NaN where they don't apply. Backtest and live forecasts
    are separate rows when both are present.
    """
    columns = [
        "target", "horizon", "model_type", "is_backtest", "n", "rmse", "mae",
        "bias", "coverage", "interval_score", "brier",
    ]  # fmt: skip
    if scored.empty:
        return pd.DataFrame(columns=columns)
    frame = scored.copy()
    frame["horizon"] = [
        f"{int(m)} months" if pd.notna(m) else f"{int(d)} days"
        for m, d in zip(frame["horizon_months"], frame["lead_days"])
    ]
    frame["interval_score"] = frame.apply(_interval_score, axis=1)
    frame["covered"] = frame["in_interval"].astype("float64")
    groups = frame.groupby(
        ["target", "horizon", "model_type", "is_backtest"], sort=True
    )
    summary = groups.agg(
        n=("forecast_id", "size"),
        rmse=(
            "error",
            lambda e: float(np.sqrt(np.mean(e**2))) if e.notna().any() else math.nan,
        ),
        mae=("error", lambda e: float(e.abs().mean())),
        bias=("error", "mean"),
        coverage=("covered", "mean"),
        interval_score=("interval_score", "mean"),
        brier=("brier_score", "mean"),
    ).reset_index()
    return summary[columns]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve answered questions, score forecasts, print scores."
    )
    parser.add_argument("--no-ingest", action="store_true")
    parser.add_argument(
        "--include-backtest", action="store_true", help="also summarize backtests"
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)

    try:
        with connection() as conn:
            if not try_lock(conn):
                log.warning("Another forecast or resolution job is running; exiting")
                return 0
            try:
                failures = [] if args.no_ingest else refresh_data()
                store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)
                resolve_and_score(conn, store, today())
                summary = score_summary(
                    scored_forecasts(conn, include_backtest=args.include_backtest)
                )
            finally:
                unlock(conn)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    if summary.empty:
        print("No scored forecasts yet.")
    else:
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(summary.round(3).to_string(index=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

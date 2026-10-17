"""The AI resolution job: resolve answered AI questions, score their forecasts,
print the scorecard.

Usage:
    uv run python -m eco_prediction.ai.jobs.ai_resolution_job
    uv run python -m eco_prediction.ai.jobs.ai_resolution_job --include-backtest
    uv run python -m eco_prediction.ai.jobs.ai_resolution_job --dry-run     # resolve, score, roll back
    uv run eco-forecast ai-resolve [same options]

Resolution and scoring are `ai/resolutions.py` (AI task 11): numeric and
threshold questions resolve to their series' first live value for the
month, curated ones to the Polymarket market's outcome; numeric forecasts
get an error and interval hit, binary ones a Brier score. The AI forecast
job runs the same step before forecasting, so this job is for checking the
scorecard, or for running resolution on its own schedule (a launchd agent
like the forecast job's, if wanted).

It shares the AI forecast job's advisory lock, not the economy's, so it
never overlaps an AI forecast run and never waits on an economy one.

The scorecard (`ai_score_summary`):
- **numeric**, by series, horizon and model: n, RMSE, relative RMSE (error /
  actual, comparable across series), bias and 80% interval coverage;
- **binary**, by kind (threshold or curated) and model: n, Brier, and a
  calibration gap: the count-weighted mean of |predicted - observed| over
  ten probability bins (0 is perfectly calibrated).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.jobs.ai_forecast_job import try_lock, unlock
from eco_prediction.ai.resolutions import resolve_and_score_ai
from eco_prediction.backtest import metrics
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.resolutions import scored_forecasts
from eco_prediction.scheduler.forecast_job import setup_logging, today

log = logging.getLogger("eco_prediction.ai_resolution_job")

DEFAULT_LOG_FILE = Path("logs/ai_resolution_job.log")
CALIBRATION_BINS = 10


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(errors**2))) if errors.notna().any() else math.nan


def calibration_gap(probabilities: pd.Series, outcomes: pd.Series) -> float:
    """Count-weighted mean |predicted - observed| over probability bins."""
    y = (outcomes == "yes").astype(float)
    if len(y) == 0:
        return math.nan
    curve = metrics.calibration_curve(probabilities, y, n_bins=CALIBRATION_BINS)
    gap = (curve["mean_predicted"] - curve["observed_frequency"]).abs()
    return float((gap * curve["count"]).sum() / curve["count"].sum())


def scored_ai_forecasts(
    conn: Connection, *, include_backtest: bool = False
) -> pd.DataFrame:
    """`db.resolutions.scored_forecasts` for the AI domain, plus `p_yes` for
    binary forecasts (stored as {"yes": p, "no": 1 - p}, not as a prediction)."""
    scored = scored_forecasts(conn, include_backtest=include_backtest, domain="ai")
    if scored.empty:
        return scored.assign(p_yes=pd.Series(dtype="float64"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, (probabilities->>'yes')::float8 FROM forecasts WHERE id = ANY(%s)",
            (scored["forecast_id"].astype(int).tolist(),),
        )
        p_yes = dict(cur.fetchall())
    return scored.assign(p_yes=scored["forecast_id"].map(p_yes).astype("float64"))


def ai_score_summary(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{'numeric': ..., 'binary': ...} from `scored_ai_forecasts`; see the module docstring."""
    if scored.empty:
        return {"numeric": pd.DataFrame(), "binary": pd.DataFrame()}
    numeric = scored[scored["question_type"] == "numeric_interval"].copy()
    binary = scored[scored["question_type"] == "binary"].copy()
    out: dict[str, pd.DataFrame] = {}
    if numeric.empty:
        out["numeric"] = pd.DataFrame()
    else:
        numeric["relative_error"] = numeric["error"] / numeric["actual_value"]
        numeric["covered"] = numeric["in_interval"].astype("float64")
        out["numeric"] = (
            numeric.groupby(["question_key", "horizon_months", "model_type", "is_backtest"])
            .agg(
                n=("forecast_id", "size"), rmse=("error", _rmse),
                relative_rmse=("relative_error", _rmse), bias=("error", "mean"),
                coverage=("covered", "mean"),
            )
            .reset_index()
            .rename(columns={"question_key": "series"})
        )  # fmt: skip
    if binary.empty:
        out["binary"] = pd.DataFrame()
    else:
        binary["kind"] = np.where(
            binary["question_key"].str.startswith("polymarket:"), "curated", "threshold"
        )
        rows: list[dict[str, Any]] = []
        for (kind, model, backtest), g in binary.groupby(
            ["kind", "model_type", "is_backtest"]
        ):
            rows.append({
                "kind": kind, "model_type": model, "is_backtest": backtest, "n": len(g),
                "brier": float(g["brier_score"].mean()),
                "calibration_gap": calibration_gap(g["p_yes"], g["actual_outcome"]),
            })  # fmt: skip
        out["binary"] = pd.DataFrame(rows)
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    connect: Callable[[], Any] = connection,
    clock: Callable[[], date] = today,
) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve AI questions and score their forecasts."
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="resolve with what was known by this date (default today)",
    )
    parser.add_argument(
        "--include-backtest", action="store_true", help="also score backtests"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and score, print, then roll everything back",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args(argv)  # fmt: skip
    setup_logging(args.log_file)
    as_of = args.date or clock()
    try:
        with connect() as conn:
            if not try_lock(conn):
                log.warning("An AI forecast or resolution job is running; exiting")
                return 0
            try:
                result = resolve_and_score_ai(conn, as_of)
                log.info("Resolved %d AI questions (%d still open), scored %d forecasts",
                         result.resolved, result.still_open, result.scored)  # fmt: skip
                summary = ai_score_summary(
                    scored_ai_forecasts(conn, include_backtest=args.include_backtest)
                )
                if args.dry_run:
                    conn.rollback()
                    log.info("Dry run: rolled back")
                else:
                    conn.commit()
            finally:
                unlock(conn)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    if all(t.empty for t in summary.values()):
        print("No scored AI forecasts yet.")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        for name, table in summary.items():
            if not table.empty:
                print(f"\n{name}\n{table.round(4).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

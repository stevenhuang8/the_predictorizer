"""Monthly live forecasts: refresh data, create questions, run every model, store.

Usage:
    uv run python -m eco_prediction.scheduler.forecast_job                   # as of today
    uv run python -m eco_prediction.scheduler.forecast_job --once-per-month  # for cron
    uv run python -m eco_prediction.scheduler.forecast_job --no-ingest
    uv run python -m eco_prediction.scheduler.forecast_job --date 2026-09-01 # a backtest

Scheduling (cron, see NOTES/TASK22.md for why daily):

    0 9 * * * cd /path/to/eco_prediction && \\
        uv run python -m eco_prediction.scheduler.forecast_job --once-per-month

A run forecasting on date D uses the data published through D - 1, the
backtest harness's convention, and asks, for a forecast made in month M:
- CPI YoY and unemployment for months M + 1, M + 3 and M + 6 (`HORIZONS`);
- the FOMC decision at the scheduled meeting in each of those months, if any.

Every model in `DEFAULT_NUMERIC` / `DEFAULT_FOMC` forecasts every question
of its kind, baselines included, so the track record always has a reference.
Models are refit on each run. LightGBM forecasts store their SHAP values and
the exact feature row the model saw (a feature snapshot).

Runs are safe to repeat. A forecast that exists for the same question,
model version and date is skipped before its model is fit, and a unique
index (migration 005) backs that up. Each model's forecasts commit
separately, so one failing model doesn't lose the others, and a rerun fills
in only what's missing. A Postgres advisory lock keeps two runs from
overlapping.

`--once-per-month` makes daily cron runs safe: if live forecasts already
exist this month, the run reuses that month's forecast date and only fills
gaps, so a machine that was asleep on the 1st catches up the next morning
without producing a second set of forecasts.

A run for a date before today is stored with `is_backtest = true`: its data is
cut off at that date, but it was not made in real time.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecaster, add_months, month_start
from eco_prediction.data.fomc import RELEASE_LAGS, meetings_between
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.forecasts import (
    forecast_exists,
    get_or_create_model_version,
    get_or_create_question,
    save_feature_snapshot,
    save_forecast,
    save_probability_forecast,
)
from eco_prediction.features.engineering import FeatureEngineer
from eco_prediction.models.baselines import RandomWalkModel
from eco_prediction.models.explainability import SHAPExplainer
from eco_prediction.models.fomc_model import FOMCForecaster, FOMCModel, Persistence
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.models.statistical import ARIMAModel

log = logging.getLogger("eco_prediction.forecast_job")

HORIZONS = (1, 3, 6)
NUMERIC_TARGETS = ("cpi_yoy", "unemployment")
# pg_advisory_lock key held for the length of a run.
LOCK_KEY = 2_202_210
DEFAULT_LOG_FILE = Path("logs/forecast_job.log")
REPO_ROOT = Path(__file__).resolve().parents[3]
EASTERN = ZoneInfo("America/New_York")

RESOLUTION_RULES = {
    "cpi_yoy": "First release of CPI-U all items (SA, CPIAUCSL) for the target "
    "month, as a % change on the same month a year earlier as known then.",
    "unemployment": "First release of the unemployment rate (UNRATE, SA) for "
    "the target month.",
    "fomc_decision": "Fed funds target (midpoint of the range) the day after the "
    "decision vs. the day before: higher = hike, lower = cut, same = hold.",
}


@dataclass(frozen=True)
class NumericModelSpec:
    model_type: str
    build: Callable[[int], Forecaster]  # horizon in months -> unfitted model
    explain: bool = False  # store SHAP values and the feature row (LightGBM)


@dataclass(frozen=True)
class FOMCModelSpec:
    model_type: str
    build: Callable[[int], FOMCModel]  # lead in days -> unfitted model


# Shared so the models of a run reuse each other's feature rows: the cache is
# keyed by store, and each run has its own (cut-off) store.
FEATURES = FeatureEngineer()

DEFAULT_NUMERIC: tuple[NumericModelSpec, ...] = (
    NumericModelSpec("random_walk", lambda h: RandomWalkModel()),
    NumericModelSpec("arima", lambda h: ARIMAModel()),
    NumericModelSpec(
        "lightgbm", lambda h: LightGBMModel(h, features=FEATURES), explain=True
    ),
)
DEFAULT_FOMC: tuple[FOMCModelSpec, ...] = (
    FOMCModelSpec("fomc_persistence", lambda lead: Persistence(lead)),
    FOMCModelSpec(
        "fomc_lightgbm", lambda lead: FOMCForecaster(lead, features=FEATURES)
    ),
)


@dataclass(frozen=True)
class Question:
    target: str
    target_date: date
    horizon_months: int


def today() -> date:
    """Today in US Eastern time, the clock FRED and the Fed publish on."""
    return datetime.now(tz=EASTERN).date()


def plan_questions(
    forecast_date: date,
    targets: Sequence[str] = NUMERIC_TARGETS,
    horizons: Sequence[int] = HORIZONS,
    *,
    include_fomc: bool = True,
) -> list[Question]:
    """The questions a forecast made on forecast_date answers."""
    questions = []
    for h in horizons:
        target_month = add_months(month_start(forecast_date), h)
        questions += [Question(t, target_month, h) for t in targets]
        if include_fomc:
            month_end = add_months(target_month, 1) - timedelta(days=1)
            questions += [
                Question("fomc_decision", meeting, h)
                for meeting in meetings_between(target_month, month_end)
            ]
    return questions


@dataclass
class JobResult:
    forecast_date: date
    is_backtest: bool
    questions_created: int = 0
    forecasts_inserted: int = 0
    forecasts_skipped: int = 0
    failures: list[str] = field(default_factory=list)


def feature_dict(row: pd.DataFrame) -> dict[str, Any]:
    """A one-row feature frame as {feature: value}, for a feature snapshot."""
    return {str(k): v for k, v in row.iloc[0].items()}


def model_parameters(model: object, **extra: Any) -> dict[str, Any]:
    """What identifies a fitted model: its settings and any tuned parameters."""
    params = dict(extra)
    for name in ("coverage", "season_length", "train_start", "predict_change"):
        if hasattr(model, name):
            params[name] = getattr(model, name)
    params.update(getattr(model, "best_params", {}) or {})
    return params


@dataclass
class _Run:
    """One run's shared state: where to write, what data, as of when."""

    conn: Connection
    store: VintageStore  # already cut off at `cutoff`
    forecast_date: date
    is_backtest: bool
    code_hash: str | None
    result: JobResult

    @property
    def cutoff(self) -> date:
        return self.forecast_date - timedelta(days=1)

    def attempt(
        self, question_id: int, model_type: str, tag: str, make: Callable[[], str]
    ) -> None:
        """Run `make` (which inserts one forecast) unless it already exists.

        Commits on success; on failure rolls back, logs and records it.
        """
        if forecast_exists(
            self.conn,
            question_id,
            model_type,
            tag,
            self.forecast_date,
            is_backtest=self.is_backtest,
        ):
            self.result.forecasts_skipped += 1
            return
        try:
            summary = make()
        except Exception as exc:
            self.conn.rollback()
            message = f"{model_type} {tag}: {exc}"
            log.exception("Failed: %s", message)
            self.result.failures.append(message)
            return
        self.conn.commit()
        self.result.forecasts_inserted += 1
        log.info("%s %s: %s", model_type, tag, summary)

    def numeric(self, q: Question, question_id: int, spec: NumericModelSpec) -> None:
        tag = f"{q.target}-h{q.horizon_months}-{self.cutoff:%Y%m%d}"

        def make() -> str:
            view = PointInTimeData(self.store, TARGETS[q.target], self.cutoff)
            model = spec.build(q.horizon_months)
            model.fit(view)
            forecast = model.predict(view, q.target_date)
            explanation, snapshot_id = None, None
            if spec.explain:
                assert isinstance(model, LightGBMModel)
                explanation = SHAPExplainer(model).explain(view, q.target_date)
                row, _ = model.prediction_inputs(view, q.target_date)
                snapshot_id = save_feature_snapshot(
                    self.conn, self.cutoff, feature_dict(row)
                )
            version_id = get_or_create_model_version(
                self.conn,
                spec.model_type,
                tag,
                parameters=model_parameters(
                    model, target=q.target, horizon_months=q.horizon_months
                ),
                training_end=self.cutoff,
                code_hash=self.code_hash,
            )
            save_forecast(
                self.conn,
                question_id,
                version_id,
                self.forecast_date,
                forecast,
                explanation=explanation,
                feature_snapshot_id=snapshot_id,
                is_backtest=self.is_backtest,
            )
            return f"{forecast.point:.2f} [{forecast.lower}, {forecast.upper}]"

        self.attempt(question_id, spec.model_type, tag, make)

    def fomc(self, q: Question, question_id: int, spec: FOMCModelSpec) -> None:
        lead = (q.target_date - self.forecast_date).days
        tag = f"fomc-lead{lead}-{self.cutoff:%Y%m%d}"

        def make() -> str:
            model = spec.build(lead)
            model.fit(self.store, self.cutoff)
            probs = model.predict_proba(self.store, q.target_date)
            snapshot_id = None
            if isinstance(model, FOMCForecaster):
                row = model.prediction_inputs(self.store, q.target_date)
                snapshot_id = save_feature_snapshot(
                    self.conn, self.cutoff, feature_dict(row)
                )
            version_id = get_or_create_model_version(
                self.conn,
                spec.model_type,
                tag,
                parameters=model_parameters(model, lead_days=lead),
                training_start=getattr(model, "train_start", None),
                training_end=self.cutoff,
                code_hash=self.code_hash,
            )
            save_probability_forecast(
                self.conn,
                question_id,
                version_id,
                self.forecast_date,
                probs,
                feature_snapshot_id=snapshot_id,
                is_backtest=self.is_backtest,
            )
            return f"meeting {q.target_date}: " + ", ".join(
                f"{k} {v:.3f}" for k, v in probs.items()
            )

        self.attempt(question_id, spec.model_type, tag, make)


def run_forecast_job(
    conn: Connection,
    store: VintageStore,
    forecast_date: date,
    *,
    is_backtest: bool | None = None,
    numeric_models: Sequence[NumericModelSpec] = DEFAULT_NUMERIC,
    fomc_models: Sequence[FOMCModelSpec] = DEFAULT_FOMC,
    targets: Sequence[str] = NUMERIC_TARGETS,
    horizons: Sequence[int] = HORIZONS,
    include_fomc: bool = True,
    code_hash: str | None = None,
) -> JobResult:
    """Create the questions for forecast_date and store every model's forecasts.

    `is_backtest` defaults to whether forecast_date is before today.
    """
    if is_backtest is None:
        is_backtest = forecast_date < today()
    cutoff = forecast_date - timedelta(days=1)
    run = _Run(
        conn,
        store.until(cutoff),
        forecast_date,
        is_backtest,
        code_hash,
        JobResult(forecast_date, is_backtest),
    )
    log.info(
        "Forecasting as of %s (data through %s, %s)",
        forecast_date,
        cutoff,
        "backtest" if is_backtest else "live",
    )

    questions = plan_questions(
        forecast_date, targets, horizons, include_fomc=include_fomc
    )
    question_ids: list[tuple[Question, int]] = []
    for q in questions:
        question_id, created = get_or_create_question(
            conn,
            q.target,
            q.target_date,
            q.horizon_months,
            resolution_rule=RESOLUTION_RULES[q.target],
        )
        question_ids.append((q, question_id))
        run.result.questions_created += created
    conn.commit()

    for q, question_id in question_ids:
        if q.target == "fomc_decision":
            for fomc_spec in fomc_models:
                run.fomc(q, question_id, fomc_spec)
        else:
            for spec in numeric_models:
                run.numeric(q, question_id, spec)

    result = run.result
    log.info(
        "Done: %d questions created, %d forecasts inserted, %d already existed, "
        "%d failed",
        result.questions_created,
        result.forecasts_inserted,
        result.forecasts_skipped,
        len(result.failures),
    )
    return result


def month_forecast_date(conn: Connection, on: date) -> date | None:
    """The date this month's live forecasts were made on, if any were."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MIN(forecast_date) FROM forecasts
            WHERE NOT is_backtest AND forecast_date >= %s AND forecast_date <= %s
            """,
            (month_start(on), on),
        )
        row = cur.fetchone()
    return None if row is None else row[0]


def try_lock(conn: Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
        row = cur.fetchone()
    conn.commit()
    return bool(row and row[0])


def unlock(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
    conn.commit()


def code_version() -> str | None:
    """`git describe --always --dirty` of the repository, if available."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()[:64] or None


def refresh_data() -> list[str]:
    """Incrementally ingest every V1 series; returns the series that failed."""
    from eco_prediction.data.fred_client import FREDClient
    from eco_prediction.data.ingest import V1_SERIES, ingest_many

    with FREDClient.from_env() as client:
        results, failures = ingest_many(client, V1_SERIES)
    for r in results:
        log.info("Ingested %s: %d new rows", r.series_id, r.inserted)
    for series_id, exc in failures.items():
        log.error("Ingest of %s failed: %s", series_id, exc)
    return list(failures)


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate this month's forecasts.")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="forecast as of this date (default today; earlier = backtest)",
    )
    parser.add_argument(
        "--no-ingest", action="store_true", help="skip the FRED data refresh"
    )
    parser.add_argument(
        "--once-per-month",
        action="store_true",
        help="reuse this month's forecast date if live forecasts exist (for cron)",
    )
    parser.add_argument("--no-fomc", action="store_true", help="skip FOMC questions")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)

    now = today()
    forecast_date = args.date or now
    try:
        with connection() as conn:
            if not try_lock(conn):
                log.warning("Another forecast job is running; exiting")
                return 0
            try:
                if args.once_per_month and args.date is None:
                    earlier = month_forecast_date(conn, now)
                    if earlier is not None:
                        log.info(
                            "Live forecasts for this month exist (made %s); "
                            "filling any gaps",
                            earlier,
                        )
                        forecast_date = earlier
                ingest_failures = [] if args.no_ingest else refresh_data()
                result = run_forecast_job(
                    conn,
                    VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS),
                    forecast_date,
                    is_backtest=forecast_date < now and args.date is not None,
                    include_fomc=not args.no_fomc,
                    code_hash=code_version(),
                )
            finally:
                unlock(conn)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    return 1 if result.failures or ingest_failures else 0


if __name__ == "__main__":
    sys.exit(main())

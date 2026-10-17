"""The AI forecast job: refresh data, create questions, run every model, store.

Usage:
    uv run python -m eco_prediction.ai.jobs.ai_forecast_job                     # as of today
    uv run python -m eco_prediction.ai.jobs.ai_forecast_job --once-per-month    # what launchd runs
    uv run python -m eco_prediction.ai.jobs.ai_forecast_job --no-ingest --dry-run
    uv run python -m eco_prediction.ai.jobs.ai_forecast_job --llm claude-haiku-5-5 --llm-budget 2
    uv run eco-forecast ai-forecast [same options]

Scheduling: the launchd agent `deploy/launchd/com.eco-prediction.ai-forecast-job.plist`
(daily at 09:30 with --once-per-month), like the economy job's; install
instructions are in the plist.

Each run, under its own advisory lock (`LOCK_KEY`, not the economy job's, so
the two never block each other):

1. **Resolves** AI questions answered by now, scores their forecasts
   (`ai/resolutions.py`) and classifies why each missed (`ai/postmortems.py`,
   rule-based; LLM drafts are a separate opt-in command); `--no-resolve`
   skips all three.
2. **Refreshes data** with each source's live ingest (arXiv, Hugging Face,
   GitHub counts and stars, Epoch, Polymarket); `--no-ingest` skips it. A
   failing source is logged and the run carries on with what's stored.
3. **Creates questions** (AI tasks 9-10): numeric and threshold questions
   for the run's month, and curated questions whose ask date has come.
4. **Forecasts** every open question that's due, with every model of its kind:
   - numeric (`NUMERIC_MODELS`): random walk, linear trend, ARIMA, ETS and
     LightGBM (pooled across benchmarks for the short SOTA series). LightGBM
     stores its SHAP values and feature row;
   - threshold: the historical base rate and the LightGBM classifier;
   - curated: the market price and the category base rate;
   - and the Claude forecaster on all three, only with `--llm MODEL`, and then
     never spending more than `--llm-budget` (USD) in a run.

**When questions are forecast.** Numeric questions are monthly: a forecast
made on day D uses data published through D - 1, and with
`--once-per-month` a later run in the month reuses the month's first
forecast date and only fills gaps. Binary questions are forecast as of their
own ask date (a curated question `lead_days` before its market closes, a
threshold question the day it was created), so a run that was missed
catches up with the same cutoff. A catch-up more than `CATCHUP_DAYS` late is
stored as a backtest, as is everything with `--date` in the past.

**A model that can't fit isn't a failure.** ARIMA needs 24 months, so it
can't forecast ARC-AGI-2 until 2027, and a market with no price yet has no
market forecast. Those are counted as `unavailable`. Anything else that
raises is a failure: logged, rolled back, and the exit status is 1.

Runs are safe to repeat: a forecast already stored for the same question,
model version and date is skipped, and each forecast commits on its own.
`--dry-run` does everything (but the data refresh and the LLM) in one
transaction and rolls it back, so it reports what a run would store.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import (
    AILinearTrend,
    AIRandomWalk,
    BinaryData,
    BinaryForecaster,
    BinaryQuestion,
    CuratedBaseRate,
    MarketPrice,
    NoForecast,
    ThresholdBaseRate,
)
from eco_prediction.ai.curated_questions import load_curated_questions
from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.lightgbm_model import AILightGBMModel, AIThresholdClassifier
from eco_prediction.ai.llm.backtest import BacktestBudget, BudgetExceeded
from eco_prediction.ai.llm.calls import attach_forecast
from eco_prediction.ai.llm.claude_forecaster import (
    ClaudeForecaster,
    ForecastError,
    register_prompt_version,
)
from eco_prediction.ai.llm.context import build_context_bundle, load_question
from eco_prediction.ai.postmortems import run_ai_postmortems
from eco_prediction.ai.questions import (
    generate_numeric_questions,
    generate_threshold_binary_questions,
)
from eco_prediction.ai.resolutions import resolve_and_score_ai
from eco_prediction.ai.statistical_models import AIARIMA, AIETS
from eco_prediction.ai.targets import AI_TARGETS, kind_of
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecast, Forecaster, month_start
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.forecasts import (
    forecast_exists,
    get_or_create_model_version,
    save_binary_forecast,
    save_feature_snapshot,
    save_forecast,
)
from eco_prediction.models.explainability import SHAPExplainer
from eco_prediction.scheduler.forecast_job import (
    code_version,
    feature_dict,
    model_parameters,
    setup_logging,
    today,
)

log = logging.getLogger("eco_prediction.ai_forecast_job")

LOCK_KEY = 2_202_212  # the economy job holds 2_202_210
CATCHUP_DAYS = 3  # a binary forecast later than this after its ask date is a backtest
DEFAULT_LOG_FILE = Path("logs/ai_forecast_job.log")
LLM_MODEL_TYPE = "claude_llm"


@dataclass(frozen=True)
class NumericSpec:
    model_type: str
    build: Callable[[str, int], Forecaster]  # (series id, horizon) -> unfitted model
    explain: bool = False  # LightGBM: store SHAP values and the feature row


@dataclass(frozen=True)
class BinarySpec:
    model_type: str
    build: Callable[[], BinaryForecaster]
    curated: bool  # which questions it answers: curated, or threshold


NUMERIC_MODELS: tuple[NumericSpec, ...] = (
    NumericSpec("ai_random_walk", lambda s, h: AIRandomWalk()),
    NumericSpec("ai_linear_trend", lambda s, h: AILinearTrend()),
    NumericSpec("ai_arima", lambda s, h: AIARIMA()),
    NumericSpec("ai_ets", lambda s, h: AIETS()),
    NumericSpec(
        "ai_lightgbm",
        lambda s, h: AILightGBMModel(h, pool=kind_of(s) == "percent"),
        explain=True,
    ),
)
BINARY_MODELS: tuple[BinarySpec, ...] = (
    BinarySpec("ai_market_price", MarketPrice, curated=True),
    BinarySpec("ai_base_rate", CuratedBaseRate, curated=True),
    BinarySpec("ai_base_rate", ThresholdBaseRate, curated=False),
)


@dataclass
class AIJobResult:
    forecast_date: date
    questions_created: int = 0
    forecasts_inserted: int = 0
    forecasts_skipped: int = 0
    unavailable: int = 0
    failures: list[str] = field(default_factory=list)
    llm_spent_usd: float = 0.0


@dataclass(frozen=True)
class DueBinary:
    question: BinaryQuestion
    is_backtest: bool


class _Run:
    """One run's shared state, and the store-once-commit-once step."""

    def __init__(
        self,
        conn: Connection,
        store: VintageStore,
        forecast_date: date,
        result: AIJobResult,
        *,
        dry_run: bool,
        code_hash: str | None,
    ) -> None:
        self.conn = conn
        self.store = store
        self.forecast_date = forecast_date
        self.result = result
        self.dry_run = dry_run
        self.code_hash = code_hash

    def commit(self) -> None:
        if not self.dry_run:
            self.conn.commit()

    def attempt(
        self,
        question_id: int,
        model_type: str,
        tag: str,
        forecast_date: date,
        is_backtest: bool,
        make: Callable[[], str],
    ) -> None:
        if forecast_exists(self.conn, question_id, model_type, tag, forecast_date,
                           is_backtest=is_backtest):  # fmt: skip
            self.result.forecasts_skipped += 1
            return
        try:
            summary = make()
        except (ValueError, NoForecast) as exc:  # can't forecast this one (yet)
            self._rollback()
            self.result.unavailable += 1
            log.info("%s %s q%d: unavailable (%s)", model_type, tag, question_id, exc)
            return
        except BudgetExceeded:
            self._rollback()
            raise
        except Exception as exc:
            self._rollback()
            message = f"{model_type} {tag} q{question_id}: {exc}"
            log.exception("Failed: %s", message)
            self.result.failures.append(message)
            return
        self.commit()
        self.result.forecasts_inserted += 1
        log.info("%s %s q%d: %s", model_type, tag, question_id, summary)

    def _rollback(self) -> None:
        # In a dry run everything is one transaction; a failure inside it is
        # undone with a savepoint instead of losing the run's other work.
        if self.dry_run:
            with self.conn.cursor() as cur:
                cur.execute("ROLLBACK TO SAVEPOINT forecast")
        else:
            self.conn.rollback()

    def savepoint(self) -> None:
        if self.dry_run:
            with self.conn.cursor() as cur:
                cur.execute("SAVEPOINT forecast")

    def version(self, model_type: str, tag: str, parameters: dict[str, Any]) -> int:
        return get_or_create_model_version(
            self.conn, model_type, tag, parameters=parameters,
            training_end=self.forecast_date - timedelta(days=1), code_hash=self.code_hash,
        )  # fmt: skip


# --- numeric ------------------------------------------------------------------------


def plain(f: Forecast) -> Forecast:
    """The forecast in Python floats: psycopg2 can't store numpy floats."""

    def to_float(x: float | None) -> float | None:
        return None if x is None else float(x)

    return Forecast(float(f.point), to_float(f.lower), to_float(f.upper))


def forecast_numeric(
    run: _Run,
    questions: Sequence[tuple[str, date, int, int]],
    specs: Sequence[NumericSpec],
    *,
    is_backtest: bool,
    llm: _LLM | None,
) -> None:
    """Every model on every (series, target month, horizon, question id)."""
    cutoff = run.forecast_date - timedelta(days=1)
    store = run.store.until(cutoff)
    for series_id, target, horizon, qid in questions:
        tag = f"{series_id}-h{horizon}-{cutoff:%Y%m%d}"
        for spec in specs:
            run.savepoint()

            def make(spec: NumericSpec = spec, series_id: str = series_id,
                     target: date = target, horizon: int = horizon, qid: int = qid,
                     tag: str = tag) -> str:  # fmt: skip
                view = PointInTimeData(store, AI_TARGETS[series_id], cutoff)
                model = spec.build(series_id, horizon)
                model.fit(view)
                f = plain(model.predict(view, target))
                explanation = snapshot = None
                if spec.explain:
                    assert isinstance(model, AILightGBMModel)
                    explanation = SHAPExplainer(model).explain(view, target)
                    row, _ = model.prediction_inputs(view, target)
                    snapshot = save_feature_snapshot(
                        run.conn, cutoff, feature_dict(row)
                    )
                params = model_parameters(
                    model, series_id=series_id, horizon_months=horizon
                )
                save_forecast(
                    run.conn, qid, run.version(spec.model_type, tag, params), run.forecast_date,
                    f, explanation=explanation, feature_snapshot_id=snapshot,
                    is_backtest=is_backtest,
                )  # fmt: skip
                return f"{f.point:,.2f} [{f.lower}, {f.upper}]"

            run.attempt(qid, spec.model_type, tag, run.forecast_date, is_backtest, make)
        if llm is not None:
            run.savepoint()
            llm.numeric(run, qid, tag, is_backtest)


# --- binary -------------------------------------------------------------------------


def due_binary_questions(conn: Connection, on: date) -> list[DueBinary]:
    """Open binary AI questions whose ask date has come and whose answer isn't
    known yet (a curated market may resolve before its ask date)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.question_key, q.target_date, q.horizon_months, q.lead_days,
                   q.platform_question_id, q.question_text,
                   -- The job's clock is US Eastern (scheduler.today); a question
                   -- created in the evening is a day later in UTC.
                   (q.created_at AT TIME ZONE 'America/New_York')::date,
                   p.resolved_at::date
            FROM questions q
            LEFT JOIN resolutions r ON r.question_id = q.id
            LEFT JOIN platform_questions p ON p.id = q.platform_question_id
            WHERE q.domain = 'ai' AND q.question_type = 'binary' AND r.id IS NULL
            ORDER BY q.target_date, q.id
            """
        )
        rows = cur.fetchall()
    due = []
    for qid, key, target, horizon, lead, pq, text, created, resolved_on in rows:
        ask = target - timedelta(days=lead) if lead is not None else created
        if ask > on or on >= target:
            continue
        if resolved_on is not None and resolved_on < ask:
            continue
        q = BinaryQuestion(qid, key, target, ask, horizon, pq, text)
        due.append(DueBinary(q, is_backtest=on > ask + timedelta(days=CATCHUP_DAYS)))
    return due


def forecast_binary(
    run: _Run,
    due: Sequence[DueBinary],
    specs: Sequence[BinarySpec],
    data: BinaryData,
    *,
    classifier: bool,
    llm: _LLM | None,
) -> None:
    classifiers: dict[date, AIThresholdClassifier] = {}
    for d in due:
        q = d.question
        assert q.id is not None
        tag = f"{'curated' if q.curated else 'threshold'}-{q.cutoff:%Y%m%d}"
        models: list[tuple[str, Callable[[], BinaryForecaster]]] = [
            (s.model_type, s.build) for s in specs if s.curated == q.curated
        ]
        if classifier and not q.curated:

            def fitted(cutoff: date = q.cutoff) -> BinaryForecaster:
                if cutoff not in classifiers:  # fit once per cutoff
                    classifiers[cutoff] = AIThresholdClassifier().fit(data, cutoff)
                return classifiers[cutoff]

            models.append(("ai_threshold_classifier", fitted))
        for model_type, build in models:
            run.savepoint()

            def make(
                build: Callable[[], BinaryForecaster] = build,
                model_type: str = model_type,
                q: BinaryQuestion = q,
                d: DueBinary = d,
                tag: str = tag,
            ) -> str:
                p = build().predict_probability(q, data)["yes"]
                version = run.version(
                    model_type, tag, {"question_kind": tag.split("-")[0]}
                )
                save_binary_forecast(run.conn, q.id, version, q.ask_date, p,  # type: ignore[arg-type]
                                     is_backtest=d.is_backtest)  # fmt: skip
                return f"P(yes) = {p:.3f}"

            run.attempt(q.id, model_type, tag, q.ask_date, d.is_backtest, make)
        if llm is not None:
            run.savepoint()
            llm.binary(run, q, d.is_backtest, data)


# --- the LLM (opt-in) -----------------------------------------------------------------


class _LLM:
    """The Claude forecaster inside the job: opt-in, and capped at a budget."""

    def __init__(
        self, conn: Connection, forecaster: ClaudeForecaster, budget: BacktestBudget
    ) -> None:
        self.forecaster = forecaster
        self.budget = budget
        self.version = register_prompt_version(conn, forecaster)
        with conn.cursor() as cur:  # the tag the version was registered under
            cur.execute(
                "SELECT version_tag FROM model_versions WHERE id = %s", (self.version,)
            )
            self.tag: str = cur.fetchone()[0]  # type: ignore[index]
        self.exhausted: str | None = None  # why the LLM stopped for this run

    def _attempt(self, run: _Run, qid: int, forecast_date: date, is_backtest: bool,
                 make: Callable[[], str]) -> None:  # fmt: skip
        """`run.attempt`, but a spent budget switches the LLM off for the rest of
        the run instead of stopping the run: the other models carry on."""
        if self.exhausted is not None:
            return
        try:
            run.attempt(qid, LLM_MODEL_TYPE, self.tag, forecast_date, is_backtest, make)
        except BudgetExceeded as exc:
            self.exhausted = str(exc)
            run.result.failures.append(f"llm budget: {exc}")
            log.warning("LLM budget reached (%s); no more LLM forecasts this run", exc)

    def _call(self, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self.budget.check()
        try:
            result = call()
        except ForecastError as exc:
            raise ValueError(f"LLM: {exc}") from exc  # counted as unavailable
        self.budget.record(result)
        return result

    def numeric(self, run: _Run, qid: int, tag: str, is_backtest: bool) -> None:
        def make() -> str:
            question = load_question(run.conn, qid)
            cutoff = run.forecast_date - timedelta(days=1)
            bundle = build_context_bundle(run.conn, question, cutoff, store=run.store)
            r = self._call(lambda: self.forecaster.predict_numeric(question, bundle))
            f = Forecast(r["point_forecast"], r["lower_80"], r["upper_80"])
            fid = save_forecast(run.conn, qid, self.version, run.forecast_date, f,
                                is_backtest=is_backtest, rationale=r.get("rationale"))  # fmt: skip
            if r.get("call_id"):
                attach_forecast(run.conn, [r["call_id"]], fid)
            return f"{f.point:,.2f} [{f.lower}, {f.upper}]"

        self._attempt(run, qid, run.forecast_date, is_backtest, make)

    def binary(
        self, run: _Run, q: BinaryQuestion, is_backtest: bool, data: BinaryData
    ) -> None:
        assert q.id is not None

        def make() -> str:
            question = load_question(run.conn, q.id)  # type: ignore[arg-type]
            bundle = build_context_bundle(run.conn, question, q.cutoff, store=run.store,
                                          binary_data=data)  # fmt: skip
            r = self._call(lambda: self.forecaster.predict_binary(question, bundle))
            fid = save_binary_forecast(run.conn, q.id, self.version, q.ask_date,  # type: ignore[arg-type]
                                       r["probability"], is_backtest=is_backtest,
                                       rationale=r.get("rationale"))  # fmt: skip
            if r.get("call_id"):
                attach_forecast(run.conn, [r["call_id"]], fid)
            return f"P(yes) = {r['probability']:.3f}"

        self._attempt(run, q.id, q.ask_date, is_backtest, make)


# --- the run ------------------------------------------------------------------------------


def refresh_ai_data() -> list[str]:
    """Each source's live ingest, as its own CLI would run it; returns the failures."""
    from eco_prediction.ai import (
        ingest_arxiv,
        ingest_epoch,
        ingest_github,
        ingest_huggingface,
        ingest_polymarket,
    )

    steps: list[tuple[str, Callable[[], int]]] = [
        ("arxiv", lambda: ingest_arxiv.main([])),
        ("huggingface", lambda: ingest_huggingface.main([])),
        ("github", lambda: ingest_github.main([])),
        ("github stars", lambda: ingest_github.main(["stars"])),
        ("epoch", lambda: ingest_epoch.main([])),
        ("polymarket", lambda: ingest_polymarket.main([])),
    ]
    failures = []
    for name, step in steps:
        try:
            status = step()
        except Exception:
            log.exception("Ingest of %s failed", name)
            status = 1
        if status != 0:
            failures.append(name)
        else:
            log.info("Ingested %s", name)
    return failures


def month_forecast_date(conn: Connection, on: date) -> date | None:
    """The date this month's live numeric AI forecasts were made on, if any."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MIN(f.forecast_date) FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            WHERE NOT f.is_backtest AND q.domain = 'ai'
              AND q.question_type = 'numeric_interval'
              AND f.forecast_date BETWEEN %s AND %s
            """,
            (month_start(on), on),
        )
        row = cur.fetchone()
    return None if row is None else row[0]


def run_ai_forecast_job(
    conn: Connection,
    forecast_date: date,
    *,
    is_backtest: bool = False,
    numeric_models: Sequence[NumericSpec] = NUMERIC_MODELS,
    binary_models: Sequence[BinarySpec] = BINARY_MODELS,
    classifier: bool = True,
    llm: ClaudeForecaster | None = None,
    llm_budget_usd: float | None = None,
    store: VintageStore | None = None,
    binary_data: BinaryData | None = None,
    dry_run: bool = False,
    code_hash: str | None = None,
) -> AIJobResult:
    """Create the run's questions and store every model's forecast of each.

    Binary questions are forecast as of their own ask dates (up to
    `forecast_date`); see the module docstring.
    """
    result = AIJobResult(forecast_date)
    store = store or store_from_conn(conn)
    run = _Run(conn, store, forecast_date, result, dry_run=dry_run, code_hash=code_hash)
    log.info("AI forecasts as of %s (%s)%s", forecast_date,
             "backtest" if is_backtest else "live", " [dry run]" if dry_run else "")  # fmt: skip

    numeric = generate_numeric_questions(conn, forecast_date)
    thresholds = generate_threshold_binary_questions(conn, forecast_date)
    curated = load_curated_questions(conn, forecast_date)
    result.questions_created = numeric.created + thresholds.created + curated.created
    if thresholds.missing_base:
        log.info(
            "No base month yet for thresholds on %s", ", ".join(thresholds.missing_base)
        )
    run.commit()

    budget = BacktestBudget(llm_budget_usd)
    llm_runner = _LLM(conn, llm, budget) if llm is not None else None
    numeric_questions = [
        (q.series_id, q.target_date, q.horizon_months, qid)
        for q, qid in numeric.questions
    ]
    data = binary_data or BinaryData.from_db(conn)
    # A spent LLM budget only switches the LLM off (_LLM._attempt): every other
    # model still forecasts every question.
    forecast_numeric(run, numeric_questions, numeric_models, is_backtest=is_backtest,
                     llm=llm_runner)  # fmt: skip
    forecast_binary(run, due_binary_questions(conn, forecast_date), binary_models, data,
                    classifier=classifier, llm=llm_runner)  # fmt: skip
    result.llm_spent_usd = budget.spent_usd
    log.info(
        "Done: %d questions created, %d forecasts inserted, %d already existed, "
        "%d unavailable, %d failed%s",
        result.questions_created, result.forecasts_inserted, result.forecasts_skipped,
        result.unavailable, len(result.failures),
        f", LLM ${budget.spent_usd:.2f}" if llm is not None else "",
    )  # fmt: skip
    return result


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


def main(
    argv: Sequence[str] | None = None,
    *,
    connect: Callable[[], Any] = connection,
    refresh: Callable[[], list[str]] = refresh_ai_data,
    clock: Callable[[], date] = today,
) -> int:
    parser = argparse.ArgumentParser(description="Generate the AI domain's forecasts.")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="forecast as of this date (default today; earlier = backtest)",
    )
    parser.add_argument(
        "--no-ingest", action="store_true", help="skip the data refresh"
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="skip resolving and scoring answered questions",
    )
    parser.add_argument(
        "--once-per-month",
        action="store_true",
        help="reuse this month's numeric forecast date if one exists (for the daily scheduled run)",
    )
    parser.add_argument(
        "--llm",
        metavar="MODEL",
        help="also forecast with Claude (e.g. claude-haiku-5-5); off by default",
    )
    parser.add_argument(
        "--llm-budget",
        type=float,
        metavar="USD",
        help="the most this run may spend on the LLM (required with --llm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be stored, then roll it all back (no ingest, no LLM)",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args(argv)  # fmt: skip
    if args.llm and args.llm_budget is None:
        parser.error("--llm needs --llm-budget")
    if args.llm and args.dry_run:
        parser.error("--dry-run never calls the LLM; drop --llm")
    setup_logging(args.log_file)

    now = clock()
    forecast_date = args.date or now
    try:
        with connect() as conn:
            if not try_lock(conn):
                log.warning("Another AI forecast job is running; exiting")
                return 0
            try:
                if args.once_per_month and args.date is None:
                    earlier = month_forecast_date(conn, now)
                    if earlier is not None:
                        log.info(
                            "AI forecasts for this month exist (made %s); filling gaps",
                            earlier,
                        )
                        forecast_date = earlier
                ingest_failures = [] if args.no_ingest or args.dry_run else refresh()
                if not args.no_resolve:
                    resolved = resolve_and_score_ai(conn, now)
                    log.info("Resolved %d AI questions (%d still open), scored %d forecasts",
                             resolved.resolved, resolved.still_open, resolved.scored)  # fmt: skip
                    # Post-mortem categories: rule-based, free. LLM drafts are a
                    # separate, opt-in command (ai/postmortems.py --draft).
                    pm = run_ai_postmortems(conn, now)
                    log.info("Post-mortems: %d classified %s", pm.classified,
                             dict(pm.by_category) or "")  # fmt: skip
                    if not args.dry_run:
                        conn.commit()
                llm = ClaudeForecaster(args.llm, conn=conn) if args.llm else None
                result = run_ai_forecast_job(
                    conn, forecast_date, is_backtest=args.date is not None and args.date < now,
                    llm=llm, llm_budget_usd=args.llm_budget, dry_run=args.dry_run,
                    code_hash=code_version(),
                )  # fmt: skip
                if args.dry_run:
                    conn.rollback()
                    log.info("Dry run: rolled back")
            finally:
                unlock(conn)
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    return 1 if result.failures or ingest_failures else 0


if __name__ == "__main__":
    sys.exit(main())

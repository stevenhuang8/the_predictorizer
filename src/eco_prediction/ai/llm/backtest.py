"""Leakage-safe backtest of the Claude forecaster, with the ensemble (AI task 19).

Usage:
    uv run python -m eco_prediction.ai.llm.backtest --dry-run                  # counts and cost estimate
    uv run python -m eco_prediction.ai.llm.backtest --budget 10                # Opus 5.5, stop at $10
    uv run python -m eco_prediction.ai.llm.backtest --model claude-haiku-5-5 --budget 2

    from eco_prediction.ai.llm.backtest import run_llm_backtest
    result = run_llm_backtest(conn, budget_usd=10)
    result.summary["binary"]      # Brier: llm, market price, base rate, ensemble

**The leakage rule.** A model may have read, in training, how a question
came out. So only forecasts made on or after the model's training data
cutoff (`TRAINING_CUTOFFS`: June 2026 for every current Claude model, so
from 2026-07-01) are evaluated. The model then knows nothing the forecast's
own data doesn't. Earlier forecast dates are dropped (and counted), whatever
`start_date` asks for. The statistical models are scored on the same window,
so the comparison is like for like; their own long backtests (tasks 13-15)
stay separate.

**What's evaluated.**
- **Binary:** the curated Polymarket questions asked in the window whose
  market has resolved yes or no. The outcome comes from the resolution
  table, or the market itself where the AI resolution job hasn't run yet.
  Each question gets the LLM, the market's price (Polymarket's forecast) and
  the category base rate, all as of the day before the ask date, and the
  ensemble of the three.
- **Numeric:** every AI series and horizon through the walk-forward harness
  (`model_training_cutoff`), with the LLM, random walk, ARIMA and LightGBM.
  Only targets already published can be scored, so in late 2026 that's a
  handful of 1-month forecasts.

**Cost.** Every call is logged to `llm_calls` and identical requests are
reused (Task 18), so a re-run costs only what's new. `budget_usd` stops the
run before the next call once new spending reaches it (`BudgetExceeded`; the
forecasts made so far are kept). `--dry-run` counts the forecasts and
estimates cost from prompt length without calling the API.

**Saving.** With `save=True` each binary LLM forecast is stored in
`forecasts` as a backtest (`is_backtest`) under the forecaster's model
version, linked to its call. Numeric backtest targets aren't questions in
the database, so those forecasts stay in the returned frame and in
`llm_calls`. Nothing is committed until the run (or a budget stop) ends.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import (
    AIRandomWalk,
    BinaryData,
    BinaryQuestion,
    CuratedBaseRate,
    MarketPrice,
    NoForecast,
)
from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.lightgbm_model import AILightGBMModel
from eco_prediction.ai.llm.calls import attach_forecast
from eco_prediction.ai.llm.claude_forecaster import (
    MAX_TOKENS,
    MODEL,
    PRICES,
    ClaudeForecaster,
    ForecastError,
    build_prompt,
    register_prompt_version,
)
from eco_prediction.ai.llm.context import (
    LLMQuestion,
    build_context_bundle,
    load_question,
)
from eco_prediction.ai.llm.ensemble import LLMEnsembleForecaster, Method
from eco_prediction.ai.questions import HORIZONS
from eco_prediction.ai.statistical_models import AIARIMA
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import (
    Forecast,
    Forecaster,
    WalkForwardBacktest,
    months_between,
)
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.forecasts import save_binary_forecast
from eco_prediction.models.lightgbm_model import origin

# Training data cutoffs (the month after the published one), from the models
# overview: "Training data cutoff: Jun 2026" for every current model.
TRAINING_CUTOFFS: dict[str, date] = {
    "claude-fable-5-1": date(2026, 7, 1),
    "claude-opus-5-5": date(2026, 7, 1),
    "claude-sonnet-5-5": date(2026, 7, 1),
    "claude-haiku-5-5": date(2026, 7, 1),
}
# For --dry-run estimates: output plus thinking tokens per call, a guess until
# real calls are logged (then use their mean from llm_calls).
ESTIMATED_OUTPUT_TOKENS = 3000

__all__ = [
    "TRAINING_CUTOFFS",
    "BacktestBudget",
    "BudgetExceeded",
    "LLMBacktestResult",
    "LLMNumericForecaster",
    "backtest_questions",
    "run_llm_backtest",
    "training_cutoff",
]


class BudgetExceeded(Exception):
    """The run stopped because new API spending reached its budget."""


@dataclass
class BacktestBudget:
    """New spending this run (reused calls cost nothing), and the cap."""

    limit_usd: float | None
    spent_usd: float = 0.0
    calls: int = 0
    reused: int = 0
    failures: list[str] = field(default_factory=list)

    def check(self) -> None:
        if self.limit_usd is not None and self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.2f} of a ${self.limit_usd:.2f} budget"
            )

    def record(self, result: dict[str, Any]) -> None:
        if result.get("cached"):
            self.reused += 1
            return
        self.calls += 1
        self.spent_usd += result.get("cost_usd") or 0.0


def training_cutoff(model: str) -> date:
    if model not in TRAINING_CUTOFFS:
        raise ValueError(f"no training cutoff known for {model}; pass cutoff_date")
    return TRAINING_CUTOFFS[model]


# --- binary ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuestion:
    question: BinaryQuestion
    llm_question: LLMQuestion
    resolved_at: datetime | None


def backtest_questions(
    conn: Connection, cutoff: date, start_date: date, end_date: date
) -> tuple[list[EvalQuestion], int]:
    """Curated questions asked in [max(start, cutoff), end] with a known
    outcome, and how many in [start, end] were dropped for being before the cutoff."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target_date - q.lead_days AS ask_date,
                   COALESCE(r.actual_outcome, p.resolved_outcome), p.resolved_at
            FROM questions q
            JOIN platform_questions p ON p.id = q.platform_question_id
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE q.lead_days IS NOT NULL
              AND q.target_date - q.lead_days BETWEEN %s AND %s
              AND COALESCE(r.actual_outcome, p.resolved_outcome) IN ('yes', 'no')
            ORDER BY ask_date, q.id
            """,
            (start_date, end_date),
        )
        rows = cur.fetchall()
    out, dropped = [], 0
    for qid, ask, outcome, resolved_at in rows:
        if ask < cutoff:
            dropped += 1
            continue
        lq = load_question(conn, qid)
        bq = lq.as_binary(ask)
        out.append(EvalQuestion(replace(bq, outcome=outcome), lq, resolved_at))
    return out, dropped


def _binary_rows(
    conn: Connection,
    questions: Sequence[EvalQuestion],
    forecaster: ClaudeForecaster,
    budget: BacktestBudget,
    data: BinaryData,
    store: VintageStore,
    rows: list[dict[str, Any]],
    prompt_tokens: list[int],
    *,
    save: bool,
    dry_run: bool,
) -> None:
    """Append each question's rows to `rows` as they're made, so a budget stop
    (raised out of here) keeps everything before it."""
    version = (
        register_prompt_version(conn, forecaster) if save and not dry_run else None
    )
    for eq in questions:
        q = eq.question
        base = {"question_id": q.id, "ask_date": q.ask_date, "outcome": q.outcome,
                "resolved_at": eq.resolved_at, "category": q.category}  # fmt: skip
        for name, model in (
            ("market", MarketPrice()),
            ("base_rate", CuratedBaseRate()),
        ):
            try:
                rows.append({**base, "component": name,
                             "p_yes": model.predict_probability(q, data)["yes"]})  # fmt: skip
            except NoForecast:
                pass
        bundle = build_context_bundle(
            conn, eq.llm_question, q.cutoff, store=store, binary_data=data
        )
        if dry_run:
            prompt_tokens.append(len(build_prompt(bundle)) // 4)
            continue
        budget.check()
        try:
            result = forecaster.predict_binary(eq.llm_question, bundle)
        except ForecastError as exc:
            budget.failures.append(f"question {q.id}: {exc}")
            continue
        budget.record(result)
        rows.append({**base, "component": "llm", "p_yes": result["probability"],
                     "call_id": result.get("call_id"), "cost_usd": result.get("cost_usd")})  # fmt: skip
        if save and version is not None and q.id is not None:
            forecast_id = _save_once(conn, q.id, version, q.ask_date, result)
            if forecast_id is not None and result.get("call_id"):
                attach_forecast(conn, [result["call_id"]], forecast_id)


def _save_once(
    conn: Connection, question_id: int, version: int, ask: date, result: dict[str, Any]
) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM forecasts WHERE question_id = %s AND model_version_id = %s"
            " AND forecast_date = %s AND is_backtest",
            (question_id, version, ask),
        )
        if cur.fetchone() is not None:
            return None
    return save_binary_forecast(
        conn, question_id, version, ask, result["probability"],
        is_backtest=True, rationale=result.get("rationale"),
    )  # fmt: skip


# --- numeric ------------------------------------------------------------------------------


class LLMNumericForecaster:
    """The Claude forecaster as a harness `Forecaster`: each forecast builds a
    bundle at the view's cutoff. A failed call gives a NaN point (counted in
    the budget's failures), so one refusal doesn't end the backtest, and so
    does a target not yet published, which is skipped without a call."""

    def __init__(
        self,
        conn: Connection,
        forecaster: ClaudeForecaster,
        budget: BacktestBudget,
        *,
        dry_run: bool = False,
    ) -> None:
        self.conn = conn
        self.forecaster = forecaster
        self.budget = budget
        self.dry_run = dry_run
        self.prompt_tokens: list[int] = []

    def fit(self, data: PointInTimeData) -> None:
        pass

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        series_id = data.target.series_id
        if data.store.first_release(series_id, target_period) is None:
            # Can't be scored yet: don't pay for it. (Only decides whether to
            # spend; nothing about the value reaches the forecast.)
            return Forecast(math.nan)
        horizon = months_between(origin(data.cutoff), target_period)
        q = LLMQuestion(None, "numeric_interval", series_id, target_period, horizon,
                        resolution_rule=_rule(self.conn, series_id, target_period))  # fmt: skip
        bundle = build_context_bundle(self.conn, q, data.cutoff, store=data.store)
        if self.dry_run:
            self.prompt_tokens.append(len(build_prompt(bundle)) // 4)
            return Forecast(math.nan)
        self.budget.check()
        try:
            r = self.forecaster.predict_numeric(q, bundle, horizon)
        except ForecastError as exc:
            self.budget.failures.append(f"{series_id} {target_period}: {exc}")
            return Forecast(math.nan)
        self.budget.record(r)
        return Forecast(r["point_forecast"], r["lower_80"], r["upper_80"])


def _rule(conn: Connection, series_id: str, month: date) -> str | None:
    """The resolution rule of any question on this series (they differ only by month)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolution_rule FROM questions WHERE question_key = %s"
            " AND question_type = 'numeric_interval' ORDER BY id LIMIT 1",
            (series_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    # Rules name their own month ("for 2026-11"); restate it for this one.
    return re.sub(r"for \d{4}-\d{2}", f"for {month:%Y-%m}", row[0])


def _numeric_rows(
    conn: Connection,
    store: VintageStore,
    llm: LLMNumericForecaster,
    cutoff: date,
    end_date: date,
    horizons: Sequence[int],
    series: Sequence[str],
    *,
    dry_run: bool,
) -> list[pd.DataFrame]:
    frames = []
    statistical: list[tuple[str, Callable[[int], Forecaster]]] = [
        ("random_walk", lambda h: AIRandomWalk()),
        ("arima", lambda h: AIARIMA()),
        ("lightgbm", lambda h: AILightGBMModel(h)),
    ]
    for series_id in series:
        for h in horizons:
            bt = WalkForwardBacktest(
                cutoff, end_date, "monthly", store=store, model_training_cutoff=cutoff
            )
            components: list[tuple[str, Forecaster]] = [("llm", llm)]
            if not dry_run:
                components += [(name, make(h)) for name, make in statistical]
            for name, model in components:
                try:
                    res = bt.run(model, AI_TARGETS[series_id], horizon_months=h)
                except ValueError:  # too little history for this model
                    continue
                res = res[res["actual"].notna()] if not dry_run else res
                frames.append(res.assign(series_id=series_id, component=name))
    return frames


# --- the run ------------------------------------------------------------------------------


@dataclass
class LLMBacktestResult:
    model: str
    cutoff: date
    binary: pd.DataFrame  # long: one row per question and component
    numeric: pd.DataFrame  # long: harness rows with series_id and component
    ensemble_binary: pd.DataFrame
    ensemble_numeric: pd.DataFrame
    budget: BacktestBudget
    dropped_before_cutoff: int
    summary: dict[str, Any]


def run_llm_backtest(
    conn: Connection,
    model: str = MODEL,
    cutoff_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    budget_usd: float | None = None,
    forecaster: ClaudeForecaster | None = None,
    binary: bool = True,
    numeric: bool = True,
    horizons: Sequence[int] = HORIZONS,
    series: Sequence[str] = tuple(AI_TARGETS),
    ensemble_method: Method = "learned",
    save: bool = True,
    dry_run: bool = False,
) -> LLMBacktestResult:
    """See the module docstring. `end_date` defaults to yesterday."""
    cutoff = cutoff_date or training_cutoff(model)
    start = start_date or cutoff
    end = end_date or datetime.now(UTC).date() - timedelta(days=1)
    budget = BacktestBudget(budget_usd)
    forecaster = forecaster or ClaudeForecaster(model, conn=conn)
    store = store_from_conn(conn)

    binary_rows: list[dict[str, Any]] = []
    tokens: list[int] = []
    numeric_frames: list[pd.DataFrame] = []
    dropped = 0
    llm = LLMNumericForecaster(conn, forecaster, budget, dry_run=dry_run)
    try:
        if binary:
            questions, dropped = backtest_questions(conn, cutoff, start, end)
            data = BinaryData.from_db(conn)
            _binary_rows(
                conn,
                questions,
                forecaster,
                budget,
                data,
                store,
                binary_rows,
                tokens,
                save=save,
                dry_run=dry_run,
            )
        if numeric:
            numeric_frames = _numeric_rows(
                conn,
                store,
                llm,
                max(start, cutoff),
                end,
                horizons,
                series,
                dry_run=dry_run,
            )
    except BudgetExceeded as exc:
        budget.failures.append(str(exc))
    finally:
        if not dry_run:
            conn.commit()

    binary_frame = pd.DataFrame(binary_rows)
    numeric_frame = (
        pd.concat(numeric_frames, ignore_index=True)
        if numeric_frames
        else pd.DataFrame()
    )
    ens = LLMEnsembleForecaster(ensemble_method)
    ens_binary = (
        ens.walk_forward_binary(binary_frame)
        if not dry_run and len(binary_frame)
        else pd.DataFrame()
    )
    ens_numeric = (
        ens.walk_forward_numeric(numeric_frame.dropna(subset=["prediction"]))
        if not dry_run and len(numeric_frame) else pd.DataFrame()
    )  # fmt: skip
    summary = summarize_backtest(
        binary_frame, numeric_frame, ens_binary, ens_numeric, budget,
        model=model, cutoff=cutoff, dropped=dropped,
        prompt_tokens=tokens + llm.prompt_tokens if dry_run else None,
    )  # fmt: skip
    return LLMBacktestResult(model, cutoff, binary_frame, numeric_frame, ens_binary,
                             ens_numeric, budget, dropped, summary)  # fmt: skip


def summarize_backtest(
    binary: pd.DataFrame,
    numeric: pd.DataFrame,
    ens_binary: pd.DataFrame,
    ens_numeric: pd.DataFrame,
    budget: BacktestBudget,
    *,
    model: str,
    cutoff: date,
    dropped: int,
    prompt_tokens: list[int] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model,
        "training_cutoff": cutoff.isoformat(),
        "dropped_before_cutoff": dropped,
        "cost": {
            "new_calls": budget.calls,
            "reused_calls": budget.reused,
            "spent_usd": round(budget.spent_usd, 4),
            "per_call_usd": round(budget.spent_usd / budget.calls, 4)
            if budget.calls
            else None,
            "failures": list(budget.failures),
        },
    }
    if prompt_tokens is not None:
        inp, outp, _, _ = PRICES.get(model, (math.nan,) * 4)
        n = len(prompt_tokens)
        per_call = (
            (sum(prompt_tokens) / n * inp + ESTIMATED_OUTPUT_TOKENS * outp) / 1e6
            if n
            else 0.0
        )
        out["estimate"] = {
            "llm_forecasts": n,
            "mean_prompt_tokens": round(sum(prompt_tokens) / n) if n else 0,
            "assumed_output_tokens": ESTIMATED_OUTPUT_TOKENS,
            "max_tokens": MAX_TOKENS,
            "usd_per_forecast": round(per_call, 4),
            "usd_total": round(per_call * n, 2),
        }
        return out
    if len(binary):
        out["binary"] = _brier_table(binary, ens_binary)
    if len(numeric):
        out["numeric"] = _rmse_table(numeric, ens_numeric)
    return out


def _brier_table(binary: pd.DataFrame, ens: pd.DataFrame) -> dict[str, Any]:
    """Brier per component on the questions every component answered."""
    wide = binary.pivot_table(
        index="question_id", columns="component", values="p_yes", aggfunc="first"
    )
    outcome = binary.groupby("question_id")["outcome"].first()
    if len(ens):
        wide["ensemble"] = ens.set_index("question_id")["p_yes"]
    common = wide.dropna()
    table: dict[str, Any] = {"n_questions": len(wide), "n_common": len(common)}
    y = (outcome.reindex(common.index) == "yes").astype(float)
    for c in common.columns:
        table[str(c)] = (
            round(float(((common[c] - y) ** 2).mean()), 4) if len(common) else None
        )
    return table


def _rmse_table(numeric: pd.DataFrame, ens: pd.DataFrame) -> dict[str, Any]:
    """Relative RMSE (forecast / actual - 1) per component, on common forecasts."""
    key = ["series_id", "forecast_date", "target_period"]
    wide = numeric.pivot_table(
        index=key, columns="component", values="prediction", aggfunc="first"
    )
    actual = numeric.groupby(key)["actual"].first()
    if len(ens):
        wide["ensemble"] = ens.set_index(key)["prediction"]
    common = wide.dropna()
    rel = common.div(actual.reindex(common.index), axis=0) - 1
    table: dict[str, Any] = {"n_forecasts": len(wide), "n_common": len(common)}
    for c in rel.columns:
        table[str(c)] = (
            round(float((rel[c] ** 2).mean() ** 0.5), 4) if len(rel) else None
        )
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the Claude forecaster after its training cutoff."
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--budget", type=float, help="stop once new spending reaches this many USD"
    )
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count forecasts and estimate cost; no API calls",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and args.budget is None:
        parser.error("set --budget (USD) for a run that calls the API")
    try:
        with connection() as conn:
            result = run_llm_backtest(
                conn, args.model, start_date=args.start, end_date=args.end,
                budget_usd=args.budget, binary=not args.numeric_only,
                numeric=not args.binary_only, save=not args.no_save, dry_run=args.dry_run,
            )  # fmt: skip
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

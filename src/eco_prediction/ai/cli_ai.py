"""The AI domain's side of `eco-forecast` (AI task 23): `cli.py` routes here
when a command has `--domain ai`.

    uv run eco-forecast forecast --domain ai --target arxiv --horizon 3
    uv run eco-forecast forecast --domain ai --target binary --question-id 2585
    uv run eco-forecast inspect --domain ai [--kind curated] [--open] [--limit 50]
    uv run eco-forecast scores --domain ai [--include-backtest]
    uv run eco-forecast backtest --domain ai          # the LLM comparison report

Like the economy's `forecast`, an AI forecast here only displays: it refits
the models on the data in the database and prints what they predict now,
without storing anything (the AI forecast job records live forecasts). The
LLM is never called from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import BinaryData, BinaryQuestion, NoForecast
from eco_prediction.ai.features import SERIES_PREFIX, store_from_conn
from eco_prediction.ai.jobs.ai_forecast_job import (
    BINARY_MODELS,
    NUMERIC_MODELS,
    NumericSpec,
)
from eco_prediction.ai.jobs.ai_resolution_job import (
    ai_score_summary,
    scored_ai_forecasts,
)
from eco_prediction.ai.lightgbm_model import AILightGBMModel, AIThresholdClassifier
from eco_prediction.ai.llm.context import load_question
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import add_months, month_start
from eco_prediction.models.explainability import SHAPExplainer

# --target names: the feature prefixes (arxiv, hf, github, gpqa, swebench,
# arc2, flops) for the numeric targets, and 'binary' for one binary question.
AI_TARGET_NAMES: dict[str, str] = {
    SERIES_PREFIX[s]: s for s in AI_TARGETS if s in SERIES_PREFIX
}
KINDS = ("numeric", "threshold", "curated")
SHAP_TOP = 5
LIST_LIMIT = 50


def _models(specs: Sequence[NumericSpec], only: str | None) -> list[NumericSpec]:
    chosen = [s for s in specs if only is None or s.model_type == only]
    if not chosen:
        names = ", ".join(s.model_type for s in specs)
        raise SystemExit(f"error: unknown model {only!r}; choose from {names}")
    return chosen


def _fmt(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}"


def forecast_ai_numeric(
    store: VintageStore,
    target: str,
    horizon: int,
    as_of: date,
    *,
    model: str | None = None,
    specs: Sequence[NumericSpec] = NUMERIC_MODELS,
) -> str:
    """Every AI model's forecast of `target` (a series id) for the month
    `horizon` months after as_of's month. A model that can't fit says why."""
    if horizon < 1:
        raise SystemExit("error: --horizon must be at least 1")
    cutoff = as_of - timedelta(days=1)
    period = add_months(month_start(as_of), horizon)
    view = PointInTimeData(store.until(cutoff), AI_TARGETS[target], cutoff)
    latest = view.target_history().dropna()
    lines = [
        f"{AI_TARGETS[target].description}: {period:%B %Y}, {horizon} month(s) ahead",
        (f"Data through {cutoff}; latest value {_fmt(float(latest.iloc[-1]))} "
         f"({latest.index[-1]:%B %Y})." if len(latest)
         else f"Data through {cutoff}; no values known."),
        "",
    ]  # fmt: skip
    rows: list[dict[str, object]] = []
    for spec in _models(specs, model):
        m = spec.build(target, horizon)
        try:
            m.fit(view)
            f = m.predict(view, period)
        except ValueError as exc:  # not enough history for this model
            rows.append({"model": spec.model_type, "forecast": None, "note": str(exc)})
            continue
        row: dict[str, object] = {"model": spec.model_type, "forecast": f.point,
                                  "lower": f.lower, "upper": f.upper}  # fmt: skip
        if isinstance(m, AILightGBMModel):
            explanation = SHAPExplainer(m).explain(view, period)
            row["note"] = "drivers: " + ", ".join(
                f"{k} {v:+.3f}" for k, v in explanation.top(SHAP_TOP)
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("forecast", "lower", "upper"):
        if col in table:
            table[col] = table[col].map(_fmt)
    lines.append(table.fillna("").to_string(index=False))
    return "\n".join(lines)


def forecast_ai_binary(
    conn: Connection,
    question_id: int,
    as_of: date,
    *,
    model: str | None = None,
    data: BinaryData | None = None,
) -> str:
    """Every binary model's probability for one AI question, as if asked on as_of."""
    q = load_question(conn, question_id)
    if not q.binary:
        raise SystemExit(
            f"error: question {question_id} is numeric; use --target "
            f"{SERIES_PREFIX.get(q.question_key, '...')} --horizon {q.horizon_months}"
        )
    data = data or BinaryData.from_db(conn)
    bq: BinaryQuestion = q.as_binary(as_of)
    candidates: list[tuple[str, object]] = [
        (s.model_type, s.build()) for s in BINARY_MODELS if s.curated == q.curated
    ]
    if not q.curated:
        candidates.append(("ai_threshold_classifier", None))  # fit below, if chosen
    chosen = [(name, m) for name, m in candidates if model is None or name == model]
    if not chosen:
        names = ", ".join(name for name, _ in candidates)
        raise SystemExit(f"error: unknown model {model!r}; choose from {names}")
    rows = []
    for name, m in chosen:
        try:
            if m is None:
                m = AIThresholdClassifier().fit(data, bq.cutoff)
            p = m.predict_probability(bq, data)["yes"]  # type: ignore[attr-defined]
            rows.append({"model": name, "P(yes)": f"{p:.3f}", "note": ""})
        except (NoForecast, ValueError) as exc:
            rows.append({"model": name, "P(yes)": "", "note": str(exc)})
    text = q.question_text or q.question_key
    return "\n".join([
        f"Question {question_id}: {text}",
        f"Resolves {q.target_date}; asked as of {as_of} (data through {bq.cutoff}).",
        "",
        pd.DataFrame(rows).to_string(index=False),
    ])  # fmt: skip


def list_ai_questions(
    conn: Connection,
    *,
    kind: str | None = None,
    open_only: bool = False,
    limit: int = LIST_LIMIT,
) -> str:
    """AI questions, soonest target first; `kind` is numeric, threshold or curated."""
    if kind is not None and kind not in KINDS:
        raise SystemExit(f"error: --kind must be one of {', '.join(KINDS)}")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id,
                   CASE WHEN q.question_type = 'numeric_interval' THEN 'numeric'
                        WHEN q.platform_question_id IS NOT NULL THEN 'curated'
                        ELSE 'threshold' END AS kind,
                   q.question_key, q.target_date,
                   COALESCE(q.horizon_months || ' months', q.lead_days || ' days') AS horizon,
                   count(f.id) FILTER (WHERE NOT f.is_backtest) AS live,
                   count(f.id) FILTER (WHERE f.is_backtest) AS backtest,
                   COALESCE(r.actual_value::text, r.actual_outcome, '') AS actual
            FROM questions q
            LEFT JOIN forecasts f ON f.question_id = q.id
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE q.domain = 'ai'
            GROUP BY q.id, r.actual_value, r.actual_outcome
            ORDER BY q.target_date, q.id
            """
        )
        columns = [c.name for c in cur.description or []]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    if kind is not None:
        frame = frame[frame["kind"] == kind]
    if open_only:
        frame = frame[frame["actual"] == ""]
    if frame.empty:
        return "No AI questions" + (" match." if kind or open_only else " yet.")
    shown = frame.head(limit)
    counts = frame["kind"].value_counts().to_dict()
    footer = (
        f"\n{len(shown)} of {len(frame)} shown "
        f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}); "
        "narrow with --kind / --open or raise --limit."
        if len(frame) > len(shown) else ""
    )  # fmt: skip
    return shown.to_string(index=False) + footer


def ai_scores(conn: Connection, *, include_backtest: bool = False) -> str:
    """The AI scorecard: numeric by series, horizon and model; binary by kind and model."""
    summary = ai_score_summary(
        scored_ai_forecasts(conn, include_backtest=include_backtest)
    )
    parts = [
        f"{name}\n{table.round(4).to_string(index=False)}"
        for name, table in summary.items()
        if not table.empty
    ]
    return "\n\n".join(parts) if parts else "No scored AI forecasts yet."


def ai_store(conn: Connection) -> VintageStore:
    return store_from_conn(conn)

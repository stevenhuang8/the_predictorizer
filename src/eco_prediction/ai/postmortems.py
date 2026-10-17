"""Post-mortems for AI forecasts: a rule-based category, and an optional
LLM-drafted explanation for a person to confirm (AI task 22).

Usage:
    uv run python -m eco_prediction.ai.postmortems                         # classify, list misses
    uv run python -m eco_prediction.ai.postmortems --draft claude-haiku-5-5 --budget 1
    uv run eco-forecast review-postmortems --domain ai                     # drafts to review
    uv run eco-forecast review-postmortems --domain ai --confirm 812 [--note "edited text"]
    uv run eco-forecast review-postmortems --domain ai --reject 812

    from eco_prediction.ai.postmortems import run_ai_postmortems, draft_llm_postmortem

**Categories** are the economy's four (`postmortem/classifier.py`), by the same
rules; the AI forecast job assigns them after resolving, at no cost:
- **Numeric** forecasts go through the economy's `classify_numeric` as is: within
  1.5x the interval's half-width is expected variance; a first live value
  later revised toward the forecast is bad data; a move beyond the 99th
  percentile of past same-length moves is a regime change; else bad model.
  Count series (arXiv, HF, GitHub) are judged on the log scale, where their
  growth is steady (their notes say so).
- **Binary** forecasts follow the FOMC rule: if the outcome got at least 20%
  (`EXPECTED_PROB`), expected variance. Otherwise, if a reference forecast
  made at the same time had favoured it (at least `PRICED_PROB`, 50%) the
  model missed an available signal: bad model; if the reference missed it
  too: a regime change (a surprise). The reference is the Polymarket price
  for curated questions and the series' historical base rate for threshold
  ones. Binary outcomes aren't revised, so never bad data.

**Drafts** (`draft_llm_postmortem`) are opt-in and cost money: for each miss
(not expected variance), the LLM is shown the question, the forecast (and the
model's own rationale or top SHAP drivers), what happened, the classifier's
category and reason, the series' values since the forecast, and documents
published between the forecast and the resolution (hindsight is the point
here), and asked for a 2-3 sentence explanation. Claude Haiku 5.5 by default
(`DRAFT_MODEL`), at low effort; calls are logged and reused like forecasts
(`ClaudeForecaster.complete`) and capped by `--budget`. Only live forecasts are
drafted unless `include_backtest`.

A draft supplements the category, never replaces it: it's stored in
`postmortems.draft_note` (migration 016) with `llm_drafted`, and stays
unconfirmed until a person confirms it (optionally editing it) or rejects it,
which clears it. An automatic reclassification keeps the draft.

Nothing here commits, except the CLI.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import (
    BinaryData,
    BinaryQuestion,
    NoForecast,
    ThresholdBaseRate,
)
from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.llm.backtest import BacktestBudget, BudgetExceeded
from eco_prediction.ai.llm.claude_forecaster import ClaudeForecaster, ForecastError
from eco_prediction.ai.llm.context import SERIES_QUERIES, load_question
from eco_prediction.ai.llm.retrieval import retrieve_documents
from eco_prediction.ai.resolutions import parse_threshold_key
from eco_prediction.ai.targets import AI_TARGETS, kind_of
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import actual_value, months_between
from eco_prediction.db.postmortems import save_postmortem
from eco_prediction.models.explainability import Explanation
from eco_prediction.postmortem.classifier import Classification, classify_numeric

EXPECTED_PROB = 0.2  # as FOMC_EXPECTED_PROB
PRICED_PROB = 0.5
# 1 - p loses a little precision: 1 - 0.8 is 0.19999999999999996, which must
# still count as the 20% boundary (AI_NOTES/Task22.md).
TOLERANCE = 1e-9
DRAFT_MODEL = "claude-haiku-5-5"
DRAFT_EFFORT = "low"
DRAFT_PROMPT_VERSION = "postmortem-v1"
DRAFT_MAX_CHARS = 1500
DRAFT_DOCUMENTS = {"arxiv_abstract": 3, "polymarket_question": 2, "hf_model_card": 2,
                   "github_release": 2}  # fmt: skip
NO_FALLBACK_MODELS = ("claude-haiku-5-5",)  # no server-side refusal fallback

DRAFT_SYSTEM = """\
You review forecasts of AI progress after they resolve. Explain plainly why a \
forecast missed, in 2-3 sentences, for the person maintaining the models.

Consider, in this order: data issues (a revised or late value), model \
limitations (a signal the model ignored, a trend it extrapolated too far), \
regime changes (a release or event with no precedent in the data), and \
inherent variance (a reasonable forecast that was simply unlucky). Use the \
evidence given, including what was published after the forecast, and name \
the specific event or number that mattered. If the evidence points to a \
different category than the classifier's, say so in one clause. Don't \
restate the numbers already shown unless they are the point."""

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}

__all__ = [
    "AIForecastToReview",
    "classify_ai",
    "classify_binary",
    "confirm_draft",
    "draft_llm_postmortem",
    "draft_postmortems",
    "list_drafts",
    "reject_draft",
    "run_ai_postmortems",
]


# --- what to review -------------------------------------------------------------------


@dataclass(frozen=True)
class AIForecastToReview:
    forecast_id: int
    question_id: int
    question_type: str  # 'numeric_interval' or 'binary'
    question_key: str
    target_date: date
    horizon_months: int | None
    platform_question_id: int | None
    forecast_date: date
    is_backtest: bool
    model_type: str
    prediction: float | None
    interval_lower: float | None
    interval_upper: float | None
    p_yes: float | None
    rationale: str | None
    shap_values: dict[str, Any] | None
    actual_value: float | None
    actual_outcome: str | None
    actual_as_of: date

    @property
    def series_id(self) -> str | None:
        if self.platform_question_id is not None:
            return None
        if self.question_type == "binary":
            return parse_threshold_key(self.question_key)[0]
        return self.question_key

    @property
    def cutoff(self) -> date:
        return self.forecast_date - timedelta(days=1)


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


def ai_forecasts_to_review(
    conn: Connection, forecast_id: int | None = None
) -> list[AIForecastToReview]:
    """Scored AI forecasts with no post-mortem, or only an automatic one (or the one asked for)."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT f.id, q.id, q.question_type::text, q.question_key, q.target_date,
                   q.horizon_months, q.platform_question_id, f.forecast_date, f.is_backtest,
                   m.model_type, f.prediction, f.interval_lower, f.interval_upper,
                   (f.probabilities->>'yes')::float8, f.rationale, f.shap_values,
                   r.actual_value, r.actual_outcome, r.actual_as_of
            FROM forecasts f
            JOIN questions q ON q.id = f.question_id
            JOIN resolutions r ON r.question_id = q.id
            JOIN model_versions m ON m.id = f.model_version_id
            LEFT JOIN postmortems p ON p.forecast_id = f.id
            WHERE f.scored_at IS NOT NULL AND q.domain = 'ai'
              AND {
                "f.id = %s"
                if forecast_id is not None
                else "(p.id IS NULL OR p.classified_by = 'auto')"
            }
            ORDER BY f.id
            """,
            (forecast_id,) if forecast_id is not None else (),
        )
        rows = cur.fetchall()
    return [
        AIForecastToReview(
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9],
            _f(r[10]),
            _f(r[11]),
            _f(r[12]),
            _f(r[13]),
            r[14],
            r[15],
            _f(r[16]),
            r[17],
            r[18],
        )
        for r in rows
    ]


# --- categories ---------------------------------------------------------------------------


def classify_binary(
    p_yes: float, outcome: str, reference: float | None, reference_name: str
) -> Classification:
    """The FOMC rule with a reference forecast in place of the bill spread."""
    p = p_yes if outcome == "yes" else 1 - p_yes
    ref = (
        None
        if reference is None
        else (reference if outcome == "yes" else 1 - reference)
    )
    evidence = {"p_outcome": p, "outcome": outcome, "reference": reference_name,
                "reference_p_outcome": ref}  # fmt: skip
    if p >= EXPECTED_PROB - TOLERANCE:
        return Classification("expected_variance",
                               f"'{outcome}' had {p:.0%} (>= {EXPECTED_PROB:.0%}).",
                               evidence=evidence)  # fmt: skip
    if ref is None:
        return Classification("bad_model",
                              f"'{outcome}' had only {p:.0%}; no {reference_name} to tell "
                              "whether it was foreseeable.", evidence=evidence)  # fmt: skip
    if ref >= PRICED_PROB - TOLERANCE:
        return Classification("bad_model",
                              f"'{outcome}' had only {p:.0%} though the {reference_name} "
                              f"gave it {ref:.0%}.", evidence=evidence)  # fmt: skip
    return Classification("regime_change",
                          f"'{outcome}' had only {p:.0%} and the {reference_name} gave it "
                          f"{ref:.0%}: a surprise.", evidence=evidence)  # fmt: skip


def _numeric(store: VintageStore, f: AIForecastToReview) -> Classification:
    assert f.prediction is not None and f.actual_value is not None and f.series_id
    target = AI_TARGETS[f.series_id]
    latest, _ = actual_value(store, target, f.target_date, "latest")
    history = PointInTimeData(store, target, f.cutoff).target_history().dropna()
    log = kind_of(f.series_id) == "count"

    def scale(x: float | None) -> float | None:
        if x is None:
            return None
        return math.log(x) if log and x > 0 else (math.nan if log else x)

    if history.empty:
        last, moves = None, np.array([])
    else:
        monthly = history.asfreq("MS")
        values = np.log(monthly) if log else monthly
        steps = max(1, months_between(history.index[-1].date(), f.target_date))
        moves = (values - values.shift(steps)).to_numpy(dtype=float)
        last = float(history.iloc[-1])
    c = classify_numeric(
        prediction=scale(f.prediction),  # type: ignore[arg-type]
        lower=scale(f.interval_lower), upper=scale(f.interval_upper),
        actual_first=scale(f.actual_value),  # type: ignore[arg-type]
        actual_latest=scale(latest), last_known=scale(last), historical_moves=moves,
    )  # fmt: skip
    if not log:
        return c
    return Classification(c.category, f"On the log scale: {c.notes}", c.revised_data_impact,
                          {**c.evidence, "scale": "log"})  # fmt: skip


def classify_ai(
    store: VintageStore, data: BinaryData, f: AIForecastToReview
) -> Classification:
    if f.question_type != "binary":
        return _numeric(store, f)
    assert f.p_yes is not None and f.actual_outcome is not None
    q = BinaryQuestion(f.question_id, f.question_key, f.target_date, f.forecast_date,
                       f.horizon_months, f.platform_question_id)  # fmt: skip
    if f.platform_question_id is not None:
        return classify_binary(f.p_yes, f.actual_outcome,
                               data.price(f.platform_question_id, q.cutoff), "market price")  # fmt: skip
    try:
        base = ThresholdBaseRate().predict_probability(q, data)["yes"]
    except NoForecast:
        base = None
    return classify_binary(f.p_yes, f.actual_outcome, base, "historical base rate")


@dataclass
class AIPostmortemResult:
    classified: int = 0
    by_category: Counter[str] = field(default_factory=Counter)
    drafted: int = 0
    failures: list[str] = field(default_factory=list)


def run_ai_postmortems(
    conn: Connection,
    as_of: date,
    *,
    store: VintageStore | None = None,
    data: BinaryData | None = None,
) -> AIPostmortemResult:
    """Classify (or reclassify) every scored AI forecast not classified by hand.
    Drafts already written are kept. No commit."""
    store = (store or store_from_conn(conn)).until(as_of)
    data = data or BinaryData.from_db(conn)
    result = AIPostmortemResult()
    for f in ai_forecasts_to_review(conn):
        c = classify_ai(store, data, f)
        save_postmortem(conn, f.forecast_id, c.category, notes=c.notes,
                        revised_data_impact=c.revised_data_impact, evidence=c.evidence)  # fmt: skip
        result.classified += 1
        result.by_category[c.category] += 1
    return result


# --- drafts --------------------------------------------------------------------------------


def make_drafter(
    conn: Connection | None, model: str = DRAFT_MODEL, **kwargs: Any
) -> ClaudeForecaster:
    """A Claude client set up for drafts: low effort, and no server-side refusal
    fallback for the models that don't offer one (Haiku 5.5)."""
    return ClaudeForecaster(model, conn=conn, effort=DRAFT_EFFORT,
                            fallbacks=model not in NO_FALLBACK_MODELS, **kwargs)  # fmt: skip


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def draft_prompt(conn: Connection, f: AIForecastToReview, category: str, notes: str,
                 store: VintageStore) -> str:  # fmt: skip
    """The user message for one draft: everything the explanation may use."""
    q = load_question(conn, f.question_id)
    question = q.question_text or (
        f"What will {f.question_key} be for {f.target_date:%Y-%m}?"
        if f.question_type != "binary" else f"Will {f.question_key} for {f.target_date:%Y-%m}?"
    )  # fmt: skip
    if f.question_type == "binary":
        forecast = f"P(yes) = {f.p_yes:.2f}"
        outcome = (
            f"The question resolved {f.actual_outcome!r} (known {f.actual_as_of})."
        )
    else:
        interval = (f", 80% interval [{_fmt(f.interval_lower)}, {_fmt(f.interval_upper)}]"
                    if f.interval_lower is not None else "")  # fmt: skip
        forecast = f"{_fmt(f.prediction)}{interval}"
        outcome = (
            f"The actual value was {_fmt(f.actual_value)} (known {f.actual_as_of})."
        )
    parts = [
        f"<question>\n{question}\n</question>",
        *([f"<resolution_rule>\n{q.resolution_rule}\n</resolution_rule>"] if q.resolution_rule else []),
        f"<forecast model=\"{f.model_type}\" made=\"{f.forecast_date}\">\n{forecast}\n</forecast>",
        f"<outcome>\n{outcome}\n</outcome>",
        f"<classifier category=\"{category}\">\n{notes}\n</classifier>",
    ]  # fmt: skip
    if f.rationale:
        parts.append(f"<model_rationale>\n{f.rationale}\n</model_rationale>")
    if f.shap_values:
        top = Explanation.from_json(f.shap_values).top(5)
        drivers = "\n".join(f"{name}: {value:+.4f}" for name, value in top)
        parts.append(
            f'<model_drivers note="SHAP, on the model\'s scale">\n{drivers}\n</model_drivers>'
        )
    if f.series_id is not None:
        known = store.known(f.series_id, f.actual_as_of)
        since = known[
            known["observed_at"]
            >= pd.Timestamp(f.forecast_date) - pd.DateOffset(months=3)
        ]
        rows = "\n".join(f"{pd.Timestamp(m):%Y-%m}: {_fmt(float(v))}"
                         for m, v in zip(since["observed_at"], since["value"]))  # fmt: skip
        if rows:
            parts.append(
                f'<series_values series="{f.series_id}">\n{rows}\n</series_values>'
            )
    query = SERIES_QUERIES.get(f.series_id or "", question)
    docs = []
    for kind, n in DRAFT_DOCUMENTS.items():
        docs += retrieve_documents(
            conn, f.actual_as_of, kind, n, query=query, since=f.cutoff
        )
    if docs:
        parts.append("<published_since_the_forecast>\n" + "\n".join(
            f"<document type=\"{d.document_type}\" published=\"{d.published_at:%Y-%m-%d}\" "
            f"title=\"{(d.title or '').replace(chr(34), chr(39))}\">\n{d.content[:800]}\n</document>"
            for d in docs
        ) + "\n</published_since_the_forecast>")  # fmt: skip
    parts.append("Explain in 2-3 sentences why this forecast missed.")
    return "\n\n".join(parts)


def _validate_draft(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("explanation"), str):
        raise TypeError("the reply has no explanation")
    text = parsed["explanation"].strip()
    if not text:
        raise ValueError("the explanation is empty")
    if len(text) > DRAFT_MAX_CHARS:
        raise ValueError(
            f"the explanation is {len(text)} characters, over {DRAFT_MAX_CHARS}"
        )
    return {"explanation": text}


def draft_llm_postmortem(
    conn: Connection,
    forecast_id: int,
    drafter: ClaudeForecaster,
    *,
    store: VintageStore | None = None,
) -> dict[str, Any]:
    """Draft and store one forecast's explanation. The forecast must have a
    post-mortem (its category) already. Returns the call's result, with
    `explanation`; raises ForecastError if the call fails."""
    (f,) = ai_forecasts_to_review(conn, forecast_id) or [None]  # type: ignore[list-item]
    if f is None:
        raise ValueError(f"forecast {forecast_id} isn't a scored AI forecast")
    with conn.cursor() as cur:
        cur.execute("SELECT miss_category::text, notes FROM postmortems WHERE forecast_id = %s",
                    (forecast_id,))  # fmt: skip
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"forecast {forecast_id} has no post-mortem to supplement yet")
    prompt = draft_prompt(conn, f, row[0], row[1] or "", store or store_from_conn(conn))
    result = drafter.complete(
        prompt, DRAFT_SCHEMA, _validate_draft, system=DRAFT_SYSTEM,
        prompt_version=DRAFT_PROMPT_VERSION, question_id=f.question_id,
        as_of_date=f.actual_as_of,
    )  # fmt: skip
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE postmortems SET llm_drafted = TRUE, draft_note = %s, draft_call_id = %s,
                   human_confirmed = FALSE, confirmed_at = NULL, updated_at = NOW()
            WHERE forecast_id = %s
            """,
            (result["explanation"], result.get("call_id"), forecast_id),
        )
    return result


def undrafted_misses(conn: Connection, *, include_backtest: bool = False) -> list[int]:
    """AI forecasts whose post-mortem is a miss (not expected variance) with no
    draft yet, newest target first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.forecast_id FROM postmortems p
            JOIN forecasts f ON f.id = p.forecast_id
            JOIN questions q ON q.id = f.question_id
            WHERE q.domain = 'ai' AND p.miss_category <> 'expected_variance'
              AND NOT p.llm_drafted AND (%s OR NOT f.is_backtest)
            ORDER BY q.target_date DESC, p.forecast_id
            """,
            (include_backtest,),
        )
        return [r[0] for r in cur.fetchall()]


def draft_postmortems(
    conn: Connection,
    drafter: ClaudeForecaster,
    budget: BacktestBudget,
    *,
    include_backtest: bool = False,
    limit: int | None = None,
    commit: Callable[[], None] | None = None,
) -> AIPostmortemResult:
    """Draft every undrafted miss until done, `limit`, or the budget runs out.
    `commit` runs after each draft (the CLI commits each)."""
    result = AIPostmortemResult()
    store = store_from_conn(conn)
    for forecast_id in undrafted_misses(conn, include_backtest=include_backtest)[
        :limit
    ]:
        try:
            budget.check()
            r = draft_llm_postmortem(conn, forecast_id, drafter, store=store)
        except BudgetExceeded as exc:
            result.failures.append(str(exc))
            break
        except ForecastError as exc:
            result.failures.append(f"forecast {forecast_id}: {exc}")
            continue
        budget.record(r)
        result.drafted += 1
        if commit:
            commit()
    return result


# --- review ----------------------------------------------------------------------------------


def list_drafts(conn: Connection, *, include_confirmed: bool = False) -> pd.DataFrame:
    """Drafted AI post-mortems, unconfirmed first, with what each explains."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.forecast_id, q.question_key, q.target_date, m.model_type,
                   p.miss_category::text AS category, p.notes, p.draft_note,
                   p.human_confirmed, p.confirmed_at
            FROM postmortems p
            JOIN forecasts f ON f.id = p.forecast_id
            JOIN questions q ON q.id = f.question_id
            JOIN model_versions m ON m.id = f.model_version_id
            WHERE q.domain = 'ai' AND p.llm_drafted AND (%s OR NOT p.human_confirmed)
            ORDER BY p.human_confirmed, q.target_date DESC, p.forecast_id
            """,
            (include_confirmed,),
        )
        columns = [c.name for c in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def confirm_draft(conn: Connection, forecast_id: int, note: str | None = None) -> bool:
    """Confirm a draft, optionally replacing its text; False if there's no draft."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE postmortems SET human_confirmed = TRUE, confirmed_at = NOW(),
                   draft_note = COALESCE(%s, draft_note), updated_at = NOW()
            WHERE forecast_id = %s AND llm_drafted
            RETURNING 1
            """,
            (note, forecast_id),
        )
        return cur.fetchone() is not None


def reject_draft(conn: Connection, forecast_id: int) -> bool:
    """Discard a draft (the category stays); False if there's no draft."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE postmortems SET llm_drafted = FALSE, human_confirmed = FALSE,
                   draft_note = NULL, draft_call_id = NULL, confirmed_at = NULL,
                   updated_at = NOW()
            WHERE forecast_id = %s AND llm_drafted
            RETURNING 1
            """,
            (forecast_id,),
        )
        return cur.fetchone() is not None


def review_main(
    argv: Sequence[str] | None = None, *, connect: Callable[[], Any] | None = None
) -> int:
    """`eco-forecast review-postmortems --domain ai`: list, confirm or reject drafts."""
    parser = argparse.ArgumentParser(prog="eco-forecast review-postmortems",
                                     description="Review LLM-drafted post-mortems.")  # fmt: skip
    parser.add_argument("--domain", choices=["ai"], default="ai",
                        help="only the AI domain has drafts")  # fmt: skip
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--confirm", type=int, metavar="FORECAST_ID")
    action.add_argument("--reject", type=int, metavar="FORECAST_ID")
    parser.add_argument(
        "--note", help="with --confirm: the text to keep instead of the draft"
    )
    parser.add_argument("--all", action="store_true", help="list confirmed drafts too")
    args = parser.parse_args(argv)
    if args.note and args.confirm is None:
        parser.error("--note goes with --confirm")
    from eco_prediction.db.connection import connection

    with (connect or connection)() as conn:
        if args.confirm is not None:
            ok = confirm_draft(conn, args.confirm, args.note)
            conn.commit()
            print(
                f"Forecast {args.confirm}: "
                + ("confirmed" if ok else "no draft to confirm")
            )
            return 0 if ok else 1
        if args.reject is not None:
            ok = reject_draft(conn, args.reject)
            conn.commit()
            print(
                f"Forecast {args.reject}: "
                + ("draft discarded" if ok else "no draft to reject")
            )
            return 0 if ok else 1
        drafts = list_drafts(conn, include_confirmed=args.all)
    if drafts.empty:
        print("No drafts to review.")
        return 0
    for _, d in drafts.iterrows():
        state = "confirmed" if d["human_confirmed"] else "unconfirmed"
        print(f"\n#{d['forecast_id']}  {d['question_key']} {d['target_date']}  "
              f"{d['model_type']}  [{d['category']}, {state}]")  # fmt: skip
        print(f"  classifier: {d['notes']}")
        print(f"  draft:      {d['draft_note']}")
    print(
        f"\n{len(drafts)} draft(s). Confirm with --confirm ID [--note TEXT], or --reject ID."
    )
    return 0


def main(
    argv: Sequence[str] | None = None, *, connect: Callable[[], Any] | None = None
) -> int:
    """Classify AI forecasts, optionally draft explanations, list the misses."""
    from eco_prediction.db.connection import connection
    from eco_prediction.scheduler.forecast_job import today

    parser = argparse.ArgumentParser(description="Post-mortems for AI forecasts.")
    parser.add_argument(
        "--draft",
        metavar="MODEL",
        nargs="?",
        const=DRAFT_MODEL,
        help=f"also draft explanations with Claude (default {DRAFT_MODEL})",
    )
    parser.add_argument(
        "--budget", type=float, metavar="USD", help="required with --draft"
    )
    parser.add_argument("--limit", type=int, help="draft at most this many")
    parser.add_argument("--include-backtest", action="store_true",
                        help="draft backtest misses too")  # fmt: skip
    args = parser.parse_args(argv)
    if args.draft and args.budget is None:
        parser.error("--draft needs --budget")
    with (connect or connection)() as conn:
        classified = run_ai_postmortems(conn, today())
        conn.commit()
        print(f"Classified {classified.classified}: {dict(classified.by_category)}")
        if args.draft:
            drafted = draft_postmortems(
                conn, make_drafter(conn, args.draft), BacktestBudget(args.budget),
                include_backtest=args.include_backtest, limit=args.limit, commit=conn.commit,
            )  # fmt: skip
            print(
                f"Drafted {drafted.drafted}"
                + (f"; stopped: {drafted.failures}" if drafted.failures else "")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

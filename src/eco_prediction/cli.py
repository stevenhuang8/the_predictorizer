"""`eco-forecast`: generate and inspect forecasts by hand.

Usage:
    uv run eco-forecast forecast --target cpi --horizon 3      # every model, data through yesterday
    uv run eco-forecast forecast --target unemployment --horizon 1 --model lightgbm
    uv run eco-forecast forecast --target fomc                 # the next scheduled meeting
    uv run eco-forecast forecast --target fomc --meeting 2027-01-27
    uv run eco-forecast inspect                                # list questions
    uv run eco-forecast inspect --question-id 3                # forecasts, scores, post-mortems
    uv run eco-forecast scores [--include-backtest]            # the live scorecard
    uv run eco-forecast backtest [--refresh] [--start ...]     # Task 19's model comparison

`forecast` only displays: it refits the models on the data in the database
and prints what they predict now, without storing anything. The daily job
(`scheduler.forecast_job`) is what records live forecasts, so a manual run
can never change the track record. LightGBM forecasts show their top SHAP
drivers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.backtest.data import TARGETS, PointInTimeData, VintageStore
from eco_prediction.backtest.harness import add_months, month_start
from eco_prediction.data.fomc import FOMC_MEETINGS, RELEASE_LAGS
from eco_prediction.models.explainability import SHAPExplainer
from eco_prediction.models.fomc_model import FOMCForecaster
from eco_prediction.models.lightgbm_model import LightGBMModel
from eco_prediction.scheduler.forecast_job import (
    DEFAULT_FOMC,
    DEFAULT_NUMERIC,
    FOMCModelSpec,
    NumericModelSpec,
    today,
)

TARGET_NAMES = {
    "cpi": "cpi_yoy",
    "unemployment": "unemployment",
    "fomc": "fomc_decision",
}
SHAP_TOP = 5


def _models(
    specs: Sequence[NumericModelSpec], only: str | None
) -> list[NumericModelSpec]:
    chosen = [s for s in specs if only is None or s.model_type == only]
    if not chosen:
        names = ", ".join(s.model_type for s in specs)
        raise SystemExit(f"error: unknown model {only!r}; choose from {names}")
    return chosen


def forecast_numeric(
    store: VintageStore,
    target: str,
    horizon: int,
    as_of: date,
    *,
    model: str | None = None,
    specs: Sequence[NumericModelSpec] = DEFAULT_NUMERIC,
) -> str:
    """Every model's forecast of `target` `horizon` months after as_of's month."""
    cutoff = as_of - timedelta(days=1)
    period = add_months(month_start(as_of), horizon)
    view = PointInTimeData(store.until(cutoff), TARGETS[target], cutoff)
    latest = view.target_history().dropna()
    lines = [
        f"{TARGETS[target].description}: {period:%B %Y}, {horizon} month(s) ahead",
        f"Data through {cutoff}; latest value {latest.iloc[-1]:.2f} "
        f"({latest.index[-1]:%B %Y})."
        if len(latest)
        else f"Data through {cutoff}; no values known.",
        "",
    ]
    rows = []
    for spec in _models(specs, model):
        m = spec.build(horizon)
        m.fit(view)
        f = m.predict(view, period)
        rows.append(
            {
                "model": spec.model_type,
                "forecast": f.point,
                "lower": f.lower,
                "upper": f.upper,
            }
        )
        if isinstance(m, LightGBMModel):
            explanation = SHAPExplainer(m).explain(view, period)
            drivers = ", ".join(f"{k} {v:+.2f}" for k, v in explanation.top(SHAP_TOP))
            rows[-1]["drivers"] = drivers
    table = pd.DataFrame(rows)
    lines.append(
        table.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="")
    )
    return "\n".join(lines)


def forecast_fomc(
    store: VintageStore,
    as_of: date,
    *,
    meeting: date | None = None,
    model: str | None = None,
    specs: Sequence[FOMCModelSpec] = DEFAULT_FOMC,
) -> str:
    """Cut/hold/hike probabilities for `meeting` (default: the next one)."""
    if meeting is None:
        upcoming = [m for m in FOMC_MEETINGS if m > as_of]
        if not upcoming:
            raise SystemExit("error: no scheduled meetings after today in the calendar")
        meeting = upcoming[0]
    elif meeting not in FOMC_MEETINGS:
        raise SystemExit(f"error: {meeting} is not a scheduled FOMC decision day")
    if meeting <= as_of:
        raise SystemExit(f"error: the {meeting} meeting has already happened")
    lead = (meeting - as_of).days
    cutoff = as_of - timedelta(days=1)
    clipped = store.until(cutoff)
    chosen = [s for s in specs if model is None or s.model_type == model]
    if not chosen:
        names = ", ".join(s.model_type for s in specs)
        raise SystemExit(f"error: unknown model {model!r}; choose from {names}")
    rows = []
    for spec in chosen:
        m = spec.build(lead)
        m.fit(clipped, cutoff)
        probs = m.predict_proba(clipped, meeting)
        row: dict[str, object] = {"model": spec.model_type, **probs}
        if isinstance(m, FOMCForecaster):
            importance = m.feature_importance()
            row["top features"] = ", ".join(importance.index[:3])
        rows.append(row)
    table = pd.DataFrame(rows)
    return "\n".join(
        [
            f"FOMC decision on {meeting}, {lead} days ahead (data through {cutoff})",
            "",
            table.to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep=""),
        ]
    )


def list_questions(conn: Connection) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.target::text, q.target_date,
                   COALESCE(q.horizon_months || ' months', q.lead_days || ' days'),
                   count(f.id) FILTER (WHERE NOT f.is_backtest),
                   count(f.id) FILTER (WHERE f.is_backtest),
                   COALESCE(r.actual_value::text, r.actual_outcome, '')
            FROM questions q
            LEFT JOIN forecasts f ON f.question_id = q.id
            LEFT JOIN resolutions r ON r.question_id = q.id
            GROUP BY q.id, r.actual_value, r.actual_outcome
            ORDER BY q.target_date, q.target, q.id
            """
        )
        rows = cur.fetchall()
    if not rows:
        return "No questions yet."
    frame = pd.DataFrame(
        rows,
        columns=[
            "id",
            "target",
            "target_date",
            "horizon",
            "live",
            "backtest",
            "actual",
        ],
    )
    return frame.to_string(index=False)


def inspect_question(conn: Connection, question_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.target::text, q.target_date, q.horizon_months, q.lead_days,
                   q.resolution_rule, r.actual_value, r.actual_outcome, r.actual_as_of
            FROM questions q LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE q.id = %s
            """,
            (question_id,),
        )
        question = cur.fetchone()
        if question is None:
            raise SystemExit(f"error: no question with id {question_id}")
        cur.execute(
            """
            SELECT f.id, m.model_type, f.forecast_date, f.is_backtest, f.prediction,
                   f.interval_lower, f.interval_upper, f.probabilities, f.error,
                   f.in_interval, f.brier_score, p.miss_category::text, p.notes
            FROM forecasts f
            JOIN model_versions m ON m.id = f.model_version_id
            LEFT JOIN postmortems p ON p.forecast_id = f.id
            WHERE f.question_id = %s
            ORDER BY f.is_backtest, f.forecast_date, m.model_type
            """,
            (question_id,),
        )
        forecasts = cur.fetchall()
    target, target_date, horizon, lead, rule, value, outcome, actual_as_of = question
    when = f"{horizon} months ahead" if horizon is not None else f"{lead} days before"
    if value is None and outcome is None:
        status = "open"
    else:
        answer = outcome if outcome is not None else f"{float(value):.2f}"
        status = f"resolved: {answer} (published {actual_as_of})"
    lines = [
        f"Question {question_id}: {target} for {target_date}, {when}",
        f"Rule: {rule}" if rule else "",
        f"Status: {status}",
        "",
    ]
    if not forecasts:
        return "\n".join([*lines, "No forecasts."])
    rows = []
    for (
        fid, model_type, fdate, backtest, pred, lower, upper, probs, error,
        in_interval, brier, category, notes,
    ) in forecasts:  # fmt: skip
        forecast = (
            ", ".join(f"{k} {v:.2f}" for k, v in probs.items())
            if probs
            else f"{float(pred):.2f} [{float(lower):.2f}, {float(upper):.2f}]"
            if lower is not None
            else f"{float(pred):.2f}"
        )
        score = (
            f"brier {float(brier):.3f}"
            if brier is not None
            else f"error {float(error):+.2f}" + ("" if in_interval else " (outside)")
            if error is not None
            else ""
        )
        rows.append(
            {
                "id": fid,
                "model": model_type,
                "made": fdate,
                "kind": "backtest" if backtest else "live",
                "forecast": forecast,
                "score": score,
                "post-mortem": category or "",
            }
        )
    lines.append(pd.DataFrame(rows).to_string(index=False))
    notes_lines = [f"  {r[0]}: {r[12]}" for r in forecasts if r[12]]
    if notes_lines:
        lines += ["", "Post-mortem notes:", *notes_lines]
    return "\n".join(line for line in lines if line is not None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eco-forecast", description="Generate and inspect forecasts by hand."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", help="show every model's forecast now (not stored)")
    f.add_argument("--target", choices=sorted(TARGET_NAMES), required=True)
    f.add_argument(
        "--horizon", type=int, default=3, help="months ahead (cpi, unemployment)"
    )
    f.add_argument(
        "--meeting", type=date.fromisoformat, help="FOMC decision day (fomc)"
    )
    f.add_argument("--model", help="only this model, e.g. lightgbm")
    f.add_argument(
        "--date",
        type=date.fromisoformat,
        help="forecast as of this date (default today), using data published before it",
    )

    i = sub.add_parser("inspect", help="list questions, or show one in detail")
    i.add_argument("--question-id", type=int)

    s = sub.add_parser("scores", help="accuracy by target, horizon and model")
    s.add_argument("--include-backtest", action="store_true")

    sub.add_parser(
        "backtest",
        help="run the model comparison report (Task 19); options pass through",
        add_help=False,  # so `backtest --help` shows the comparison's options
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.command == "backtest":
        # Everything after `backtest` goes to the comparison script as is.
        from eco_prediction.scripts.model_comparison import main as comparison

        return comparison(extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    from eco_prediction.db.connection import connection

    if args.command == "forecast":
        if args.horizon < 1:
            raise SystemExit("error: --horizon must be at least 1")
        store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)
        as_of = args.date or today()
        target = TARGET_NAMES[args.target]
        if target == "fomc_decision":
            print(forecast_fomc(store, as_of, meeting=args.meeting, model=args.model))
        else:
            print(
                forecast_numeric(store, target, args.horizon, as_of, model=args.model)
            )
        return 0

    with connection() as conn:
        if args.command == "inspect":
            print(
                list_questions(conn)
                if args.question_id is None
                else inspect_question(conn, args.question_id)
            )
        elif args.command == "scores":
            from eco_prediction.db.resolutions import scored_forecasts
            from eco_prediction.scheduler.resolution_job import score_summary

            summary = score_summary(
                scored_forecasts(conn, include_backtest=args.include_backtest)
            )
            print(
                "No scored forecasts yet."
                if summary.empty
                else summary.round(3).to_string(index=False)
            )
    return 0


def cli() -> None:
    """Console-script entry point (`eco-forecast`)."""
    sys.exit(main())


if __name__ == "__main__":
    cli()

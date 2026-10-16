"""Forecast questions for the AI domain: numeric, threshold and curated.

Usage:
    uv run python -m eco_prediction.ai.questions                  # all three, as of today
    uv run python -m eco_prediction.ai.questions numeric          # or thresholds, curated
    uv run python -m eco_prediction.ai.questions --date 2026-11-01
    uv run python -m eco_prediction.ai.questions --dry-run        # print, write nothing

    from eco_prediction.ai.questions import generate_numeric_questions
    result = generate_numeric_questions(conn, date(2026, 10, 15))
    conn.commit()

**Numeric questions** (AI task 9):

Like the economy's CPI and unemployment questions, a run in month M asks
about months M + 1, M + 3 and M + 6, one question per series and horizon:
- ARXIV_AI_SUBMISSIONS, HF_NEW_MODELS, GITHUB_AI_ACTIVITY: the month's count.
- EPOCH_GPQA_SOTA, EPOCH_SWEBENCH_SOTA, EPOCH_ARC2_SOTA (`benchmark_sota`)
  and EPOCH_LARGEST_FLOPS (`largest_training_run`): step series, asked the
  same way: the best value so far at the target month's end. Whether a step
  series moves at all is asked by the threshold questions below.

Each question's `question_key` is its series id (migration 012), so the three
benchmarks sharing `benchmark_sota` are separate questions and each names the
series that resolves it. GITHUB_AI_STARS is a feature, not a target.

A question resolves to the first live value stored for its month, the AI
counterpart of the economy's first release. Live pulls run after a month
ends, and backfilled values (`reconstructed`) don't count. Later revisions
(late evaluations on Epoch, deleted models on Hugging Face) don't change it.

**Threshold questions** (AI task 10) are the yes/no twins of the numeric
ones: same series, month, horizon and first-live resolution, asking whether
the value will be at least a threshold. Thresholds are steps up from the
series' value for the month before the ask month (M - 1), as known on the ask
date, rounded to readable numbers:
- counts: +10%, +25% and +50%, to 3 significant figures;
- SOTA percentages: closing 25%, 50% and 75% of the gap to 100%, to 0.1;
- compute (log10 FLOP): 2x, 3x and 10x the record, to 0.01.
The key is '<series>>=<threshold>', e.g. 'EPOCH_SWEBENCH_SOTA>=87.6'. A month
gets one set per series and horizon: once any exists it's reused, so a value
revised mid-month doesn't add a second set. A series with no value for M - 1
yet (not ingested) is skipped and picked up by a later run.

**Curated questions** (AI task 10) mirror Polymarket markets: see
`ai/curated_questions.py`.

Runs are safe to repeat: existing questions are reused.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from psycopg2.extensions import connection as Connection

from eco_prediction.ai import (
    ingest_arxiv,
    ingest_epoch,
    ingest_github,
    ingest_huggingface,
)
from eco_prediction.ai.curated_questions import CuratedResult, load_curated_questions
from eco_prediction.ai.months import month_start
from eco_prediction.ai.store import AISeriesInfo
from eco_prediction.backtest.harness import add_months
from eco_prediction.db.connection import ConfigError, connection
from eco_prediction.db.forecasts import get_or_create_question

HORIZONS = (1, 3, 6)


@dataclass(frozen=True)
class NumericTarget:
    target: str  # forecast_target enum value
    series: AISeriesInfo  # the series that resolves it


NUMERIC_TARGETS: tuple[NumericTarget, ...] = (
    NumericTarget("arxiv_ai_submissions", ingest_arxiv.SERIES.info),
    NumericTarget("hf_new_models", ingest_huggingface.SERIES.info),
    NumericTarget("github_ai_activity", ingest_github.SERIES.info),
    NumericTarget("benchmark_sota", ingest_epoch.SERIES["gpqa"].series.info),
    NumericTarget("benchmark_sota", ingest_epoch.SERIES["swe_bench"].series.info),
    NumericTarget("benchmark_sota", ingest_epoch.SERIES["arc_agi_2"].series.info),
    NumericTarget("largest_training_run", ingest_epoch.SERIES["compute"].series.info),
)


@dataclass(frozen=True)
class AIQuestion:
    target: str
    series_id: str  # stored as question_key
    target_date: date  # first day of the target month
    horizon_months: int
    resolution_rule: str


@dataclass
class GenerateResult:
    questions: list[tuple[AIQuestion, int]]  # with their ids
    created: int


def resolution_rule(series: AISeriesInfo, month: date) -> str:
    return (
        f"{series.name}, for {month:%Y-%m} ({series.series_id}, "
        f"{series.units}): the first live value stored after the month ends "
        "(not a reconstructed backfill). Later revisions don't change it."
    )


def plan_numeric_questions(
    as_of_date: date,
    targets: Sequence[NumericTarget] = NUMERIC_TARGETS,
    horizons: Sequence[int] = HORIZONS,
) -> list[AIQuestion]:
    """The numeric questions a forecast made on as_of_date answers."""
    questions = []
    for h in horizons:
        month = add_months(month_start(as_of_date), h)
        questions += [
            AIQuestion(
                t.target,
                t.series.series_id,
                month,
                h,
                resolution_rule(t.series, month),
            )
            for t in targets
        ]
    return questions


def generate_numeric_questions(
    conn: Connection,
    as_of_date: date,
    targets: Sequence[NumericTarget] = NUMERIC_TARGETS,
    horizons: Sequence[int] = HORIZONS,
) -> GenerateResult:
    """Create (or reuse) every numeric AI question for as_of_date. No commit."""
    result = GenerateResult([], 0)
    for q in plan_numeric_questions(as_of_date, targets, horizons):
        question_id, created = get_or_create_question(
            conn,
            q.target,
            q.target_date,
            q.horizon_months,
            resolution_rule=q.resolution_rule,
            question_key=q.series_id,
        )
        result.questions.append((q, question_id))
        result.created += created
    return result


# --- threshold questions ----------------------------------------------------

BINARY_TARGET = "ai_binary_event"
COUNT_STEPS = ((1.10, "+10%"), (1.25, "+25%"), (1.50, "+50%"))
GAP_STEPS = ((0.25, "25%"), (0.50, "50%"), (0.75, "75%"))  # of the gap to 100%
FLOP_STEPS = ((2, "2x"), (3, "3x"), (10, "10x"))


def _kind(series: AISeriesInfo) -> str:
    return {"Percent": "gap", "log10(FLOP)": "flop"}.get(series.units, "count")


def format_value(series: AISeriesInfo, value: float) -> str:
    kind = _kind(series)
    if kind == "gap":
        return f"{value:.1f}%"
    if kind == "flop":
        return f"10^{value:.2f} FLOP"
    return f"{value:,.0f}"


@dataclass(frozen=True)
class Threshold:
    value: float
    key: str  # as written in question_key
    step: str  # how it was set, e.g. '+10%'


def thresholds(series: AISeriesInfo, base: float) -> list[Threshold]:
    """The steps up from `base`, rounded; ones that round to base or repeat are dropped."""
    kind = _kind(series)
    found: list[Threshold] = []
    if kind == "count":
        for mult, label in COUNT_STEPS:
            v = float(f"{base * mult:.3g}")
            found.append(Threshold(v, f"{v:.0f}", label))
    elif kind == "gap":
        for frac, label in GAP_STEPS:
            v = round(base + (100 - base) * frac, 1)
            if v <= 100:
                found.append(Threshold(v, f"{v:.1f}", f"{label} of the gap to 100%"))
    else:
        for mult, label in FLOP_STEPS:
            v = round(base + math.log10(mult), 2)
            found.append(Threshold(v, f"{v:.2f}", f"{label} the record"))
    unique: dict[str, Threshold] = {}
    for t in found:
        if t.value > base:
            unique.setdefault(t.key, t)
    return list(unique.values())


@dataclass(frozen=True)
class ThresholdQuestion:
    series: AISeriesInfo
    target_date: date  # first day of the target month
    horizon_months: int
    base_month: date
    base_value: float
    threshold: Threshold

    @property
    def question_key(self) -> str:
        return f"{self.series.series_id}>={self.threshold.key}"

    @property
    def resolution_rule(self) -> str:
        return (
            f"{resolution_rule(self.series, self.target_date)} Resolves yes if "
            f"that value is at least {format_value(self.series, self.threshold.value)}"
            ", otherwise no."
        )

    @property
    def question_text(self) -> str:
        s = self.series
        return (
            f"Will {s.name} be at least {format_value(s, self.threshold.value)} "
            f"for {self.target_date:%Y-%m}? It was "
            f"{format_value(s, self.base_value)} for {self.base_month:%Y-%m} "
            f"(threshold: {self.threshold.step}).\n\n{self.resolution_rule}"
        )


def latest_known(
    conn: Connection, series_id: str, month: date, as_of_date: date
) -> float | None:
    """The series' latest value for `month` known on as_of_date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.value FROM observations o JOIN series s ON s.id = o.series_id
            WHERE s.series_id = %s AND o.observed_at = %s AND o.as_of <= %s
              AND o.value IS NOT NULL
            ORDER BY o.as_of DESC LIMIT 1
            """,
            (series_id, month, as_of_date),
        )
        row = cur.fetchone()
    return None if row is None else float(row[0])


def _existing_set(
    conn: Connection, series_id: str, month: date, horizon: int
) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT question_key, id FROM questions
            WHERE target = %s AND target_date = %s AND horizon_months = %s
              AND starts_with(question_key, %s)
            ORDER BY id
            """,
            (BINARY_TARGET, month, horizon, f"{series_id}>="),
        )
        return [(str(k), int(i)) for k, i in cur.fetchall()]


@dataclass
class ThresholdResult:
    # (question_key, target month, horizon, id) of every question asked
    questions: list[tuple[str, date, int, int]]
    created: int
    missing_base: list[str]  # series with no value for M - 1 yet


def generate_threshold_binary_questions(
    conn: Connection,
    as_of_date: date,
    targets: Sequence[NumericTarget] = NUMERIC_TARGETS,
    horizons: Sequence[int] = HORIZONS,
) -> ThresholdResult:
    """Create (or reuse) the threshold questions for as_of_date. No commit."""
    ask_month = month_start(as_of_date)
    base_month = add_months(ask_month, -1)
    result = ThresholdResult([], 0, [])
    for t in targets:
        sid = t.series.series_id
        base: float | None = None
        for h in horizons:
            month = add_months(ask_month, h)
            existing = _existing_set(conn, sid, month, h)
            if existing:
                result.questions += [(k, month, h, i) for k, i in existing]
                continue
            if base is None:
                base = latest_known(conn, sid, base_month, as_of_date)
                if base is None:
                    result.missing_base.append(sid)
                    break
            for th in thresholds(t.series, base):
                q = ThresholdQuestion(t.series, month, h, base_month, base, th)
                question_id, created = get_or_create_question(
                    conn,
                    BINARY_TARGET,
                    month,
                    h,
                    resolution_rule=q.resolution_rule,
                    question_key=q.question_key,
                    question_text=q.question_text,
                )
                result.questions.append((q.question_key, month, h, question_id))
                result.created += created
    return result


# --- CLI --------------------------------------------------------------------

KINDS = ("numeric", "thresholds", "curated")


def format_question(q: AIQuestion) -> str:
    return f"{q.target_date:%Y-%m}  h{q.horizon_months}  {q.target:<21} {q.series_id}"


def _print_numeric(result: GenerateResult, ids: bool) -> None:
    for q, question_id in result.questions:
        print(format_question(q) + (f"  id={question_id}" if ids else ""))
    print(f"numeric questions={len(result.questions)} created={result.created}")


def _print_thresholds(result: ThresholdResult, ids: bool) -> None:
    for key, month, h, question_id in result.questions:
        print(f"{month:%Y-%m}  h{h}  {key}" + (f"  id={question_id}" if ids else ""))
    for sid in result.missing_base:
        print(f"skipped {sid}: no value for last month yet (ingest it first)")
    print(f"threshold questions={len(result.questions)} created={result.created}")


def _print_curated(result: CuratedResult, dry_run: bool) -> None:
    # A dry run lists every question due; a real run only the new ones.
    ids = not dry_run
    for q, question_id in result.questions if dry_run else result.new:
        print(
            f"{q.target_date}  {q.lead_days:>2}d  {q.category:<9} "
            f"{q.question_key:<20} {q.market.title}"
            + (f"  id={question_id}" if ids else "")
        )
    kept = ", ".join(f"{c} {result.categories[c]}" for c in sorted(result.categories))
    print(f"markets by category: {kept or 'none (run ingest_polymarket first)'}")
    print(f"curated questions={len(result.questions)} created={result.created}")


def main(
    argv: Sequence[str] | None = None, *, today: Callable[[], date] = date.today
) -> int:
    parser = argparse.ArgumentParser(
        description="Create the AI domain's forecast questions."
    )
    parser.add_argument(
        "kind",
        nargs="?",
        choices=(*KINDS, "all"),
        default="all",
        help="which questions (default all)",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="ask as of this date (default today)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the questions, write nothing"
    )
    args = parser.parse_args(argv)
    as_of = args.date or today()
    kinds = KINDS if args.kind == "all" else (args.kind,)

    if kinds == ("numeric",) and args.dry_run:
        planned = plan_numeric_questions(as_of)
        for q in planned:
            print(format_question(q))
        print(f"{len(planned)} questions")
        return 0
    try:
        with connection() as conn:
            if "numeric" in kinds:
                _print_numeric(
                    generate_numeric_questions(conn, as_of), not args.dry_run
                )
            if "thresholds" in kinds:
                _print_thresholds(
                    generate_threshold_binary_questions(conn, as_of), not args.dry_run
                )
            if "curated" in kinds:
                _print_curated(load_curated_questions(conn, as_of), args.dry_run)
            if args.dry_run:
                conn.rollback()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

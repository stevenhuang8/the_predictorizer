"""Curated binary questions mirrored from Polymarket (AI task 10).

    from eco_prediction.ai.curated_questions import load_curated_questions
    result = load_curated_questions(conn, date(2026, 10, 15))
    conn.commit()

(Run from the CLI as `python -m eco_prediction.ai.questions curated`.)

**Which markets.** Polymarket's `ai` tag is broad (AI_NOTES/Task8.md), so
`classify` sorts each market's title by keyword rules and only AI-progress
markets become questions:
- kept: `benchmark` (benchmark or Arena scores, IMO, AGI), `ranking` ("best AI
  model at the end of M", "#2 AI lab") and `release` ("GPT-6 released by D").
- dropped: `exact_day` legs ("released on February 12": one day of a
  many-way split), `finance` (valuations, IPOs, price indexes, AI trading contests),
  `people_legal`, `outage_app` (outages, app-store ranks), `hardware`
  (devices, robots, cars, chips) and anything matching no rule (`other`).
Drop rules are checked first, so "Anthropic's valuation at release" is
finance. The markets come from `platform_questions`, so `ingest_polymarket`
runs first; new markets are picked up on the next run.

**When they're asked.** Each kept market is asked `LEADS` (7, 30 and 90) days
before its scheduled close, one question per lead (`lead_days`), like the FOMC
week-before question. `target_date` is the close's UTC date. A lead is only
asked once its ask date has come (so a daily run adds each lead when it's
due), and only if the market was open on that date: opened on or before it,
and not yet resolved or closed. "Released by D" markets often resolve early;
asking after that would leak the answer. Markets that closed with no yes/no
outcome (50/50) can't resolve, so they're skipped.

**Links.** `question_key` is 'polymarket:<market id>', unique per lead, and
`platform_question_id` (migration 013) is the foreign key to the market,
whose stored prices are the benchmark. The question's text is the market
title, its rule the market's resolution criteria, both as first seen.

Runs are safe to repeat: existing questions are reused.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from psycopg2.extensions import connection as Connection

from eco_prediction.db.forecasts import get_or_create_question

PLATFORM = "polymarket"
LEADS = (7, 30, 90)
KEPT = ("benchmark", "ranking", "release")

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
# (category, alternatives), first match wins. Drops come first.
_RULES: list[tuple[str, list[str]]] = [
    (
        "exact_day",
        [rf"\b(release|released|launch|launched|out|debut)\b[^?]*\bon ({_MONTHS})\b"],
    ),
    (
        "finance",
        [
            r"valuation",
            r"market cap",
            r"\bIPO\b",
            r"initial public offering",
            r"acqui",
            r"funding",
            r"\braise[sd]?\b",
            r"revenue",
            r"stock",
            r"share price",
            r"\bindex\b",
            r"\$\d",
            r"for-profit",
            r"nonprofit",
            r"bankrupt",
            r"trading competition",
            r"alpha arena",
            r"\bNOF1\b",
            r"okbet",
        ],
    ),
    (
        "people_legal",
        [
            r"\bleave\b",
            r"\bCEO\b",
            r"step down",
            r"\bfired\b",
            r"resign",
            r"lawsuit",
            r"\bsue[sd]?\b",
            r"court",
            r"indict",
            r"charged",
            r"\bban\b",
            r"banned",
            r"regulat",
            r"\bbill\b",
            r"\bact\b",
            r"executive order",
            r"restored",
            r"congress",
            r"senate",
            r"\bEU\b",
            r"antitrust",
            r"testify",
            r"arrest",
        ],
    ),
    (
        "outage_app",
        [r"outage", r"app store", r"play store", r"\bdown\b", r"downtime", r"free app"],
    ),
    (
        "hardware",
        [
            r"hardware",
            r"device",
            r"robot",
            r"humanoid",
            r"cybercab",
            r"waymo",
            r"tesla",
            r"self-driving",
            r"robotaxi",
            r"\bchips?\b",
            r"\bGPUs?\b",
            r"data ?center",
        ],
    ),
    (
        "benchmark",
        [
            r"score",
            r"arena",
            r"leaderboard",
            r"humanity.s last exam",
            r"benchmark",
            r"ARC-AGI",
            r"FrontierMath",
            r"SWE-bench",
            r"GPQA",
            r"\bElo\b",
            r"\bIMO\b",
            r"gold medal",
            r"olympiad",
            r"\bAGI\b",
        ],
    ),
    (
        "ranking",
        [
            r"\bbest\b.*\b(model|AI|agent)\b",
            r"top \d+ AI",
            r"top AI model",
            r"#\d+",
            r"\bAI lab\b",
        ],
    ),
    (
        "release",
        [
            r"\b(release|released|launch|launched)\b",
            r"\b(debut|debuts|announce|announced|available)\b",
        ],
    ),
]
RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile("|".join(alts), re.IGNORECASE)) for name, alts in _RULES
)


def classify(title: str) -> str:
    """The market's category: one of `KEPT`, a drop reason, or 'other'."""
    for name, pattern in RULES:
        if pattern.search(title):
            return name
    return "other"


@dataclass(frozen=True)
class MarketRow:
    id: int  # platform_questions.id
    external_id: str
    title: str
    resolution_criteria: str | None
    url: str | None
    opened_at: datetime | None
    closes_at: datetime | None
    resolved_at: datetime | None
    resolved_outcome: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def stopped_on(self) -> date | None:
        """The day trading stopped, if it has (resolution or close)."""
        days = [d.date() for d in (self.resolved_at, _closed_at(self)) if d]
        return min(days) if days else None


def _closed_at(m: MarketRow) -> datetime | None:
    value = m.metadata.get("closed_at")
    return datetime.fromisoformat(value) if value else None


def _no_outcome(m: MarketRow) -> bool:
    """Closed and settled without a yes or no (e.g. 50/50)."""
    return (
        m.resolved_outcome is None
        and m.metadata.get("uma_resolution_status") == "resolved"
    )


@dataclass(frozen=True)
class CuratedQuestion:
    market: MarketRow
    category: str
    target_date: date  # the market's scheduled close (UTC date)
    lead_days: int

    @property
    def ask_date(self) -> date:
        return self.target_date - timedelta(days=self.lead_days)

    @property
    def question_key(self) -> str:
        return f"{PLATFORM}:{self.market.external_id}"

    @property
    def resolution_rule(self) -> str:
        criteria = (self.market.resolution_criteria or "").strip()
        source = f"Resolves as Polymarket market {self.market.external_id} does"
        if self.market.url:
            source += f" ({self.market.url})"
        return f"{criteria}\n\n{source}." if criteria else f"{source}."


def plan_curated_questions(
    markets: Sequence[MarketRow], today: date, leads: Sequence[int] = LEADS
) -> tuple[list[CuratedQuestion], Counter[str]]:
    """The questions due by `today`, and every market's category."""
    categories: Counter[str] = Counter()
    questions = []
    for m in markets:
        category = classify(m.title)
        categories[category] += 1
        if category not in KEPT or m.closes_at is None or m.opened_at is None:
            continue
        if _no_outcome(m):
            continue
        if m.resolved_at is None and "closed_at" not in m.metadata:
            continue  # stored before closes were recorded: whether it's open is unknown
        stopped = m.stopped_on()
        for lead in leads:
            q = CuratedQuestion(m, category, m.closes_at.date(), lead)
            if q.ask_date > today or m.opened_at.date() > q.ask_date:
                continue
            if stopped is not None and stopped <= q.ask_date:
                continue
            questions.append(q)
    return questions, categories


def platform_markets(conn: Connection) -> list[MarketRow]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, external_id, title, resolution_criteria, url, opened_at,
                   closes_at, resolved_at, resolved_outcome, metadata
            FROM platform_questions WHERE platform = %s ORDER BY id
            """,
            (PLATFORM,),
        )
        return [MarketRow(*row) for row in cur.fetchall()]


@dataclass
class CuratedResult:
    questions: list[tuple[CuratedQuestion, int]]  # with their ids
    new: list[tuple[CuratedQuestion, int]]  # the ones this call created
    categories: Counter[str]

    @property
    def created(self) -> int:
        return len(self.new)


def load_curated_questions(
    conn: Connection, today: date, leads: Sequence[int] = LEADS
) -> CuratedResult:
    """Create (or reuse) every curated question due by `today`. No commit."""
    planned, categories = plan_curated_questions(platform_markets(conn), today, leads)
    result = CuratedResult([], [], categories)
    for q in planned:
        question_id, created = get_or_create_question(
            conn,
            "ai_binary_event",
            q.target_date,
            lead_days=q.lead_days,
            resolution_rule=q.resolution_rule,
            question_key=q.question_key,
            question_text=q.market.title,
            platform_question_id=q.market.id,
        )
        result.questions.append((q, question_id))
        if created:
            result.new.append((q, question_id))
    return result

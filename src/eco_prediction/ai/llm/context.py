"""Point-in-time context for the LLM forecaster (AI task 18).

    from eco_prediction.ai.llm.context import build_context_bundle, load_question

    question = load_question(conn, 82)
    bundle = build_context_bundle(conn, question, date(2026, 10, 14))

A bundle is everything the prompt shows, each piece as known by the end of
`as_of_date` (the harness's cutoff: the day before the forecast):
- **recent values**: the question's series, its last `RECENT_MONTHS` (12)
  months, for numeric and threshold questions;
- **baselines**: for numeric questions the random walk, linear trend and
  ARIMA forecasts of the target month (`ai/baselines.py`,
  `ai/statistical_models.py`); for binary ones the base rate and, for
  curated questions, the market's price. A model that can't answer (too
  little history, no price yet) is left out;
- **documents**: retrieved from the corpus (`retrieval.retrieve_documents`)
  with a quota per document type (`DOCUMENT_QUOTAS`), because one type
  (dozens of near-identical Polymarket markets) would otherwise fill the
  list; only documents from the last `DOCUMENT_DAYS` (365).

The query is the question's own text for curated questions, and a
per-series description (`SERIES_QUERIES`) for numeric and threshold ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
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
from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.llm.retrieval import Document, retrieve_documents
from eco_prediction.ai.resolutions import parse_threshold_key
from eco_prediction.ai.statistical_models import AIARIMA
from eco_prediction.ai.targets import AI_TARGETS, kind_of
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecast
from eco_prediction.models.baselines import NotFittedError

RECENT_MONTHS = 12
DOCUMENT_DAYS = 365
DOCUMENT_QUOTAS: dict[str, int] = {
    "arxiv_abstract": 4,
    "polymarket_question": 3,
    "hf_model_card": 2,
    "github_release": 2,
}
SERIES_QUERIES: dict[str, str] = {
    "ARXIV_AI_SUBMISSIONS": "growth of AI research papers language models machine learning",
    "HF_NEW_MODELS": "open weights model release Hugging Face",
    "GITHUB_AI_ACTIVITY": "open source AI repository framework release",
    "EPOCH_GPQA_SOTA": "GPQA Diamond benchmark score frontier model",
    "EPOCH_SWEBENCH_SOTA": "SWE-bench Verified agent score",
    "EPOCH_ARC2_SOTA": "ARC-AGI-2 benchmark score",
    "EPOCH_LARGEST_FLOPS": "training compute FLOP largest frontier model scaling",
}

__all__ = [
    "Baseline",
    "ContextBundle",
    "LLMQuestion",
    "build_context_bundle",
    "load_question",
]


@dataclass(frozen=True)
class LLMQuestion:
    id: int | None
    question_type: str  # 'numeric_interval' or 'binary'
    question_key: str
    target_date: date
    horizon_months: int | None = None
    lead_days: int | None = None
    question_text: str | None = None
    resolution_rule: str | None = None
    platform_question_id: int | None = None
    created_at: datetime | None = None

    @property
    def binary(self) -> bool:
        return self.question_type == "binary"

    @property
    def curated(self) -> bool:
        return self.platform_question_id is not None

    @property
    def series_id(self) -> str | None:
        """The series the question is about; None for curated questions."""
        if self.curated:
            return None
        if self.binary:
            return parse_threshold_key(self.question_key)[0]
        return self.question_key

    @property
    def ask_date(self) -> date | None:
        """When a binary question is asked (as in `baselines.binary_questions`)."""
        if self.lead_days is not None:
            return self.target_date - timedelta(days=self.lead_days)
        return self.created_at.date() if self.created_at else None

    def as_binary(self, ask_date: date) -> BinaryQuestion:
        return BinaryQuestion(
            self.id, self.question_key, self.target_date, ask_date,
            self.horizon_months, self.platform_question_id, self.question_text,
        )  # fmt: skip


def load_question(conn: Connection, question_id: int) -> LLMQuestion:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, question_type::text, question_key, target_date, horizon_months,
                   lead_days, question_text, resolution_rule, platform_question_id,
                   created_at
            FROM questions WHERE id = %s
            """,
            (question_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"no question {question_id}")
    if row[2] is None:
        raise ValueError(f"question {question_id} is not an AI question (no key)")
    return LLMQuestion(*row)


@dataclass(frozen=True)
class Baseline:
    name: str
    point: float | None = None  # numeric
    lower: float | None = None
    upper: float | None = None
    probability: float | None = None  # binary


@dataclass(frozen=True)
class ContextBundle:
    question: LLMQuestion
    as_of_date: date  # data published by the end of this day
    series_id: str | None
    series_name: str | None
    units: str | None
    recent_values: list[tuple[date, float]] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


def _recent(
    store: VintageStore, series_id: str, as_of: date
) -> list[tuple[date, float]]:
    known = store.known(series_id, as_of).dropna(subset=["value"])
    tail = known.sort_values("observed_at").tail(RECENT_MONTHS)
    return [
        (pd.Timestamp(m).date(), float(v))
        for m, v in zip(tail["observed_at"], tail["value"])
    ]


def _numeric_baselines(
    store: VintageStore, series_id: str, as_of: date, target: date
) -> list[Baseline]:
    out = []
    for name, model in (
        ("random walk", AIRandomWalk()),
        ("linear trend (12 months)", AILinearTrend()),
        ("ARIMA", AIARIMA()),
    ):
        view = PointInTimeData(store, AI_TARGETS[series_id], as_of)
        try:
            model.fit(view)
            f: Forecast = model.predict(view, target)
        except (ValueError, NotFittedError):
            continue
        out.append(
            Baseline(
                name,
                float(f.point),
                None if f.lower is None else float(f.lower),
                None if f.upper is None else float(f.upper),
            )
        )
    return out


def _binary_baselines(
    question: LLMQuestion, data: BinaryData, as_of: date
) -> list[Baseline]:
    q = question.as_binary(as_of + timedelta(days=1))
    models: list[tuple[str, BinaryForecaster]] = (
        [
            ("market price (Polymarket)", MarketPrice()),
            ("category base rate", CuratedBaseRate()),
        ]
        if question.curated
        else [("historical base rate", ThresholdBaseRate())]
    )
    out = []
    for name, model in models:
        try:
            p = model.predict_probability(q, data)["yes"]
        except NoForecast:
            continue
        out.append(Baseline(name, probability=p))
    return out


def _documents(
    conn: Connection, query: str, as_of: date, quotas: Mapping[str, int]
) -> list[Document]:
    since = as_of - timedelta(days=DOCUMENT_DAYS)
    docs: list[Document] = []
    for document_type, n in quotas.items():
        if n > 0:
            docs += retrieve_documents(
                conn, as_of, document_type, n, query=query, since=since
            )
    return docs


def build_context_bundle(
    conn: Connection,
    question: LLMQuestion,
    as_of_date: date,
    *,
    store: VintageStore | None = None,
    binary_data: BinaryData | None = None,
    quotas: Mapping[str, int] = DOCUMENT_QUOTAS,
    baselines: bool = True,
) -> ContextBundle:
    """Everything the prompt shows for `question`, as known on `as_of_date`.

    Pass a shared `store` / `binary_data` when building many bundles.
    """
    store = store or store_from_conn(conn)
    series_id = question.series_id
    name = units = None
    recent: list[tuple[date, float]] = []
    if series_id is not None:
        name, units = _series_info(conn, series_id)
        recent = _recent(store, series_id, as_of_date)
    found: list[Baseline] = []
    if baselines:
        if question.binary:
            data = binary_data or BinaryData.from_db(conn)
            found = _binary_baselines(question, data, as_of_date)
        elif series_id is not None:
            found = _numeric_baselines(
                store, series_id, as_of_date, question.target_date
            )
    query = (
        question.question_text or ""
        if series_id is None
        else SERIES_QUERIES.get(series_id, name or series_id)
    )
    return ContextBundle(
        question, as_of_date, series_id, name, units, recent, found,
        _documents(conn, query, as_of_date, quotas) if query else [],
    )  # fmt: skip


def _series_info(conn: Connection, series_id: str) -> tuple[str | None, str | None]:
    with conn.cursor() as cur:
        cur.execute("SELECT name, units FROM series WHERE series_id = %s", (series_id,))
        row = cur.fetchone()
    if row is None:
        kind = kind_of(series_id)
        return series_id, {"percent": "Percent", "log10": "log10(FLOP)"}.get(
            kind, "count"
        )
    return row[0], row[1]

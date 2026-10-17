"""Baselines for the AI questions (AI task 13).

Numeric (the harness's `Forecaster`, on `ai.targets.AI_TARGETS`):

    bt = WalkForwardBacktest(date(2023, 1, 1), date(2026, 6, 1), "monthly")
    bt.run(AIRandomWalk(), AI_TARGETS["ARXIV_AI_SUBMISSIONS"], horizon_months=3)

- `AIRandomWalk`: the last known value. The interval is the `coverage`
  quantiles of past changes over the same number of months.
- `AIHistoricalMean`: the mean of the last `window_months` (12) values, with a
  normal interval.
- `AILinearTrend`: an OLS line through the last `window_months` (12) values,
  projected to the target month, with an OLS prediction interval.
Each follows the series' kind (`ai.targets.kind_of`). Counts are modelled on
a log scale: changes are ratios, the mean is geometric, the trend is
exponential growth. Running maxima (SOTA, compute) never fall, so the random
walk and the trend are floored at the last value, and percentages are capped
at 100.

Binary (`predict_probability(question, data) -> {"yes": p, "no": 1 - p}`):

    data = BinaryData.from_db(conn)
    evaluate_binary(MarketPrice(), binary_questions(conn), data)

- `ThresholdBaseRate`: for 'SERIES>=T' questions, how often the series has
  cleared the same relative step over the same gap. If T was set 10% above
  the base month's value, that's the share of past months m whose value h + 1
  months later was at least 10% above m's. Gap steps (SOTA) and log10 steps
  (compute) are compared the same way they were set (AI task 10).
- `CuratedBaseRate`: for Polymarket questions, the share of markets in the same
  category (`curated_questions.classify`) that resolved yes before the ask date.
- `MarketPrice` (the task's CommunityForecast): the market's price the day
  before the ask date. Curated questions only.
`BaseRate` routes each question to the right base rate. Rates are smoothed,
(yes + 1) / (n + 2), so a short record doesn't give 0 or 1. A question a model
can't answer raises `NoForecast`; `evaluate_binary` records it as missing.

Everything is point-in-time. Forecasts for a question asked on day D use data
published by the end of D - 1, the harness's convention. A threshold
question is asked on the day it was created; a curated one `lead_days`
before its market closes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import NormalDist
from typing import Protocol

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.curated_questions import MarketRow, classify, platform_markets
from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.resolutions import parse_threshold_key
from eco_prediction.ai.targets import Kind, kind_of
from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import Forecast, months_between
from eco_prediction.models.baselines import NotFittedError, _check_coverage

WINDOW_MONTHS = 12
MIN_TREND_POINTS = 3
TOLERANCE = 1e-9  # relative, for comparing a value with a restated threshold


# --- numeric ------------------------------------------------------------------


def _bounded(kind: Kind, last: float, f: Forecast) -> Forecast:
    """Clip a running maximum's forecast to [last, 100 if a percentage]."""
    if kind == "count":
        return f
    high = 100.0 if kind == "percent" else math.inf

    def clip(x: float | None) -> float | None:
        # float(): numpy floats can't be stored by psycopg2 (AI_NOTES/Task21.md)
        return None if x is None else float(min(max(x, last), high))

    return Forecast(clip(f.point), clip(f.lower), clip(f.upper))  # type: ignore[arg-type]


def _history(data: PointInTimeData) -> pd.Series:
    history = data.target_history()
    if history.empty:
        raise ValueError(f"no {data.target.name} data known by {data.cutoff}")
    return history


class AIRandomWalk:
    """The last known value, with an interval from past same-length changes."""

    def __init__(self, coverage: float = 0.8) -> None:
        self.coverage = _check_coverage(coverage)
        self.history: pd.Series | None = None

    def fit(self, data: PointInTimeData) -> None:
        self.history = data.target_history().asfreq("MS")

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.history is None:
            raise NotFittedError("call fit() first")
        kind = kind_of(data.target.series_id)
        latest = _history(data)
        last = float(latest.iloc[-1])
        steps = months_between(latest.index[-1].date(), target_period)
        h = self.history
        if kind == "count":
            changes = (np.log(h) - np.log(h.shift(steps))).dropna()
        else:
            changes = (h - h.shift(steps)).dropna()
        if steps < 1 or changes.empty:
            return Forecast(last)
        tail = (1 - self.coverage) / 2
        low, high = np.quantile(changes.to_numpy(), [tail, 1 - tail])
        if kind == "count":
            return Forecast(last, last * math.exp(low), last * math.exp(high))
        return _bounded(kind, last, Forecast(last, last + low, last + high))


class AIHistoricalMean:
    """The mean of the last `window_months` values (geometric for counts)."""

    def __init__(
        self, window_months: int = WINDOW_MONTHS, coverage: float = 0.8
    ) -> None:
        if window_months < 2:
            raise ValueError("window_months must be at least 2")
        self.window_months = window_months
        self.coverage = _check_coverage(coverage)
        self.z = NormalDist().inv_cdf(0.5 + coverage / 2)
        self.forecast: Forecast | None = None

    def fit(self, data: PointInTimeData) -> None:
        kind = kind_of(data.target.series_id)
        recent = _history(data).iloc[-self.window_months :]
        if len(recent) < 2:
            raise ValueError(f"need at least 2 {data.target.name} values to fit")
        values = np.log(recent) if kind == "count" else recent
        mean, half = float(values.mean()), self.z * float(values.std())
        if kind == "count":
            self.forecast = Forecast(
                math.exp(mean), math.exp(mean - half), math.exp(mean + half)
            )
        else:
            high = 100.0 if kind == "percent" else math.inf
            self.forecast = Forecast(
                min(mean, high), min(mean - half, high), min(mean + half, high)
            )

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if self.forecast is None:
            raise NotFittedError("call fit() first")
        return self.forecast


class AILinearTrend:
    """An OLS line through the last `window_months` values, projected forward.

    Counts are fit on the log scale (constant growth). The line is re-fit on
    each forecast's data, so it always ends at the latest known month. The
    interval is the OLS prediction interval with a normal quantile.
    """

    def __init__(
        self, window_months: int = WINDOW_MONTHS, coverage: float = 0.8
    ) -> None:
        if window_months < MIN_TREND_POINTS:
            raise ValueError(f"window_months must be at least {MIN_TREND_POINTS}")
        self.window_months = window_months
        self.coverage = _check_coverage(coverage)
        self.z = NormalDist().inv_cdf(0.5 + coverage / 2)
        self.fitted = False

    def fit(self, data: PointInTimeData) -> None:
        if len(_history(data)) < MIN_TREND_POINTS:
            raise ValueError(
                f"need at least {MIN_TREND_POINTS} {data.target.name} values to fit"
            )
        self.fitted = True

    def predict(self, data: PointInTimeData, target_period: date) -> Forecast:
        if not self.fitted:
            raise NotFittedError("call fit() first")
        kind = kind_of(data.target.series_id)
        history = _history(data)
        recent = history.iloc[-self.window_months :]
        last = float(history.iloc[-1])
        if len(recent) < MIN_TREND_POINTS:
            return Forecast(last)
        origin = recent.index[0].date()
        x = np.array([months_between(origin, d.date()) for d in recent.index], float)
        y = np.log(recent.to_numpy()) if kind == "count" else recent.to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        x0 = float(months_between(origin, target_period))
        point = intercept + slope * x0
        residuals = y - (intercept + slope * x)
        n = len(x)
        s = math.sqrt(float((residuals**2).sum()) / (n - 2)) if n > 2 else 0.0
        sxx = float(((x - x.mean()) ** 2).sum())
        se = s * math.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / sxx)
        low, high = point - self.z * se, point + self.z * se
        if kind == "count":
            return Forecast(math.exp(point), math.exp(low), math.exp(high))
        return _bounded(kind, last, Forecast(point, low, high))


# --- binary questions ------------------------------------------------------------


class NoForecast(Exception):
    """The model has nothing to say about this question (e.g. no market price)."""


@dataclass(frozen=True)
class BinaryQuestion:
    id: int | None
    question_key: str
    target_date: date
    ask_date: date  # forecasts use data published by the end of the day before
    horizon_months: int | None = None  # threshold questions
    platform_question_id: int | None = None  # curated questions
    title: str | None = None
    outcome: str | None = None  # 'yes'/'no' once resolved

    @property
    def curated(self) -> bool:
        return self.platform_question_id is not None

    @property
    def cutoff(self) -> date:
        return self.ask_date - timedelta(days=1)

    @property
    def category(self) -> str | None:
        return classify(self.title) if self.curated and self.title else None


def binary_questions(conn: Connection) -> list[BinaryQuestion]:
    """Every binary question in the database, with its outcome if resolved."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, q.question_key, q.target_date, q.horizon_months,
                   q.lead_days, q.platform_question_id, q.question_text,
                   -- The jobs' clock is US Eastern; evening creations are a
                   -- day later in UTC (AI_NOTES/Task21.md).
                   (q.created_at AT TIME ZONE 'America/New_York')::date,
                   r.actual_outcome
            FROM questions q LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE q.question_type = 'binary'
            ORDER BY q.target_date, q.id
            """
        )
        rows = cur.fetchall()
    out = []
    for qid, key, target, horizon, lead, pq, text, created, outcome in rows:
        ask = target - timedelta(days=lead) if lead is not None else created
        out.append(BinaryQuestion(qid, key, target, ask, horizon, pq, text, outcome))
    return out


class BinaryData:
    """What binary baselines may read, each as known on a cutoff date.

    `store` holds the AI series; `markets` the Polymarket markets
    (`platform_questions`); `prices(platform_question_id)` a market's daily
    probabilities, indexed by date.
    """

    def __init__(
        self,
        store: VintageStore,
        markets: Sequence[MarketRow] = (),
        prices: Callable[[int], pd.Series] | None = None,
    ) -> None:
        self.store = store
        self.markets = list(markets)
        self._prices = prices
        self._price_cache: dict[int, pd.Series] = {}
        self._categories = {m.id: classify(m.title) for m in self.markets}

    @classmethod
    def from_db(cls, conn: Connection) -> BinaryData:
        def prices(platform_question_id: int) -> pd.Series:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT as_of, probability::float8 FROM platform_probabilities"
                    " WHERE question_id = %s ORDER BY as_of",
                    (platform_question_id,),
                )
                rows = cur.fetchall()
            return pd.Series(
                [p for _, p in rows],
                index=pd.DatetimeIndex([d for d, _ in rows]),
                dtype="float64",
            )

        return cls(store_from_conn(conn), platform_markets(conn), prices)

    def series(self, series_id: str, cutoff: date) -> pd.Series:
        known = self.store.known(series_id, cutoff)
        return pd.Series(
            known["value"].to_numpy(),
            index=pd.DatetimeIndex(known["observed_at"]),
            dtype="float64",
        ).asfreq("MS")

    def price(self, platform_question_id: int, cutoff: date) -> float | None:
        """The market's latest daily price on or before cutoff."""
        if self._prices is None:
            return None
        if platform_question_id not in self._price_cache:
            self._price_cache[platform_question_id] = self._prices(platform_question_id)
        prices = self._price_cache[platform_question_id]
        known = prices[prices.index <= pd.Timestamp(cutoff)]
        return None if known.empty else float(known.iloc[-1])

    def resolved_markets(self, category: str, cutoff: date) -> list[MarketRow]:
        """Markets in `category` that resolved yes or no by cutoff."""
        return [
            m
            for m in self.markets
            if self._categories[m.id] == category
            and m.resolved_outcome is not None
            and m.resolved_at is not None
            and m.resolved_at.date() <= cutoff
        ]


def _probabilities(p: float) -> dict[str, float]:
    return {"yes": p, "no": 1 - p}


def smoothed_rate(yes: int, n: int) -> float:
    return (yes + 1) / (n + 2)


class BinaryForecaster(Protocol):
    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]: ...


def cleared(
    kind: Kind, base: float, threshold: float
) -> Callable[[float, float], bool]:
    """A test of `later >= threshold`, restated relative to `base`, so it can be
    applied to any earlier value: did `later` clear the same step from `start`?

    Equality counts as clearing, within float rounding (TOLERANCE, relative).
    """
    if kind == "count":
        ratio = threshold / base

        def bound(start: float) -> float:
            return start * ratio
    elif kind == "percent":
        gap = 100 - base
        share = (threshold - base) / gap if gap > 0 else 1.0

        def bound(start: float) -> float:
            return start + share * (100 - start)
    else:
        step = threshold - base

        def bound(start: float) -> float:
            return start + step

    def test(start: float, later: float) -> bool:
        b = bound(start)
        return later >= b - TOLERANCE * max(1.0, abs(b))

    return test


class ThresholdBaseRate:
    """How often the series cleared the question's relative step historically."""

    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]:
        if question.curated or question.horizon_months is None:
            raise NoForecast("not a threshold question")
        series_id, threshold = parse_threshold_key(question.question_key)
        kind = kind_of(series_id)
        series = data.series(series_id, question.cutoff)
        # The threshold was set from the month before the asking month.
        ask_month = pd.Timestamp(question.target_date) - pd.DateOffset(
            months=question.horizon_months
        )
        base = series.get(ask_month - pd.DateOffset(months=1))
        if base is None or pd.isna(base):
            raise NoForecast(f"{series_id}: base month not known by {question.cutoff}")
        test = cleared(kind, float(base), threshold)
        gap = question.horizon_months + 1
        later = series.shift(-gap)
        pairs = pd.DataFrame({"start": series, "later": later}).dropna()
        hits = sum(test(s, v) for s, v in zip(pairs["start"], pairs["later"]))
        return _probabilities(smoothed_rate(int(hits), len(pairs)))


class CuratedBaseRate:
    """The yes rate of resolved markets in the question's category."""

    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]:
        if not question.curated or question.category is None:
            raise NoForecast("not a curated question")
        resolved = data.resolved_markets(question.category, question.cutoff)
        yes = sum(m.resolved_outcome == "yes" for m in resolved)
        return _probabilities(smoothed_rate(yes, len(resolved)))


class BaseRate:
    """The threshold or curated base rate, whichever fits the question."""

    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]:
        model: BinaryForecaster = (
            CuratedBaseRate() if question.curated else ThresholdBaseRate()
        )
        return model.predict_probability(question, data)


class MarketPrice:
    """The Polymarket price the day before the ask date (the community forecast)."""

    def predict_probability(
        self, question: BinaryQuestion, data: BinaryData
    ) -> dict[str, float]:
        if question.platform_question_id is None:
            raise NoForecast("no market for this question")
        price = data.price(question.platform_question_id, question.cutoff)
        if price is None:
            raise NoForecast(f"no price by {question.cutoff}")
        return _probabilities(price)


def evaluate_binary(
    model: BinaryForecaster,
    questions: Iterable[BinaryQuestion],
    data: BinaryData,
) -> pd.DataFrame:
    """One row per question: p_yes (NaN if the model had none), outcome, Brier."""
    rows = []
    for q in questions:
        try:
            p: float | None = model.predict_probability(q, data)["yes"]
        except NoForecast:
            p = None
        brier = (
            metrics.brier_score_binary(p, q.outcome)
            if p is not None and q.outcome is not None
            else None
        )
        rows.append(
            {
                "question_id": q.id,
                "question_key": q.question_key,
                "category": q.category,
                "ask_date": q.ask_date,
                "p_yes": p,
                "outcome": q.outcome,
                "brier": brier,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "question_id", "question_key", "category", "ask_date", "p_yes",
            "outcome", "brier",
        ],
    )  # fmt: skip
    frame[["p_yes", "brier"]] = frame[["p_yes", "brier"]].astype("float64")
    return frame


__all__ = [
    "AIHistoricalMean",
    "AILinearTrend",
    "AIRandomWalk",
    "BaseRate",
    "BinaryData",
    "BinaryQuestion",
    "CuratedBaseRate",
    "MarketPrice",
    "NoForecast",
    "ThresholdBaseRate",
    "binary_questions",
    "evaluate_binary",
]

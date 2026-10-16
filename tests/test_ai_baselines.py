"""Tests for the AI baselines (AI task 13).

Most numeric cases use an arXiv series that grows exactly 2% a month from
1,000 in 2023-01 to 2025-12, each month known on the 1st of the next, so
every expected value has a closed form: month i (0 = 2023-01) is 1000 * 1.02^i.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.baselines import (
    AIHistoricalMean,
    AILinearTrend,
    AIRandomWalk,
    BaseRate,
    BinaryData,
    BinaryQuestion,
    CuratedBaseRate,
    MarketPrice,
    NoForecast,
    ThresholdBaseRate,
    binary_questions,
    cleared,
    evaluate_binary,
)
from eco_prediction.ai.curated_questions import MarketRow
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import (
    PlatformProbability,
    PlatformQuestion,
    insert_platform_probabilities,
    upsert_platform_question,
)
from eco_prediction.ai.targets import AI_TARGETS, kind_of
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest, summarize
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import resolve_binary_question
from eco_prediction.models.baselines import NotFittedError

ARXIV = "ARXIV_AI_SUBMISSIONS"
GPQA = "EPOCH_GPQA_SOTA"
FLOPS = "EPOCH_LARGEST_FLOPS"


def arxiv_value(i: int) -> float:
    return 1000 * 1.02**i


def monthly(start: date, values: list[float]) -> pd.DataFrame:
    months = list(iter_months(start, date(2100, 1, 1)))[: len(values)]
    return pd.DataFrame(
        [(m, next_month(m), v) for m, v in zip(months, values)],
        columns=["observed_at", "as_of", "value"],
    )


def build_store(**extra: pd.DataFrame) -> VintageStore:
    return VintageStore(
        {ARXIV: monthly(date(2023, 1, 1), [arxiv_value(i) for i in range(36)]), **extra}
    )


def view(store: VintageStore, series_id: str, cutoff: date) -> PointInTimeData:
    return PointInTimeData(store, AI_TARGETS[series_id], cutoff)


CUTOFF = date(2026, 1, 14)  # knows through 2025-12 (i = 35)
MAR = date(2026, 3, 1)  # i = 38, three months after the latest


def fitted(model: Any, data: PointInTimeData) -> Any:
    model.fit(data)
    return model


# --- targets ------------------------------------------------------------------


def test_targets_and_kinds() -> None:
    assert len(AI_TARGETS) == 7
    assert AI_TARGETS[ARXIV].series_id == ARXIV
    assert (kind_of(ARXIV), kind_of(GPQA), kind_of(FLOPS)) == (
        "count",
        "percent",
        "log10",
    )
    with pytest.raises(ValueError):
        kind_of("GITHUB_AI_STARS")


# --- numeric baselines -----------------------------------------------------------


def test_random_walk_returns_the_last_value() -> None:
    data = view(build_store(), ARXIV, CUTOFF)
    f = fitted(AIRandomWalk(), data).predict(data, MAR)
    assert f.point == pytest.approx(arxiv_value(35))
    # Every past 3-month change is a ratio of 1.02^3, so the interval is that.
    assert f.lower == pytest.approx(arxiv_value(38))
    assert f.upper == pytest.approx(arxiv_value(38))


def test_random_walk_on_a_running_max_stays_between_last_and_100() -> None:
    # Past 2-month changes are +5, 0 and +5: 98 + 5 is capped at 100.
    values = [88.0, 93.0, 93.0, 93.0, 98.0]
    store = build_store(**{GPQA: monthly(date(2025, 8, 1), values)})
    data = view(store, GPQA, CUTOFF)
    f = fitted(AIRandomWalk(), data).predict(data, date(2026, 2, 1))
    assert f.point == 98.0 and f.upper == 100.0
    assert 98.0 <= f.lower <= 100.0  # type: ignore[operator]


def test_historical_mean_is_geometric_for_counts() -> None:
    data = view(build_store(), ARXIV, CUTOFF)
    f = fitted(AIHistoricalMean(), data).predict(data, MAR)
    # The mean of log values for i = 24..35 is the log of 1000 * 1.02^29.5.
    assert f.point == pytest.approx(arxiv_value(29.5))  # type: ignore[arg-type]
    assert f.lower < f.point < f.upper  # type: ignore[operator]


def test_historical_mean_for_a_step_series() -> None:
    store = build_store(**{GPQA: monthly(date(2025, 9, 1), [60.0, 60.0, 70.0, 70.0])})
    data = view(store, GPQA, CUTOFF)
    f = fitted(AIHistoricalMean(window_months=4), data).predict(data, MAR)
    assert f.point == pytest.approx(65.0)


def test_linear_trend_projects_exponential_growth_exactly() -> None:
    data = view(build_store(), ARXIV, CUTOFF)
    f = fitted(AILinearTrend(), data).predict(data, MAR)
    assert f.point == pytest.approx(arxiv_value(38))
    assert f.lower == pytest.approx(arxiv_value(38))  # no residuals, no width


def test_linear_trend_on_levels_is_capped_at_100() -> None:
    values = [80.0, 84.0, 88.0, 92.0, 96.0]  # +4 a month
    store = build_store(**{GPQA: monthly(date(2025, 8, 1), values)})
    data = view(store, GPQA, CUTOFF)
    model = fitted(AILinearTrend(), data)
    assert model.predict(data, date(2026, 1, 1)).point == pytest.approx(100.0)
    flops = build_store(**{FLOPS: monthly(date(2025, 9, 1), [26.0, 26.1, 26.2, 26.3])})
    f = fitted(AILinearTrend(), view(flops, FLOPS, CUTOFF)).predict(
        view(flops, FLOPS, CUTOFF), MAR
    )
    assert f.point == pytest.approx(26.6)  # unbounded above


def test_models_need_data_and_fitting() -> None:
    empty = view(build_store(), GPQA, CUTOFF)
    for model in (AIRandomWalk(), AIHistoricalMean(), AILinearTrend()):
        with pytest.raises(NotFittedError):
            model.predict(view(build_store(), ARXIV, CUTOFF), MAR)
    with pytest.raises(ValueError, match="no EPOCH_GPQA_SOTA data"):
        fitted(AIRandomWalk(), empty).predict(empty, MAR)
    with pytest.raises(ValueError):
        AIHistoricalMean().fit(empty)


def test_baselines_run_through_the_harness() -> None:
    bt = WalkForwardBacktest(
        date(2024, 6, 1), date(2025, 6, 1), "monthly", store=build_store()
    )
    trend = summarize(bt.run(AILinearTrend(), AI_TARGETS[ARXIV], horizon_months=3))
    walk = summarize(bt.run(AIRandomWalk(), AI_TARGETS[ARXIV], horizon_months=3))
    mean = summarize(bt.run(AIHistoricalMean(), AI_TARGETS[ARXIV], horizon_months=3))
    assert trend["n_resolved"] == walk["n_resolved"] == 13
    assert trend["rmse"] == pytest.approx(0, abs=1e-6)  # the series is a pure trend
    assert 0 < walk["rmse"] < mean["rmse"]  # type: ignore[operator]


# --- the threshold base rate --------------------------------------------------------


def test_cleared_restates_each_kind_of_step() -> None:
    assert cleared("count", 100, 110)(200, 220) and not cleared("count", 100, 110)(
        200, 219
    )
    # 25% of the gap: 80 -> 85; from 60 the same step is 70.
    assert cleared("percent", 80, 85)(60, 70) and not cleared("percent", 80, 85)(
        60, 69.9
    )
    assert cleared("log10", 26.0, 26.3)(27.0, 27.3)


def threshold_question(threshold: float, horizon: int = 1) -> BinaryQuestion:
    # Asked 2026-01-15 about 2026-02 (horizon 1), from 2025-12's value.
    return BinaryQuestion(
        None,
        f"{ARXIV}>={threshold}",
        date(2026, 1 + horizon, 1),
        date(2026, 1, 15),
        horizon_months=horizon,
    )


def test_threshold_base_rate_counts_past_clears() -> None:
    data = BinaryData(build_store())
    base = arxiv_value(35)
    # A two-month gap always grows 1.02^2 = 4.04%: +10% never, +3% always.
    # There are 34 month pairs two months apart, smoothed: 1/36 and 35/36.
    never = ThresholdBaseRate().predict_probability(
        threshold_question(base * 1.10), data
    )
    always = ThresholdBaseRate().predict_probability(
        threshold_question(base * 1.03), data
    )
    assert never["yes"] == pytest.approx(1 / 36)
    assert always["yes"] == pytest.approx(35 / 36)
    assert never["yes"] + never["no"] == 1


def test_threshold_base_rate_needs_its_base_month() -> None:
    early = BinaryQuestion(
        None, f"{ARXIV}>=2000", date(2026, 2, 1), date(2026, 1, 1), horizon_months=1
    )  # cutoff 2025-12-31: 2025-12 isn't known yet
    with pytest.raises(NoForecast, match="base month"):
        ThresholdBaseRate().predict_probability(early, BinaryData(build_store()))


# --- curated base rate and market price ------------------------------------------


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 12, tzinfo=UTC)


def market(
    id: int, title: str, outcome: str | None, resolved: datetime | None
) -> MarketRow:
    return MarketRow(
        id, str(id), title, None, None, utc(2026, 1, 1), utc(2026, 12, 31),
        resolved, outcome, {},
    )  # fmt: skip


RELEASE = "GPT-7 released by December 31, 2026?"
MARKETS = [
    market(1, "Claude 6 released by June 30?", "yes", utc(2026, 6, 1)),
    market(2, "Gemini 5 released by May 31?", "yes", utc(2026, 5, 1)),
    market(3, "Grok 6 released by April 30?", "yes", utc(2026, 4, 1)),
    market(4, "DeepSeek V5 released by March 31?", "no", utc(2026, 3, 31)),
    market(5, "Llama 6 released by September 30?", "yes", utc(2026, 9, 30)),  # later
    market(6, "ChatGPT Full Outage by March 31?", "yes", utc(2026, 3, 1)),  # other kind
    market(7, RELEASE, None, None),
]
PRICES = {
    7: pd.Series(
        [0.30, 0.42, 0.55],
        index=pd.to_datetime(["2026-08-30", "2026-08-31", "2026-09-01"]),
    )
}


def curated(
    ask: date = date(2026, 9, 1), outcome: str | None = "yes"
) -> BinaryQuestion:
    return BinaryQuestion(
        7, "polymarket:7", date(2026, 12, 31), ask, platform_question_id=7,
        title=RELEASE, outcome=outcome,
    )  # fmt: skip


def curated_data() -> BinaryData:
    return BinaryData(
        build_store(), MARKETS, lambda pk: PRICES.get(pk, pd.Series(dtype="float64"))
    )


def test_curated_base_rate_uses_markets_resolved_before_the_ask_date() -> None:
    # Releases resolved by 2026-08-31: 3 yes, 1 no -> (3 + 1) / (4 + 2).
    p = CuratedBaseRate().predict_probability(curated(), curated_data())
    assert p["yes"] == pytest.approx(4 / 6)
    assert BaseRate().predict_probability(curated(), curated_data()) == p


def test_market_price_is_the_day_before_the_ask_date() -> None:
    p = MarketPrice().predict_probability(curated(), curated_data())
    assert p == {"yes": 0.42, "no": pytest.approx(0.58)}
    with pytest.raises(NoForecast):
        MarketPrice().predict_probability(curated(ask=date(2026, 8, 1)), curated_data())
    with pytest.raises(NoForecast):
        MarketPrice().predict_probability(threshold_question(2000.0), curated_data())


def test_evaluate_binary_scores_what_each_model_can_answer() -> None:
    questions = [curated(), threshold_question(arxiv_value(35) * 1.03)]
    frame = evaluate_binary(MarketPrice(), questions, curated_data())
    assert frame["p_yes"].tolist()[0] == 0.42
    assert math.isnan(frame["p_yes"].tolist()[1])  # no market for a threshold
    assert frame["brier"].tolist()[0] == pytest.approx((0.42 - 1) ** 2)
    rates = evaluate_binary(BaseRate(), questions, curated_data())
    assert rates["p_yes"].notna().all()
    assert rates["category"].iloc[0] == "release"
    assert pd.isna(rates["category"].iloc[1])


# --- from the database -------------------------------------------------------------


def test_binary_questions_and_data_from_the_database(conn: Connection) -> None:
    migrate(conn)
    pk, _, _ = upsert_platform_question(
        conn,
        PlatformQuestion(
            "polymarket", "7", RELEASE, opened_at=utc(2026, 1, 1),
            closes_at=utc(2026, 12, 31),
        ),
    )  # fmt: skip
    insert_platform_probabilities(
        conn, pk, [PlatformProbability(date(2026, 11, 30), 0.6, True)]
    )
    curated_id, _ = get_or_create_question(
        conn, "ai_binary_event", date(2026, 12, 31), lead_days=30,
        question_key="polymarket:7", question_text=RELEASE, platform_question_id=pk,
    )  # fmt: skip
    get_or_create_question(
        conn, "ai_binary_event", date(2026, 11, 1), 1, question_key=f"{ARXIV}>=33000"
    )
    resolve_binary_question(conn, curated_id, "no", date(2026, 12, 31))

    questions = {q.question_key: q for q in binary_questions(conn)}
    c = questions["polymarket:7"]
    assert (c.ask_date, c.outcome, c.category, c.curated) == (
        date(2026, 12, 1), "no", "release", True,
    )  # fmt: skip
    t = questions[f"{ARXIV}>=33000"]
    assert (t.horizon_months, t.outcome, t.curated) == (1, None, False)
    assert (
        abs((t.ask_date - datetime.now(UTC).date()).days) <= 1
    )  # created now (UTC date)

    data = BinaryData.from_db(conn)
    assert MarketPrice().predict_probability(c, data)["yes"] == pytest.approx(0.6)

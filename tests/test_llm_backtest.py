"""Tests for the leakage-safe LLM backtest and the ensemble (AI task 19).

The Anthropic API is never called: a fake client answers each request from
its prompt. Database tests use a throwaway database (skipped if Postgres
isn't reachable).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.features import store_from_conn
from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV_SERIES
from eco_prediction.ai.llm.backtest import (
    BacktestBudget,
    BudgetExceeded,
    LLMNumericForecaster,
    run_llm_backtest,
    training_cutoff,
)
from eco_prediction.ai.llm.claude_forecaster import MODEL, ClaudeForecaster
from eco_prediction.ai.llm.ensemble import LLMEnsembleForecaster, simplex_grid
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.store import (
    AIObservation,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.data import PointInTimeData, VintageStore
from eco_prediction.backtest.harness import WalkForwardBacktest
from eco_prediction.data.ingest import ensure_source
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate

ARXIV = "ARXIV_AI_SUBMISSIONS"
CUTOFF = date(2026, 7, 1)


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


# --- the training cutoff ---------------------------------------------------------------


def monthly_store() -> VintageStore:
    months = list(iter_months(date(2020, 1, 1), date(2026, 9, 1)))
    frame = pd.DataFrame(
        [(m, next_month(m), 1000.0 * 1.01**i) for i, m in enumerate(months)],
        columns=["observed_at", "as_of", "value"],
    )
    return VintageStore({ARXIV: frame})


def test_the_harness_skips_forecast_dates_before_the_training_cutoff() -> None:
    store = monthly_store()
    plain = WalkForwardBacktest(
        date(2026, 1, 1), date(2026, 9, 1), "monthly", store=store
    )
    assert plain.get_forecast_dates()[0] == date(2026, 1, 1)  # unchanged without one
    bt = WalkForwardBacktest(
        date(2026, 1, 1),
        date(2026, 9, 1),
        "monthly",
        store=store,
        model_training_cutoff=CUTOFF,
    )
    assert bt.get_forecast_dates() == [date(2026, m, 1) for m in (7, 8, 9)]
    mid = WalkForwardBacktest(
        date(2026, 1, 1), date(2026, 9, 1), "monthly", store=store,
        model_training_cutoff=date(2026, 7, 15),
    )  # fmt: skip
    assert mid.get_forecast_dates()[0] == date(2026, 8, 1)


def test_cutoffs_are_known_per_model() -> None:
    assert training_cutoff(MODEL) == date(2026, 7, 1)
    assert training_cutoff("claude-haiku-5-5") == date(2026, 7, 1)
    with pytest.raises(ValueError, match="pass cutoff_date"):
        training_cutoff("claude-unknown")


def test_the_budget_counts_new_spending_only() -> None:
    budget = BacktestBudget(0.10)
    budget.record({"cost_usd": 0.06})
    budget.record({"cached": True, "cost_usd": 0.06})
    budget.check()
    budget.record({"cost_usd": 0.05})
    assert (budget.calls, budget.reused, budget.spent_usd) == (
        2,
        1,
        pytest.approx(0.11),
    )
    with pytest.raises(BudgetExceeded, match=r"\$0.11 of a \$0.10"):
        budget.check()
    BacktestBudget(None).check()  # no limit


# --- the ensemble ---------------------------------------------------------------------------


def test_the_simplex_grid() -> None:
    grid = simplex_grid(3, 0.5)
    assert np.allclose(grid.sum(axis=1), 1) and len(grid) == 6
    assert len(simplex_grid(2)) == 21


def history(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """'good' is close to the outcome, 'bad' is noise."""
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n).astype(float))
    good = (0.85 * y + 0.075 + rng.normal(0, 0.03, n)).clip(0, 1)
    bad = pd.Series(rng.uniform(0, 1, n))
    return pd.DataFrame({"good": good, "bad": bad}), y


def test_weights_by_method() -> None:
    p, y = history(40)
    assert LLMEnsembleForecaster("equal").weights(p, y, binary=True) == {
        "good": 0.5,
        "bad": 0.5,
    }
    few = LLMEnsembleForecaster("learned").weights(p.head(5), y.head(5), binary=True)
    assert few == {"good": 0.5, "bad": 0.5}  # under min_history
    learned = LLMEnsembleForecaster("learned").weights(p, y, binary=True)
    assert learned["good"] >= 0.9
    inv = LLMEnsembleForecaster("inverse_variance").weights(p, y, binary=True)
    assert inv["good"] > 0.8 and inv["good"] + inv["bad"] == pytest.approx(1)
    with pytest.raises(ValueError):
        LLMEnsembleForecaster("stacking")  # type: ignore[arg-type]


def test_missing_components_are_renormalized() -> None:
    combine = LLMEnsembleForecaster.combine
    assert combine({"a": 0.2, "b": 0.6}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.4)
    assert combine({"a": 0.2, "b": None}, {"a": 0.25, "b": 0.75}) == pytest.approx(0.2)
    assert combine({"a": None}, {"a": 1.0}) is None
    assert combine({"a": 0.2, "b": 0.4}, {"a": 0.0, "b": 0.0}) == pytest.approx(0.3)


def long_binary(n: int) -> pd.DataFrame:
    p, y = history(n, seed=1)
    rows = []
    for i in range(n):
        ask = pd.Timestamp(2026, 7, 1) + pd.Timedelta(days=i)
        for c in ("good", "bad"):
            rows.append({
                "question_id": i, "ask_date": ask.date(), "component": c, "p_yes": p.at[i, c],
                "outcome": "yes" if y[i] else "no",
                "resolved_at": ask + pd.Timedelta(days=2),  # known two days after asking
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_binary_weights_are_learned_walk_forward() -> None:
    frame = long_binary(60)
    out = LLMEnsembleForecaster("learned").walk_forward_binary(frame)
    assert len(out) == 60
    # Question i sees only questions resolved before its ask date: i - 2 of them.
    assert out["n_history"].tolist()[:5] == [0, 0, 0, 1, 2]
    assert out["weights"].iloc[0] == {"bad": 0.5, "good": 0.5}
    assert out["weights"].iloc[-1]["good"] >= 0.9
    # Better than equal weights throughout. It matches the best component
    # only once warmed up: until min_history outcomes are known it uses
    # equal weights, and pays for the noisy component (the cold start).
    good = frame[frame.component == "good"].set_index("question_id")
    good_brier = (good["p_yes"] - (good["outcome"] == "yes")) ** 2
    equal = LLMEnsembleForecaster("equal").walk_forward_binary(frame)["brier"]
    assert out["brier"].mean() < equal.mean()
    warm = out["n_history"] >= 10
    assert out.loc[warm, "brier"].mean() < good_brier[warm.to_numpy()].mean() + 0.005
    assert out.loc[~warm, "brier"].mean() == pytest.approx(equal[~warm].mean())


def test_numeric_weights_use_only_published_actuals() -> None:
    rows = []
    for i, m in enumerate(iter_months(date(2025, 1, 1), date(2026, 8, 1))):
        target = next_month(m)
        actual = 1000.0 + 10 * i
        for c, pred in (("close", actual * 1.01), ("far", actual * 1.3)):
            rows.append({
                "series_id": ARXIV, "forecast_date": m, "target_period": target,
                "data_as_of": m, "component": c, "prediction": pred,
                "lower": pred * 0.9, "upper": pred * 1.1, "actual": actual,
                "actual_as_of": next_month(target),
            })  # fmt: skip
    out = LLMEnsembleForecaster("learned", min_history=5).walk_forward_numeric(
        pd.DataFrame(rows)
    )
    assert (
        out["n_history"].iloc[0] == 0 and out["n_history"].iloc[3] == 2
    )  # two published
    last = out.iloc[-1]
    assert last["weights"]["close"] == 1.0
    assert last["prediction"] == pytest.approx(last["actual"] * 1.01)
    assert last["lower"] < last["prediction"] < last["upper"]


# --- the full run (database, fake API) -----------------------------------------------------


class AnsweringClient:
    """Answers from the prompt: 'happen' questions are likely, others aren't."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_on = fail_on
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "point_forecast" in json.dumps(kwargs["output_config"]):
            last = float(prompt.split("<recent_values")[1].split("</recent_values>")[0]
                         .strip().split("\n")[-1].split(": ")[1].replace(",", ""))  # fmt: skip
            text = json.dumps({"rationale": "trend", "point_forecast": last * 1.02,
                               "lower_80": last * 0.95, "upper_80": last * 1.1})  # fmt: skip
        else:
            question = prompt.split("<question>")[1].split("</question>")[0]
            if self.fail_on and self.fail_on in question:
                text = "not json"
            else:
                p = 0.9 if "happen" in question else 0.1
                text = json.dumps({"rationale": "evidence", "probability": p})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
            model=MODEL, stop_details=None,
            usage=SimpleNamespace(input_tokens=4000, output_tokens=1000, iterations=None,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )  # fmt: skip


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def seed_markets(conn: Connection) -> None:
    """Six markets closing in 2026, asked 30 days before; one asked before the cutoff."""
    specs = [
        ("m1", "Will event one happen by August 15?", date(2026, 8, 15), "yes"),
        ("m2", "Will event two fail by August 20?", date(2026, 8, 20), "no"),
        ("m3", "Will event three happen by September 1?", date(2026, 9, 1), "yes"),
        ("m4", "Will event four fail by September 10?", date(2026, 9, 10), "no"),
        ("m5", "Will event five happen by September 20?", date(2026, 9, 20), "yes"),
        (
            "m0",
            "Will event zero happen by June 20?",
            date(2026, 6, 20),
            "yes",
        ),  # asked in May
    ]
    for ext, title, close, outcome in specs:
        closes = datetime(close.year, close.month, close.day, tzinfo=UTC)
        pq, _, _ = upsert_platform_question(conn, PlatformQuestion(
            "polymarket", ext, f"OpenAI GPT model released? {title}", opened_at=utc(2026, 1, 1),
            closes_at=closes, resolved_at=closes, resolved_outcome=outcome,
        ))  # fmt: skip
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO platform_probabilities VALUES (%s, '2026-01-02', 0.5, false)",
                (pq,),
            )
        get_or_create_question(
            conn, "ai_binary_event", close, lead_days=30, question_key=f"polymarket:{ext}",
            question_text=title, platform_question_id=pq,
        )  # fmt: skip
    conn.commit()


def fake_forecaster(conn: Connection, client: AnsweringClient) -> ClaudeForecaster:
    return ClaudeForecaster(conn=conn, client=client)  # type: ignore[arg-type]


def test_binary_backtest_is_leakage_safe_saved_and_costed(migrated: Connection) -> None:
    seed_markets(migrated)
    client = AnsweringClient()
    result = run_llm_backtest(
        migrated, start_date=date(2026, 1, 1), end_date=date(2026, 10, 15),
        forecaster=fake_forecaster(migrated, client), numeric=False, budget_usd=5,
    )  # fmt: skip
    assert result.dropped_before_cutoff == 1  # m0 was asked 2026-05-21
    llm = result.binary[result.binary.component == "llm"]
    assert (
        len(llm) == 5
        and (pd.to_datetime(llm["ask_date"]) >= pd.Timestamp(CUTOFF)).all()
    )
    assert len(client.requests) == 5
    # Each prompt is as of the day before the ask date.
    assert "Today is 2026-07-16." in client.requests[0]["messages"][0]["content"]
    summary = result.summary
    # Opus 5.5: 4,000 in at $4, 1,000 out at $20 per million = $0.036 a call.
    assert summary["cost"] == {
        "new_calls": 5, "reused_calls": 0, "spent_usd": pytest.approx(0.18),
        "per_call_usd": pytest.approx(0.036), "failures": [],
    }  # fmt: skip
    brier = summary["binary"]
    assert brier["n_common"] == 5 and brier["llm"] == pytest.approx(0.01)
    assert brier["llm"] < brier["base_rate"] and brier["llm"] < brier["market"]
    assert brier["ensemble"] <= max(brier["llm"], brier["base_rate"], brier["market"])
    with migrated.cursor() as cur:
        cur.execute("SELECT count(*), bool_and(is_backtest) FROM forecasts")
        assert cur.fetchone() == (5, True)
        cur.execute("SELECT count(*) FROM llm_calls WHERE forecast_id IS NOT NULL")
        assert cur.fetchone()[0] == 5  # type: ignore[index]

    # A re-run pays for nothing and stores nothing twice.
    again_client = AnsweringClient()
    again = run_llm_backtest(
        migrated, start_date=date(2026, 1, 1), end_date=date(2026, 10, 15),
        forecaster=fake_forecaster(migrated, again_client), numeric=False, budget_usd=5,
    )  # fmt: skip
    assert again_client.requests == [] and again.summary["cost"]["reused_calls"] == 5
    with migrated.cursor() as cur:
        cur.execute("SELECT count(*) FROM forecasts")
        assert cur.fetchone()[0] == 5  # type: ignore[index]


def test_the_budget_stops_the_run_and_failures_are_counted(
    migrated: Connection,
) -> None:
    seed_markets(migrated)
    client = AnsweringClient(fail_on="event two")
    result = run_llm_backtest(
        migrated, start_date=date(2026, 1, 1), end_date=date(2026, 10, 15),
        forecaster=fake_forecaster(migrated, client), numeric=False, budget_usd=0.07,
        save=False,
    )  # fmt: skip
    # $0.036 a call: after two successes ($0.072) the next call is refused.
    failures = result.summary["cost"]["failures"]
    assert len(client.requests) == 3 and result.summary["cost"]["new_calls"] == 2
    assert any("malformed reply" in f for f in failures)
    assert failures[-1].startswith("spent $0.07 of a $0.07 budget")
    assert len(result.binary[result.binary.component == "llm"]) == 2  # kept


def seed_arxiv(conn: Connection) -> None:
    source = ensure_source(conn, *ARXIV_SERIES.source)
    series = upsert_ai_series(conn, source, ARXIV_SERIES.info)
    months = list(iter_months(date(2021, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(conn, series, [
        AIObservation(m, next_month(m), 4000 * 1.015**i, True) for i, m in enumerate(months)
    ])  # fmt: skip
    conn.commit()


def test_numeric_backtest_scores_only_published_targets(migrated: Connection) -> None:
    seed_arxiv(migrated)
    client = AnsweringClient()
    result = run_llm_backtest(
        migrated, start_date=date(2026, 1, 1), end_date=date(2026, 10, 15),
        forecaster=fake_forecaster(migrated, client), binary=False, series=[ARXIV],
        horizons=[1], budget_usd=5,
    )  # fmt: skip
    # Forecasts from 07-01 to 10-01 at 1 month; only Aug and Sep targets are published.
    assert len(client.requests) == 2
    llm = result.numeric[result.numeric.component == "llm"]
    assert sorted(llm["target_period"]) == [date(2026, 8, 1), date(2026, 9, 1)]
    assert set(result.numeric["component"]) == {
        "llm",
        "random_walk",
        "arima",
        "lightgbm",
    }
    table = result.summary["numeric"]
    assert (
        table["n_common"] == 2 and table["llm"] < table["random_walk"]
    )  # it adds the trend
    assert len(result.ensemble_numeric) == 2


def test_numeric_adapter_returns_nan_without_spending(migrated: Connection) -> None:
    seed_arxiv(migrated)
    client = AnsweringClient()
    budget = BacktestBudget(None)
    llm = LLMNumericForecaster(migrated, fake_forecaster(migrated, client), budget)
    view = PointInTimeData(
        store_from_conn(migrated), AI_TARGETS[ARXIV], date(2026, 9, 30)
    )
    assert math.isnan(
        llm.predict(view, date(2026, 11, 1)).point
    )  # October isn't published
    assert client.requests == [] and budget.calls == 0

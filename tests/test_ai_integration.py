"""End-to-end test of the AI domain, with the economy pipeline beside it (AI task 25).

One throwaway database (skipped if Postgres isn't reachable) goes through the
AI lifecycle, every external API faked:

  ingest (arXiv, HF, GitHub, Epoch via the real `monthly_ingest`; Polymarket
  via the real `ingest_polymarket.ingest`) -> questions -> forecasts from
  every model, the LLM mocked -> two months later, live pulls and a market
  resolution -> resolve -> score -> post-mortems -> LLM drafts and review ->
  the track record (scorecard, CLI, the dashboard's queries).

The economy job runs on the same database before and after, on the economy
test's synthetic store, and must come out the same as alone. Nothing calls a
real API.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection
from test_forecast_job import (
    FORECAST_DATE as ECONOMY_DATE,
)  # the economy job's own test data
from test_forecast_job import synthetic_store

from eco_prediction import cli
from eco_prediction.ai import (
    ingest_arxiv,
    ingest_github,
    ingest_huggingface,
    ingest_polymarket,
)
from eco_prediction.ai import postmortems as pm
from eco_prediction.ai.baselines import AILinearTrend, AIRandomWalk
from eco_prediction.ai.ingest_epoch import SERIES as EPOCH_SERIES
from eco_prediction.ai.jobs import ai_forecast_job as job
from eco_prediction.ai.jobs.ai_resolution_job import (
    ai_score_summary,
    scored_ai_forecasts,
)
from eco_prediction.ai.lightgbm_model import AILightGBMModel
from eco_prediction.ai.llm.backtest import BacktestBudget
from eco_prediction.ai.llm.claude_forecaster import ClaudeForecaster
from eco_prediction.ai.monthly_ingest import MonthlySeries, ingest_backfill, ingest_live
from eco_prediction.ai.months import MonthlyCount
from eco_prediction.ai.resolutions import resolve_and_score_ai
from eco_prediction.ai.sources.polymarket_client import Market
from eco_prediction.db import connection as db_connection
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import scored_forecasts
from eco_prediction.models.baselines import RandomWalkModel
from eco_prediction.scheduler.forecast_job import NumericModelSpec, run_forecast_job

TODAY = date(2026, 10, 16)
LATER = date(2026, 12, 10)  # November's values are in; a market has resolved
SMALL_GRID = {"max_depth": [3], "learning_rate": [0.1], "min_child_samples": [10]}


def utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


# --- fake sources ----------------------------------------------------------------------


class FakeCounter:
    """A source's monthly counts: geometric growth, with a jump in November."""

    def __init__(
        self, start: float, growth: float, november: float | None = None
    ) -> None:
        self.start, self.growth, self.november = start, growth, november

    def value(self, month: date) -> float:
        if self.november is not None and month == date(2026, 11, 1):
            return self.november
        i = (month.year - 2022) * 12 + month.month - 1
        return round(self.start * self.growth**i, 1)

    def count_months(self, months: Sequence[date]) -> Iterator[MonthlyCount]:
        for m in months:
            yield MonthlyCount(m, self.value(m))


SOURCES: list[tuple[MonthlySeries, FakeCounter]] = [
    (
        ingest_arxiv.SERIES,
        FakeCounter(3000, 1.02, november=25000),
    ),  # ~2.6x trend: a surprise
    (ingest_huggingface.SERIES, FakeCounter(5000, 1.05)),
    (ingest_github.SERIES, FakeCounter(1500, 1.01)),
    (EPOCH_SERIES["gpqa"].series, FakeCounter(40, 1.01)),
    (EPOCH_SERIES["compute"].series, FakeCounter(25, 1.001)),
]


def market(ext: str, closes: date, *, outcome: str | None = None) -> Market:
    return Market(
        id=ext,
        question=f"Will OpenAI release GPT-7 by {closes}? ({ext})",
        description="Resolves yes if it ships.", url=f"https://polymarket.com/event/{ext}",
        yes_token=f"tok-{ext}", volume=50_000, opened_at=utc(date(2026, 6, 1)),
        closes_at=utc(closes), closed=outcome is not None,
        closed_at=utc(closes) if outcome else None, outcome=outcome,  # type: ignore[arg-type]
        yes_price=None if outcome else 0.3,
        # As the real client records it: the key is there even while open.
        metadata={"closed_at": utc(closes).isoformat() if outcome else None},
    )  # fmt: skip


class FakePolymarket:
    def __init__(self, markets: list[Market]) -> None:
        self.markets = markets

    def ai_markets(self, min_volume: float) -> list[Market]:
        return self.markets

    def price_history(self, token: str) -> list[tuple[datetime, float]]:
        price = 0.15 if token == "tok-soon" else 0.3
        return [
            (utc(d), price)
            for d in (date(2026, 6, 1) + timedelta(days=i) for i in range(200))
        ]


class FakeClaude:
    """Numeric: the latest value +3%. Binary: 0.6. Drafts: a sentence."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        schema = json.dumps(kwargs["output_config"]["format"]["schema"])
        if "explanation" in schema:
            body: dict[str, Any] = {
                "explanation": "A wave of agent papers in November."
            }
        elif "point_forecast" in schema:
            prompt = kwargs["messages"][0]["content"]
            last = float(prompt.split("<recent_values")[1].split("</recent_values>")[0]
                         .strip().split("\n")[-1].split(": ")[1].replace(",", ""))  # fmt: skip
            body = {"rationale": "trend", "point_forecast": last * 1.03,
                    "lower_80": last * 0.95, "upper_80": last * 1.12}  # fmt: skip
        else:
            body = {"rationale": "evidence", "probability": 0.6}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(body))],
            stop_reason="end_turn", model=kwargs["model"], stop_details=None,
            usage=SimpleNamespace(input_tokens=3000, output_tokens=500, iterations=None,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )  # fmt: skip


NUMERIC = (
    job.NumericSpec("ai_random_walk", lambda s, h: AIRandomWalk()),
    job.NumericSpec("ai_linear_trend", lambda s, h: AILinearTrend()),
    job.NumericSpec("ai_lightgbm", lambda s, h: AILightGBMModel(h, param_grid=SMALL_GRID),
                    explain=True),
)  # fmt: skip
ECONOMY_MODELS = (NumericModelSpec("random_walk", lambda h: RandomWalkModel()),)


def count(conn: Connection, sql: str, *params: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()[0])  # type: ignore[index]


def economy_snapshot(conn: Connection) -> tuple[int, int, int]:
    return (
        count(conn, "SELECT count(*) FROM questions WHERE domain = 'economy'"),
        count(
            conn,
            """SELECT count(*) FROM forecasts f JOIN questions q ON q.id = f.question_id
                       WHERE q.domain = 'economy'""",
        ),
        len(scored_forecasts(conn, include_backtest=True)),
    )


def run_economy(conn: Connection) -> Any:
    # The economy job's own test store has unemployment, not CPI.
    return run_forecast_job(
        conn,
        synthetic_store(),
        ECONOMY_DATE,
        numeric_models=ECONOMY_MODELS,
        include_fomc=False,
        targets=("unemployment",),
    )


@contextmanager
def using(conn: Connection) -> Iterator[Connection]:
    yield conn


@pytest.fixture
def db(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> Connection:
    migrate(conn)
    monkeypatch.setattr(db_connection, "connection", lambda: using(conn))
    monkeypatch.setattr(cli, "today", lambda: LATER)
    return conn


def test_the_ai_pipeline_end_to_end_beside_the_economy(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    # --- the economy, alone first --------------------------------------------------------
    economy_alone = run_economy(db)
    assert economy_alone.failures == [] and economy_alone.forecasts_inserted > 0
    before = economy_snapshot(db)

    # --- 1. ingest -------------------------------------------------------------------------
    for series, counter in SOURCES:
        result = ingest_backfill(
            db, series, counter, date(2022, 1, 1), date(2026, 9, 1)
        )
        assert result.inserted == 57  # 2022-01 .. 2026-09
    poly = FakePolymarket([market("live", TODAY + timedelta(days=89)),
                           market("soon", date(2026, 12, 5))])  # fmt: skip
    ingested = ingest_polymarket.ingest(db, poly, TODAY)  # type: ignore[arg-type]
    db.commit()
    assert ingested.markets == 2 and ingested.history_rows > 0

    # --- 2-4. questions and forecasts, the LLM mocked and capped -------------------------------
    fake = FakeClaude()
    llm = ClaudeForecaster(conn=db, client=fake)  # type: ignore[arg-type]
    first = job.run_ai_forecast_job(db, TODAY, numeric_models=NUMERIC, llm=llm,
                                    llm_budget_usd=10.0)  # fmt: skip
    assert first.failures == []
    with db.cursor() as cur:
        cur.execute("""SELECT question_type::text, (platform_question_id IS NOT NULL), count(*)
                       FROM questions WHERE domain = 'ai' GROUP BY 1, 2""")
        made = {(t, c): n for t, c, n in cur.fetchall()}
    assert made[("numeric_interval", False)] == 21  # 7 targets x 3 horizons
    assert made[("binary", False)] > 0  # thresholds for the 5 ingested series
    assert made[("binary", True)] == 2  # each market's 90-day question
    with db.cursor() as cur:
        cur.execute("""SELECT m.model_type, count(*) FROM forecasts f
                       JOIN model_versions m ON m.id = f.model_version_id
                       JOIN questions q ON q.id = f.question_id
                       WHERE q.domain = 'ai' GROUP BY 1""")
        stored = dict(cur.fetchall())
    for model in ("ai_random_walk", "ai_linear_trend", "ai_lightgbm", "ai_base_rate",
                  "ai_threshold_classifier", "ai_market_price", "claude_llm"):  # fmt: skip
        assert stored.get(model, 0) > 0, model
    assert (
        count(db, "SELECT count(*) FROM llm_calls WHERE forecast_id IS NOT NULL")
        == stored["claude_llm"]
    )
    # $0.022 a call (3,000 in, 500 out at Opus 5.5 prices): well inside the cap.
    assert 0 < first.llm_spent_usd < 10.0

    # --- 5. two months later: live pulls and a market resolution ---------------------------------
    for series, counter in SOURCES:
        ingest_live(db, series, counter, date(2026, 12, 2))
    poly.markets = [market("live", TODAY + timedelta(days=89)),
                    market("soon", date(2026, 12, 5), outcome="yes")]  # fmt: skip
    ingest_polymarket.ingest(db, poly, date(2026, 12, 6))  # type: ignore[arg-type]
    db.commit()

    # --- 6. resolve and score -----------------------------------------------------------------
    resolved = resolve_and_score_ai(db, LATER)
    db.commit()
    assert resolved.resolved > 0 and resolved.scored > 0
    assert count(db, """SELECT count(*) FROM resolutions r JOIN questions q ON q.id = r.question_id
                        WHERE q.question_key = 'ARXIV_AI_SUBMISSIONS'
                          AND q.target_date = '2026-11-01'""") == 1  # fmt: skip
    assert (
        count(
            db,
            """SELECT count(*) FROM resolutions r JOIN questions q ON q.id = r.question_id
                        WHERE q.question_key = 'polymarket:soon' AND r.actual_outcome = 'yes'""",
        )
        == 1
    )

    # --- 7. post-mortems: categories, then LLM drafts and a review --------------------------------
    classified = pm.run_ai_postmortems(db, LATER)
    db.commit()
    assert classified.classified == resolved.scored
    # arXiv's November at 3x trend is a surprise no model saw coming.
    with db.cursor() as cur:
        cur.execute("""SELECT DISTINCT p.miss_category::text FROM postmortems p
                       JOIN forecasts f ON f.id = p.forecast_id JOIN questions q ON q.id = f.question_id
                       WHERE q.question_key = 'ARXIV_AI_SUBMISSIONS' AND q.horizon_months = 1""")
        assert {r[0] for r in cur.fetchall()} == {"regime_change"}
    drafter = pm.make_drafter(db, client=fake)
    drafted = pm.draft_postmortems(db, drafter, BacktestBudget(1.0), include_backtest=True,
                                   commit=db.commit)  # fmt: skip
    assert drafted.drafted > 0 and drafted.failures == []
    first_draft = int(pm.list_drafts(db)["forecast_id"].iloc[0])
    assert (
        pm.review_main(["--confirm", str(first_draft)], connect=lambda: using(db)) == 0
    )

    # --- 8. the track record ---------------------------------------------------------------------
    summary = ai_score_summary(scored_ai_forecasts(db, include_backtest=True))
    numeric = summary["numeric"]
    assert set(numeric["model_type"]) >= {"ai_random_walk", "ai_lightgbm", "claude_llm"}
    assert (
        numeric[numeric["series"] == "ARXIV_AI_SUBMISSIONS"]["relative_rmse"] > 0.4
    ).all()
    binary = summary["binary"]
    assert set(binary["kind"]) >= {"threshold", "curated"}
    assert binary["brier"].between(0, 1).all()
    assert cli.main(["scores", "--domain", "ai", "--include-backtest"]) == 0
    assert "ARXIV_AI_SUBMISSIONS" in capsys.readouterr().out

    # The next run carries on: new month, nothing duplicated, the stored LLM forecasts skipped.
    llm_before = count(db, "SELECT count(*) FROM llm_calls")
    later = job.run_ai_forecast_job(
        db, LATER, numeric_models=NUMERIC[:1], classifier=False
    )
    assert later.failures == [] and later.forecasts_inserted > 0
    assert (
        count(db, "SELECT count(*) FROM llm_calls") == llm_before
    )  # no LLM unless asked

    # --- the economy, alongside -----------------------------------------------------------------
    assert economy_snapshot(db) == before  # the AI pipeline changed nothing in it
    again = run_economy(db)
    assert (
        again.forecasts_inserted == 0
        and again.forecasts_skipped == economy_alone.forecasts_inserted
    )
    assert economy_snapshot(db) == before
    assert all(q == "unemployment" for q in
               scored_forecasts(db, include_backtest=True)["target"].unique())  # fmt: skip
    assert cli.main(["inspect"]) == 0  # the economy's listing has no AI questions
    assert "ARXIV" not in capsys.readouterr().out

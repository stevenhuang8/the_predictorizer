"""Tests for AI post-mortems: categories, LLM drafts and their review (AI task 22).

The API is never called: a fake client returns canned drafts and records
each request. Database tests use a throwaway, migrated database (skipped if
Postgres isn't reachable).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import postmortems as pm
from eco_prediction.ai.baselines import AIRandomWalk
from eco_prediction.ai.ingest_arxiv import SERIES as ARXIV_SERIES
from eco_prediction.ai.jobs import ai_forecast_job as job
from eco_prediction.ai.llm.backtest import BacktestBudget
from eco_prediction.ai.llm.retrieval import Document, upsert_documents
from eco_prediction.ai.months import iter_months, next_month
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.resolutions import resolve_and_score_ai
from eco_prediction.ai.store import (
    AIObservation,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.backtest.harness import Forecast
from eco_prediction.data.ingest import ensure_source
from eco_prediction.db.forecasts import (
    get_or_create_model_version,
    get_or_create_question,
    save_forecast,
)
from eco_prediction.db.migrate import migrate
from eco_prediction.db.postmortems import save_postmortem

ARXIV = "ARXIV_AI_SUBMISSIONS"
TODAY = date(2026, 10, 16)
RESOLVED_ON = date(2026, 12, 10)


def utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


# --- categories (no database) -----------------------------------------------------------


def test_binary_categories_follow_the_fomc_rule() -> None:
    assert (
        pm.classify_binary(0.3, "yes", 0.9, "market price").category
        == "expected_variance"
    )
    # 1 - 0.8 is 0.19999999999999996 in floating point: still the 20% boundary.
    assert (
        pm.classify_binary(0.8, "no", 0.5, "market price").category
        == "expected_variance"
    )
    missed = pm.classify_binary(0.1, "yes", 0.7, "market price")
    assert missed.category == "bad_model" and "market price gave it 70%" in missed.notes
    surprise = pm.classify_binary(0.9, "no", 0.95, "historical base rate")
    assert surprise.category == "regime_change"
    assert surprise.evidence["reference_p_outcome"] == pytest.approx(0.05)
    assert pm.classify_binary(0.1, "yes", None, "market price").category == "bad_model"


# --- the database ---------------------------------------------------------------------------


class FakeClaude:
    def __init__(
        self, text: str = "The jump came from a wave of agent papers."
    ) -> None:
        self.text = text
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"explanation": self.text}))],
            stop_reason="end_turn", model=kwargs["model"], stop_details=None,
            usage=SimpleNamespace(input_tokens=3000, output_tokens=200, iterations=None,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )  # fmt: skip


@pytest.fixture
def db(conn: Connection) -> Connection:
    """arXiv through 2026-09, then a November far above trend; two markets, one
    resolving 'yes' at a 10% price. Forecast on TODAY, resolved on RESOLVED_ON."""
    migrate(conn)
    source = ensure_source(conn, *ARXIV_SERIES.source)
    series = upsert_ai_series(conn, source, ARXIV_SERIES.info)
    months = list(iter_months(date(2021, 1, 1), date(2026, 9, 1)))
    insert_ai_observations(conn, series, [
        AIObservation(m, next_month(m), 4000 * 1.015**i, True) for i, m in enumerate(months)
    ])  # fmt: skip
    for ext, ask in (
        ("hit", TODAY - timedelta(days=1)),
        ("miss", TODAY - timedelta(days=2)),
    ):
        close = ask + timedelta(days=90)
        pq, _, _ = upsert_platform_question(conn, PlatformQuestion(
            "polymarket", ext, f"Will OpenAI release GPT-7 by {close}? ({ext})",
            opened_at=utc(date(2026, 6, 1)), closes_at=utc(close),
            metadata={"closed_at": None},
        ))  # fmt: skip
        with conn.cursor() as cur:
            cur.execute("INSERT INTO platform_probabilities VALUES (%s, %s, %s, false)",
                        (pq, ask - timedelta(days=3), 0.6 if ext == "hit" else 0.1))  # fmt: skip
    conn.commit()
    job.run_ai_forecast_job(
        conn, TODAY, numeric_models=(job.NumericSpec("ai_random_walk", lambda s, h: AIRandomWalk()),),
        classifier=False,
    )  # fmt: skip
    # November arrives far above trend (a live value); both markets resolve yes.
    insert_ai_observations(conn, series, [AIObservation(date(2026, 11, 1), date(2026, 12, 2),
                                                        9000.0, False)])  # fmt: skip
    with conn.cursor() as cur:
        cur.execute("UPDATE platform_questions SET resolved_at = %s, resolved_outcome = 'yes'",
                    (utc(date(2026, 12, 5)),))  # fmt: skip
    resolve_and_score_ai(conn, RESOLVED_ON)
    upsert_documents(conn, [
        Document("arxiv_abstract", "after", "A wave of agent papers on growth of AI research.",
                 utc(date(2026, 11, 3)), title="Agents everywhere"),
        Document("arxiv_abstract", "too-late", "Growth of AI research language models, later.",
                 utc(date(2026, 12, 20)), title="Too late"),
        Document("arxiv_abstract", "before", "Growth of AI research language models, earlier.",
                 utc(date(2026, 9, 1)), title="Before the forecast"),
    ])  # fmt: skip
    conn.commit()
    return conn


def categories(db: Connection) -> dict[tuple[str, str], str]:
    with db.cursor() as cur:
        cur.execute("""
            SELECT q.question_key, m.model_type, p.miss_category::text FROM postmortems p
            JOIN forecasts f ON f.id = p.forecast_id JOIN questions q ON q.id = f.question_id
            JOIN model_versions m ON m.id = f.model_version_id""")
        return {(k, m): c for k, m, c in cur.fetchall()}


def test_ai_forecasts_get_categories(db: Connection) -> None:
    result = pm.run_ai_postmortems(db, RESOLVED_ON)
    cats = categories(db)
    assert result.classified == len(cats) > 0
    # The market said 10% for 'miss' and it happened: the market missed it too.
    assert cats[("polymarket:miss", "ai_market_price")] == "regime_change"
    assert cats[("polymarket:hit", "ai_market_price")] == "expected_variance"  # 60%
    # November at ~2x trend is far beyond any past 1-month log move.
    assert cats[(ARXIV, "ai_random_walk")] == "regime_change"
    with db.cursor() as cur:
        cur.execute("""SELECT p.notes, p.evidence->>'scale' FROM postmortems p
            JOIN forecasts f ON f.id = p.forecast_id JOIN questions q ON q.id = f.question_id
            WHERE q.question_key = %s AND q.horizon_months = 1""", (ARXIV,))  # fmt: skip
        notes, scale = cur.fetchone()  # type: ignore[misc]
    assert notes.startswith("On the log scale:") and scale == "log"


def test_numeric_count_series_are_judged_on_the_log_scale() -> None:
    # Steady 1.5% growth: a +3% month is ordinary on logs, though large in levels late on.
    levels = 4000 * 1.015 ** np.arange(60)
    moves = np.diff(np.log(levels))
    c = pm.classify_numeric(
        prediction=math.log(levels[-1] * 1.015), lower=None, upper=None,
        actual_first=math.log(levels[-1] * 1.03), actual_latest=None,
        last_known=math.log(levels[-1]), historical_moves=np.append(moves, 0.02),
    )  # fmt: skip
    assert c.category == "regime_change"  # 3% beats every past move here


def draft(db: Connection, fake: FakeClaude, **kwargs: Any) -> pm.AIPostmortemResult:
    drafter = pm.make_drafter(db, client=fake)
    return pm.draft_postmortems(
        db, drafter, BacktestBudget(kwargs.pop("budget", 5.0)), **kwargs
    )


def test_misses_are_drafted_with_hindsight_and_stored_unconfirmed(
    db: Connection,
) -> None:
    pm.run_ai_postmortems(db, RESOLVED_ON)
    fake = FakeClaude()
    result = draft(db, fake, include_backtest=True)
    cats = categories(db)
    misses = sum(c != "expected_variance" for c in cats.values())
    assert result.drafted == misses == len(fake.requests) > 0
    req = fake.requests[0]
    assert (
        req["model"] == "claude-haiku-5-5" and req["output_config"]["effort"] == "low"
    )
    assert "fallbacks" not in req and req["system"] == pm.DRAFT_SYSTEM
    # Find the arXiv draft: it saw the classifier's call, the values since the
    # forecast, and only documents published between forecast and resolution.
    arxiv = next(r["messages"][0]["content"] for r in fake.requests
                 if "ARXIV_AI_SUBMISSIONS be for" in r["messages"][0]["content"])  # fmt: skip
    assert '<classifier category="regime_change">' in arxiv
    assert "2026-11: 9,000" in arxiv
    assert "Agents everywhere" in arxiv
    assert "Too late" not in arxiv and "Before the forecast" not in arxiv
    with db.cursor() as cur:
        cur.execute("""SELECT count(*), bool_and(NOT human_confirmed),
            bool_and(draft_note = %s), count(draft_call_id) FROM postmortems WHERE llm_drafted""",
                    (fake.text,))  # fmt: skip
        assert cur.fetchone() == (misses, True, True, misses)
        cur.execute(
            "SELECT count(*) FROM postmortems WHERE llm_drafted AND miss_category = 'expected_variance'"
        )
        assert cur.fetchone() == (0,)
    # Nothing left to draft; a reclassification keeps the drafts.
    assert draft(db, FakeClaude(), include_backtest=True).drafted == 0
    pm.run_ai_postmortems(db, RESOLVED_ON)
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM postmortems WHERE llm_drafted")
        assert cur.fetchone() == (misses,)


def test_only_live_misses_are_drafted_by_default_and_the_budget_caps_it(
    db: Connection,
) -> None:
    pm.run_ai_postmortems(db, RESOLVED_ON)
    live = len(pm.undrafted_misses(db))
    every = len(pm.undrafted_misses(db, include_backtest=True))
    assert 0 < live <= every
    # $0.0004 a draft on Haiku (3,000 in at $0.10, 200 out at $0.50 per million).
    fake = FakeClaude()
    result = draft(db, fake, include_backtest=True, budget=0.0004)
    assert result.drafted == 1 and "budget" in result.failures[0]
    assert len(fake.requests) == 1


def test_a_failed_draft_is_skipped_and_logged(db: Connection) -> None:
    pm.run_ai_postmortems(db, RESOLVED_ON)
    fake = FakeClaude(text="")  # an empty explanation is refused
    result = draft(db, fake, include_backtest=True, limit=2)
    assert result.drafted == 0 and len(result.failures) == 2
    assert "empty" in result.failures[0]
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM llm_calls WHERE error LIKE 'malformed reply%'"
        )
        assert cur.fetchone() == (2,)
        cur.execute("SELECT count(*) FROM postmortems WHERE llm_drafted")
        assert cur.fetchone() == (0,)


def test_drafting_needs_a_category_first(db: Connection) -> None:
    with db.cursor() as cur:
        cur.execute("SELECT min(id) FROM forecasts")
        fid = cur.fetchone()[0]  # type: ignore[index]
    with pytest.raises(ValueError, match="no post-mortem"):
        pm.draft_llm_postmortem(db, fid, pm.make_drafter(db, client=FakeClaude()))
    with pytest.raises(ValueError, match="isn't a scored AI forecast"):
        pm.draft_llm_postmortem(db, 999_999, pm.make_drafter(db, client=FakeClaude()))


# --- review ----------------------------------------------------------------------------------


@contextmanager
def using(conn: Connection) -> Iterator[Connection]:
    yield conn


def test_confirm_edit_and_reject(
    db: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    pm.run_ai_postmortems(db, RESOLVED_ON)
    draft(db, FakeClaude(), include_backtest=True)
    db.commit()
    first, second = pm.list_drafts(db)["forecast_id"].tolist()[:2]

    assert pm.review_main(["--domain", "ai"], connect=lambda: using(db)) == 0
    out = capsys.readouterr().out
    assert f"#{first}" in out and "unconfirmed" in out and "agent papers" in out

    assert pm.review_main(["--confirm", str(first), "--note", "Edited: an agents boom."],
                          connect=lambda: using(db)) == 0  # fmt: skip
    assert pm.review_main(["--reject", str(second)], connect=lambda: using(db)) == 0
    with db.cursor() as cur:
        cur.execute("SELECT human_confirmed, draft_note, confirmed_at IS NOT NULL, miss_category::text"
                    " FROM postmortems WHERE forecast_id = %s", (first,))  # fmt: skip
        confirmed, note, dated, category = cur.fetchone()  # type: ignore[misc]
        assert confirmed and note == "Edited: an agents boom." and dated and category
        cur.execute("SELECT llm_drafted, draft_note, miss_category::text FROM postmortems"
                    " WHERE forecast_id = %s", (second,))  # fmt: skip
        drafted, note, category = cur.fetchone()  # type: ignore[misc]
        assert not drafted and note is None and category  # the category stays
    assert second not in set(pm.list_drafts(db, include_confirmed=True)["forecast_id"])
    assert first not in set(
        pm.list_drafts(db)["forecast_id"]
    )  # confirmed: off the queue
    assert (
        pm.review_main(["--confirm", str(second)], connect=lambda: using(db)) == 1
    )  # no draft
    with pytest.raises(SystemExit):
        pm.review_main(["--note", "x"], connect=lambda: using(db))


def test_the_schema_refuses_inconsistent_drafts(db: Connection) -> None:
    pm.run_ai_postmortems(db, RESOLVED_ON)
    db.commit()  # each violation below rolls back; keep the rows across them
    with db.cursor() as cur:
        cur.execute("SELECT min(forecast_id) FROM postmortems")
        fid = cur.fetchone()[0]  # type: ignore[index]
    for sql in (
        "UPDATE postmortems SET llm_drafted = TRUE WHERE forecast_id = %s",  # no note
        "UPDATE postmortems SET human_confirmed = TRUE WHERE forecast_id = %s",  # not drafted
    ):
        with db.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(sql, (fid,))
        db.rollback()


def test_economy_postmortems_are_untouched(db: Connection) -> None:
    qid, _ = get_or_create_question(db, "cpi_yoy", date(2026, 9, 1), 3)
    version = get_or_create_model_version(db, "random_walk", "cpi")
    fid = save_forecast(db, qid, version, date(2026, 6, 1), Forecast(3.0, 2.5, 3.5))
    save_postmortem(db, fid, "bad_model", notes="economy note", classified_by="manual")
    pm.run_ai_postmortems(db, RESOLVED_ON)
    draft(db, FakeClaude(), include_backtest=True)
    with db.cursor() as cur:
        cur.execute("SELECT miss_category::text, notes, llm_drafted, human_confirmed, draft_note"
                    " FROM postmortems WHERE forecast_id = %s", (fid,))  # fmt: skip
        assert cur.fetchone() == ("bad_model", "economy note", False, False, None)
    assert fid not in {f.forecast_id for f in pm.ai_forecasts_to_review(db)}


def test_the_cli_passes_review_options_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from eco_prediction import cli

    seen: list[list[str]] = []
    monkeypatch.setattr(pm, "review_main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["review-postmortems", "--domain", "ai", "--confirm", "7"]) == 0
    assert seen == [["--domain", "ai", "--confirm", "7"]]

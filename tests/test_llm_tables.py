"""Tests for the LLM tables (migration 014): call logging and the retrieval
corpus (AI task 16)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai.llm.calls import (
    LLMCall,
    attach_forecast,
    cached_call,
    context_hash,
    log_call,
)
from eco_prediction.ai.llm.retrieval import (
    POLYMARKET_QUESTION,
    Document,
    end_of_day,
    index_platform_questions,
    retrieve_documents,
    upsert_documents,
)
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.backtest.harness import Forecast
from eco_prediction.db.forecasts import (
    get_or_create_model_version,
    get_or_create_question,
    save_forecast,
)
from eco_prediction.db.migrate import migrate

MODEL = "claude-opus-5-5"


def utc(y: int, m: int, d: int, h: int = 0) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def a_forecast(conn: Connection) -> tuple[int, int]:
    """(question id, forecast id) of a stored numeric AI forecast."""
    question, _ = get_or_create_question(
        conn, "arxiv_ai_submissions", date(2027, 1, 1), 3,
        question_key="ARXIV_AI_SUBMISSIONS",
    )  # fmt: skip
    version = get_or_create_model_version(conn, "claude", "v1")
    forecast = save_forecast(conn, question, version, date(2026, 10, 15), Forecast(1.0))
    return question, forecast


def count(conn: Connection, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        return int(cur.fetchone()[0])  # type: ignore[index]


# --- llm_calls ----------------------------------------------------------------------


def test_context_hash_covers_model_prompt_and_params() -> None:
    key = context_hash(MODEL, "p", {"max_tokens": 10, "temperature": 0})
    assert len(key) == 64 and key == key.lower()
    assert key == context_hash(MODEL, "p", {"temperature": 0, "max_tokens": 10})
    assert key != context_hash(MODEL, "p", {"max_tokens": 11, "temperature": 0})
    assert key != context_hash(MODEL, "q", {"max_tokens": 10, "temperature": 0})
    assert key != context_hash(
        "claude-sonnet-5-5", "p", {"max_tokens": 10, "temperature": 0}
    )


def test_calls_are_logged_and_linked_to_their_forecast(migrated: Connection) -> None:
    question, forecast = a_forecast(migrated)
    call = log_call(
        migrated,
        LLMCall(
            MODEL, "prompt", raw_response='{"probability": 0.3}',
            parsed_response={"probability": 0.3}, question_id=question,
            context_hash=context_hash(MODEL, "prompt"), prompt_version="v1",
            as_of_date=date(2026, 10, 14), input_tokens=1200, output_tokens=300,
            cache_read_tokens=1000, cost_usd=0.012345,
        ),
    )  # fmt: skip
    attach_forecast(migrated, [call], forecast)
    with migrated.cursor() as cur:
        cur.execute(
            "SELECT forecast_id, question_id, parsed_response, cost_usd::float8"
            " FROM llm_calls WHERE id = %s",
            (call,),
        )
        assert cur.fetchone() == (forecast, question, {"probability": 0.3}, 0.012345)
    # Deleting the forecast deletes its calls.
    with migrated.cursor() as cur:
        cur.execute("DELETE FROM forecasts WHERE id = %s", (forecast,))
    assert count(migrated, "SELECT count(*) FROM llm_calls") == 0


def test_a_call_needs_a_real_forecast_and_a_response_or_error(
    migrated: Connection,
) -> None:
    with pytest.raises(ValueError, match="raw_response or an error"):
        log_call(migrated, LLMCall(MODEL, "prompt"))
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        log_call(migrated, LLMCall(MODEL, "prompt", raw_response="x", forecast_id=999))
    migrated.rollback()
    with pytest.raises(psycopg2.errors.CheckViolation, match="context_hash"):
        log_call(migrated, LLMCall(MODEL, "p", raw_response="x", context_hash="nothex"))
    migrated.rollback()
    with migrated.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO llm_calls (model_id, prompt) VALUES ('m', 'p')")
    migrated.rollback()
    failed = log_call(migrated, LLMCall(MODEL, "prompt", error="overloaded"))
    assert failed > 0


def test_cached_call_returns_the_latest_success_for_the_same_context(
    migrated: Connection,
) -> None:
    key = context_hash(MODEL, "prompt")
    assert cached_call(migrated, MODEL, key) is None
    first = log_call(
        migrated, LLMCall(MODEL, "prompt", raw_response="a", context_hash=key)
    )
    second = log_call(
        migrated,
        LLMCall(MODEL, "prompt", raw_response="b", context_hash=key, cost_usd=0.5),
    )
    log_call(migrated, LLMCall(MODEL, "prompt", error="timeout", context_hash=key))
    log_call(
        migrated, LLMCall("other-model", "prompt", raw_response="c", context_hash=key)
    )
    hit = cached_call(migrated, MODEL, key)
    assert hit is not None and hit.id == second != first
    assert hit.raw_response == "b" and hit.cost_usd == 0.5 and hit.created_at
    assert cached_call(migrated, MODEL, context_hash(MODEL, "other prompt")) is None


# --- retrieval_documents ------------------------------------------------------------


def doc(source_id: str, content: str, published: datetime, kind: str = "arxiv_abstract",
        **kwargs: object) -> Document:  # fmt: skip
    return Document(kind, source_id, content, published, **kwargs)  # type: ignore[arg-type]


CORPUS = [
    doc("2401.1", "SWE-bench agents resolve more GitHub issues", utc(2024, 1, 10)),
    doc("2402.1", "Scaling laws for language model pretraining", utc(2024, 2, 5)),
    doc("2403.1", "A new SWE-bench Verified record from agent scaffolds", utc(2024, 3, 1)),
    doc("rel-1", "transformers v5 release: SWE-bench harness", utc(2024, 2, 20),
        kind="github_release", title="v5.0.0"),
]  # fmt: skip


def test_upsert_reports_inserts_updates_and_unchanged(migrated: Connection) -> None:
    assert upsert_documents(migrated, CORPUS) == (4, 0, 0)
    edited = doc("2402.1", "Scaling laws, revised (v2)", utc(2024, 2, 5))
    result = upsert_documents(migrated, [*CORPUS[:1], edited, CORPUS[0]])
    assert (result.inserted, result.updated, result.unchanged) == (0, 1, 1)
    assert count(migrated, "SELECT count(*) FROM retrieval_documents") == 4
    with pytest.raises(ValueError, match="naive"):
        upsert_documents(migrated, [doc("x", "y", datetime(2024, 1, 1))])  # noqa: DTZ001


def test_retrieval_is_point_in_time(migrated: Connection) -> None:
    upsert_documents(migrated, CORPUS)
    # Published 2024-03-01 00:00 UTC: not known by the end of 2024-02-29,
    # known by the end of 2024-03-01.
    before = retrieve_documents(migrated, date(2024, 2, 29), limit=50)
    assert {d.source_id for d in before} == {"2401.1", "2402.1", "rel-1"}
    assert all(d.published_at < end_of_day(date(2024, 2, 29)) for d in before)
    on = retrieve_documents(migrated, date(2024, 3, 1), limit=50)
    assert [d.source_id for d in on] == [
        "2403.1",
        "rel-1",
        "2402.1",
        "2401.1",
    ]  # newest first
    assert retrieve_documents(migrated, date(2023, 12, 31)) == []


def test_retrieval_filters_types_and_ranks_by_query(migrated: Connection) -> None:
    upsert_documents(migrated, CORPUS)
    as_of = date(2024, 6, 1)
    assert [
        d.source_id for d in retrieve_documents(migrated, as_of, "github_release")
    ] == ["rel-1"]
    both = retrieve_documents(migrated, as_of, ["github_release", "arxiv_abstract"])
    assert len(both) == 4
    hits = retrieve_documents(
        migrated, as_of, "arxiv_abstract", query="SWE-bench agent", match="all"
    )
    # Stemmed (agents ~ agent); the adjacent "SWE-bench agents" ranks first.
    assert [d.source_id for d in hits] == ["2401.1", "2403.1"]
    excluded = retrieve_documents(
        migrated, as_of, query="SWE-bench -agent", match="all"
    )
    assert [d.source_id for d in excluded] == ["rel-1"]
    assert (
        retrieve_documents(migrated, as_of, query="scaling -pretraining", match="all")
        == []
    )
    # Any word matches by default, most matches first; "all" needs every word.
    loose = retrieve_documents(migrated, as_of, query="pretraining agents yesterday")
    assert {d.source_id for d in loose} == {"2401.1", "2402.1", "2403.1"}
    strict = retrieve_documents(
        migrated, as_of, query="pretraining agents yesterday", match="all"
    )
    assert strict == []
    ranked = retrieve_documents(migrated, as_of, query="SWE-bench verified record")
    assert ranked[0].source_id == "2403.1"
    assert (
        len(retrieve_documents(migrated, as_of, query="the of")) == 4
    )  # stop words only
    assert [d.source_id for d in retrieve_documents(migrated, as_of, limit=1)] == [
        "2403.1"
    ]
    since = retrieve_documents(migrated, as_of, since=date(2024, 2, 20))
    assert {d.source_id for d in since} == {"2403.1", "rel-1"}
    with pytest.raises(ValueError):
        retrieve_documents(migrated, as_of, limit=0)


def test_polymarket_markets_are_indexed_without_their_outcomes(
    migrated: Connection,
) -> None:
    market = PlatformQuestion(
        "polymarket", "573001", "Will an AI score 90% on SWE-bench Verified by June 30?",
        resolution_criteria="Resolves yes if the leaderboard shows 90%.",
        url="https://polymarket.com/event/x", opened_at=utc(2025, 1, 5, 12),
        closes_at=utc(2025, 6, 30), resolved_at=utc(2025, 7, 1), resolved_outcome="no",
        metadata={"volume": 50000},
    )  # fmt: skip
    upsert_platform_question(migrated, market)
    unopened = PlatformQuestion("polymarket", "573002", "Will GPT-7 be released?")
    upsert_platform_question(migrated, unopened)
    assert index_platform_questions(migrated) == (2, 0, 0)
    assert index_platform_questions(migrated).unchanged == 2  # re-indexing is a no-op

    docs = retrieve_documents(
        migrated, date(2026, 10, 14), POLYMARKET_QUESTION, query="SWE-bench"
    )
    (d,) = docs
    assert d.source_id == "573001" and d.published_at == utc(2025, 1, 5, 12)
    assert d.content.endswith("leaderboard shows 90%.") and d.title == market.title
    assert d.metadata["category"] == "benchmark" and d.metadata["url"] == market.url
    text = f"{d.content} {d.metadata}"
    assert "resolved" not in text and "'no'" not in text and "50000" not in text
    assert retrieve_documents(migrated, date(2025, 1, 4), POLYMARKET_QUESTION) == []
    # A market without an open date is dated when we first saw it.
    with migrated.cursor() as cur:
        cur.execute(
            "SELECT first_seen_at FROM platform_questions WHERE external_id = '573002'"
        )
        seen = cur.fetchone()[0]  # type: ignore[index]
    gpt7 = retrieve_documents(migrated, seen.date(), query="GPT-7")
    assert [x.published_at for x in gpt7] == [seen]

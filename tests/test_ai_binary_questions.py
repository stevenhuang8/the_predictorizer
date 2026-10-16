"""Tests for threshold and curated binary questions (AI task 10)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import ingest_arxiv, ingest_epoch
from eco_prediction.ai.curated_questions import (
    MarketRow,
    classify,
    load_curated_questions,
    plan_curated_questions,
)
from eco_prediction.ai.platform_store import PlatformQuestion, upsert_platform_question
from eco_prediction.ai.questions import (
    NUMERIC_TARGETS,
    generate_threshold_binary_questions,
    main,
    resolution_rule,
    thresholds,
)
from eco_prediction.ai.store import (
    AIObservation,
    ensure_source,
    insert_ai_observations,
    upsert_ai_series,
)
from eco_prediction.db import connection as db
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import unresolved_questions

ARXIV = ingest_arxiv.SERIES.info
SWE = ingest_epoch.SERIES["swe_bench"].series.info
FLOPS = ingest_epoch.SERIES["compute"].series.info
AS_OF = date(2026, 10, 15)
SEP = date(2026, 9, 1)


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def rows(conn: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def store(conn: Connection, info: Any, *obs: tuple[date, date, float]) -> None:
    source = ensure_source(conn, "test", "https://example.com", "test source")
    pk = upsert_ai_series(conn, source, info)
    insert_ai_observations(
        conn, pk, [AIObservation(m, a, v, reconstructed=False) for m, a, v in obs]
    )


def keys(th: list[Any]) -> list[str]:
    return [t.key for t in th]


# --- thresholds -------------------------------------------------------------


def test_count_thresholds_are_10_25_50_percent_up_to_3_figures() -> None:
    assert keys(thresholds(ARXIV, 30_000)) == ["33000", "37500", "45000"]
    assert keys(thresholds(ARXIV, 29_800)) == ["32800", "37200", "44700"]


def test_sota_thresholds_close_part_of_the_gap_to_100() -> None:
    th = thresholds(SWE, 83.47)
    assert keys(th) == ["87.6", "91.7", "95.9"]
    assert th[0].step == "25% of the gap to 100%"


def test_saturated_sota_has_one_or_no_threshold() -> None:
    assert keys(thresholds(SWE, 99.96)) == ["100.0"]  # all three round to 100
    assert thresholds(SWE, 100.0) == []


def test_compute_thresholds_are_2_3_10_times_the_record() -> None:
    assert keys(thresholds(FLOPS, 27.97)) == ["28.27", "28.45", "28.97"]


def test_every_target_series_has_a_threshold_kind() -> None:
    for t in NUMERIC_TARGETS:
        assert len(thresholds(t.series, 50.0)) == 3, t.series.series_id


# --- threshold questions ----------------------------------------------------


def test_threshold_questions_twin_the_numeric_ones(migrated: Connection) -> None:
    store(migrated, ARXIV, (SEP, date(2026, 10, 2), 30_000))
    result = generate_threshold_binary_questions(migrated, AS_OF)
    migrated.commit()

    assert result.created == 9
    stored = rows(
        migrated,
        """
        SELECT question_key, target::text, question_type::text, domain,
               target_date, horizon_months, lead_days
        FROM questions ORDER BY id
        """,
    )
    months = {1: date(2026, 11, 1), 3: date(2027, 1, 1), 6: date(2027, 4, 1)}
    assert stored == [
        (f"ARXIV_AI_SUBMISSIONS>={t}", "ai_binary_event", "binary", "ai", m, h, None)
        for h, m in months.items()
        for t in ("33000", "37500", "45000")
    ]
    # Every other series has no September value yet.
    assert len(result.missing_base) == len(NUMERIC_TARGETS) - 1


def test_question_text_and_rule(migrated: Connection) -> None:
    store(migrated, SWE, (SEP, date(2026, 10, 1), 83.47))
    generate_threshold_binary_questions(migrated, AS_OF)
    text, rule = rows(
        migrated,
        "SELECT question_text, resolution_rule FROM questions WHERE question_key = %s"
        " AND horizon_months = 3",
        "EPOCH_SWEBENCH_SOTA>=87.6",
    )[0]
    numeric_rule = resolution_rule(SWE, date(2027, 1, 1))
    assert (
        rule
        == f"{numeric_rule} Resolves yes if that value is at least 87.6%, otherwise no."
    )
    assert text.startswith(
        "Will Best SWE-bench Verified score among models released by month end "
        "(Epoch AI) be at least 87.6% for 2027-01? It was 83.5% for 2026-09 "
        "(threshold: 25% of the gap to 100%)."
    )
    assert text.endswith(rule)


def test_base_is_point_in_time(migrated: Connection) -> None:
    # A revision stored after the ask date isn't seen.
    store(
        migrated,
        ARXIV,
        (SEP, date(2026, 10, 2), 30_000),
        (SEP, date(2026, 10, 20), 40_000),
    )
    result = generate_threshold_binary_questions(migrated, AS_OF, horizons=[1])
    assert [k for k, *_ in result.questions] == [
        "ARXIV_AI_SUBMISSIONS>=33000",
        "ARXIV_AI_SUBMISSIONS>=37500",
        "ARXIV_AI_SUBMISSIONS>=45000",
    ]


def test_a_month_keeps_its_first_set(migrated: Connection) -> None:
    store(
        migrated,
        ARXIV,
        (SEP, date(2026, 10, 2), 30_000),
        (SEP, date(2026, 10, 20), 40_000),
    )
    first = generate_threshold_binary_questions(migrated, AS_OF)
    # Later in the month the base is revised, but the set is reused as is.
    again = generate_threshold_binary_questions(migrated, date(2026, 10, 25))
    assert again.created == 0
    assert [q[3] for q in again.questions] == [q[3] for q in first.questions]
    assert rows(migrated, "SELECT count(*) FROM questions") == [(9,)]


def test_a_missing_base_is_picked_up_later(migrated: Connection) -> None:
    targets = [t for t in NUMERIC_TARGETS if t.series == ARXIV]
    early = generate_threshold_binary_questions(migrated, date(2026, 10, 1), targets)
    assert (early.created, early.missing_base) == (0, ["ARXIV_AI_SUBMISSIONS"])
    store(migrated, ARXIV, (SEP, date(2026, 10, 2), 30_000))
    later = generate_threshold_binary_questions(migrated, date(2026, 10, 3), targets)
    assert (later.created, later.missing_base) == (9, [])


# --- classifying markets ----------------------------------------------------


@pytest.mark.parametrize(
    ("title", "category"),
    [
        (
            "Will any AI model reach 1530 Overall Arena Score by September 30, 2026?",
            "benchmark",
        ),
        (
            "Will an Anthropic Claude model score at least 35% on Humanity’s Last Exam?",
            "benchmark",
        ),
        ("AI wins IMO gold medal in 2026?", "benchmark"),
        ("Will OpenAI have the best AI model at the end of September 2026?", "ranking"),
        ("Will Anthropic have the #3 AI model at the end of June 2026?", "ranking"),
        (
            "Will a Chinese company have one of the top 3 AI models by December 31?",
            "ranking",
        ),
        ("GPT-5.6 released by June 23, 2026?", "release"),
        ("Will the next Claude Opus model be released by July 17, 2026?", "release"),
        (
            "Will the next Google Gemini Pro model be released on June 30, 2026?",
            "exact_day",
        ),
        ("Will OpenAI release a new frontier model on December 11?", "exact_day"),
        ("Will Anthropic's valuation hit (HIGH) $1T by December 31?", "finance"),
        (
            "Will the Ornn RTX 5090 Index be between $0.50 and $0.55 on August 31, 2026?",
            "finance",
        ),
        (
            "Will Unknown’s Mystery Model win the Alpha Arena Season 1.5 competition?",
            "finance",
        ),
        ("Will Sam Altman sue OpenAI?", "people_legal"),
        (
            "Will Claude Fable 5 be restored for US customers by June 18?",
            "people_legal",
        ),
        ("ChatGPT Full Outage by March 31?", "outage_app"),
        (
            "Will ChatGPT be out as the #1 Free App in the US Apple Store by February 24?",
            "outage_app",
        ),
        (
            "Will OpenAI launch a new consumer hardware product by March 31, 2026?",
            "hardware",
        ),
        ("Will Waymo launch in Nashville by June 30 2026?", "hardware"),
        ("Will Elon Musk buy OpenAI before May?", "other"),
    ],
)
def test_classify(title: str, category: str) -> None:
    assert classify(title) == category


# --- planning curated questions ---------------------------------------------


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 12, tzinfo=UTC)


def market_row(**fields: Any) -> MarketRow:
    values: dict[str, Any] = {
        "id": 1,
        "external_id": "555",
        "title": "GPT-7 released by December 31, 2026?",
        "resolution_criteria": "Yes if OpenAI releases GPT-7.",
        "url": "https://polymarket.com/event/gpt-7",
        "opened_at": utc(2026, 6, 1),
        "closes_at": utc(2026, 12, 31),
        "resolved_at": None,
        "resolved_outcome": None,
        "metadata": {"closed_at": None},
        **fields,
    }
    return MarketRow(**values)


def leads(m: MarketRow, today: date) -> list[int]:
    planned, _ = plan_curated_questions([m], today)
    return [q.lead_days for q in planned]


def test_leads_are_asked_once_due() -> None:
    m = market_row()
    assert leads(m, date(2026, 10, 1)) == []  # 90 days before is 2026-10-02
    assert leads(m, date(2026, 11, 15)) == [90]
    assert leads(m, date(2026, 12, 28)) == [7, 30, 90]


def test_a_lead_before_the_market_opened_is_skipped() -> None:
    assert leads(market_row(opened_at=utc(2026, 11, 1)), date(2026, 12, 28)) == [7, 30]


def test_leads_after_an_early_resolution_are_skipped() -> None:
    m = market_row(resolved_at=utc(2026, 11, 20), resolved_outcome="yes")
    assert leads(m, date(2027, 1, 5)) == [90]  # asked 10-02; not 12-01 or 12-24
    closed = market_row(metadata={"closed_at": "2026-12-01T09:00:00+00:00"})
    assert leads(closed, date(2027, 1, 5)) == [90]  # closed on the 30-day ask date


def test_unresolvable_and_unknown_markets_are_skipped() -> None:
    fifty_fifty = market_row(
        metadata={
            "closed_at": "2027-01-01T00:00:00+00:00",
            "uma_resolution_status": "resolved",
        }
    )
    assert leads(fifty_fifty, date(2027, 1, 5)) == []
    stale = market_row(metadata={})  # stored before closes were recorded
    assert leads(stale, date(2027, 1, 5)) == []
    assert leads(market_row(closes_at=None), date(2027, 1, 5)) == []


def test_dropped_categories_are_never_asked() -> None:
    planned, categories = plan_curated_questions(
        [
            market_row(),
            market_row(
                id=2, external_id="556", title="ChatGPT Full Outage by March 31?"
            ),
        ],
        date(2026, 12, 28),
    )
    assert {q.market.external_id for q in planned} == {"555"}
    assert categories == {"release": 1, "outage_app": 1}


# --- loading curated questions ----------------------------------------------


def platform_question(external_id: str = "555", **fields: Any) -> PlatformQuestion:
    values: dict[str, Any] = {
        "platform": "polymarket",
        "external_id": external_id,
        "title": "GPT-7 released by December 31, 2026?",
        "resolution_criteria": "Yes if OpenAI releases GPT-7.",
        "url": "https://polymarket.com/event/gpt-7",
        "opened_at": utc(2026, 6, 1),
        "closes_at": utc(2026, 12, 31),
        "metadata": {"closed_at": None},
        **fields,
    }
    return PlatformQuestion(**values)


def test_curated_questions_link_their_market(migrated: Connection) -> None:
    market_pk, _, _ = upsert_platform_question(migrated, platform_question())
    result = load_curated_questions(migrated, date(2026, 12, 1))
    migrated.commit()

    assert result.created == 2
    assert rows(
        migrated,
        """
        SELECT question_key, target::text, question_type::text, target_date,
               lead_days, horizon_months, platform_question_id, question_text,
               resolution_rule
        FROM questions ORDER BY lead_days
        """,
    ) == [
        (
            "polymarket:555",
            "ai_binary_event",
            "binary",
            date(2026, 12, 31),
            lead,
            None,
            market_pk,
            "GPT-7 released by December 31, 2026?",
            (
                "Yes if OpenAI releases GPT-7.\n\nResolves as Polymarket market 555"
                " does (https://polymarket.com/event/gpt-7)."
            ),
        )
        for lead in (30, 90)
    ]


def test_loading_daily_adds_each_lead_once(migrated: Connection) -> None:
    upsert_platform_question(migrated, platform_question())
    assert load_curated_questions(migrated, date(2026, 12, 1)).created == 2
    assert load_curated_questions(migrated, date(2026, 12, 2)).created == 0
    later = load_curated_questions(migrated, date(2026, 12, 24))
    assert [q.lead_days for q, _ in later.new] == [7]
    assert rows(migrated, "SELECT count(*) FROM questions") == [(3,)]


def test_numeric_questions_cant_link_a_market(migrated: Connection) -> None:
    market_pk, _, _ = upsert_platform_question(migrated, platform_question())
    with pytest.raises(psycopg2.errors.CheckViolation) as exc:
        get_or_create_question(
            migrated,
            "hf_new_models",
            date(2026, 11, 1),
            1,
            question_key="HF_NEW_MODELS",
            platform_question_id=market_pk,
        )
    assert exc.value.diag.constraint_name == "questions_platform_link_binary"


def test_economy_resolution_ignores_binary_questions(migrated: Connection) -> None:
    upsert_platform_question(migrated, platform_question())
    load_curated_questions(migrated, date(2026, 12, 1))
    assert unresolved_questions(migrated) == []


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def pool(test_dsn: str) -> Iterator[None]:
    db.init_pool(test_dsn, maxconn=2)
    with db.connection() as conn:
        migrate(conn)
        store(conn, ARXIV, (SEP, date(2026, 10, 2), 30_000))
        upsert_platform_question(conn, platform_question())
    try:
        yield
    finally:
        db.close_pool()


def count_questions() -> int:
    with db.connection() as conn:
        return int(rows(conn, "SELECT count(*) FROM questions")[0][0])


def test_cli_dry_run_writes_nothing(
    pool: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--date", "2026-10-15", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "numeric questions=21 created=21" in out
    assert "threshold questions=9 created=9" in out
    assert "2026-12-31  90d  release   polymarket:555" in out
    assert "curated questions=1 created=1" in out
    assert count_questions() == 0


def test_cli_creates_all_kinds(pool: None, capsys: pytest.CaptureFixture[str]) -> None:
    # The asking month is December, so the base is November: not stored.
    assert main(["--date", "2026-12-01"]) == 0
    out = capsys.readouterr().out
    assert "skipped ARXIV_AI_SUBMISSIONS: no value for last month yet" in out
    assert "markets by category: release 1" in out
    assert count_questions() == 21 + 2
    assert main(["thresholds", "--date", "2026-10-15"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "threshold questions=9 created=9"
    )

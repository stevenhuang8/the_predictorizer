"""Tests for the AI numeric question generator (AI task 9)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.ai import (
    ingest_arxiv,
    ingest_epoch,
    ingest_github,
    ingest_huggingface,
)
from eco_prediction.ai.questions import (
    NUMERIC_TARGETS,
    generate_numeric_questions,
    main,
    plan_numeric_questions,
)
from eco_prediction.db import connection as db
from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import migrate
from eco_prediction.db.resolutions import unresolved_questions

AS_OF = date(2026, 10, 15)
SERIES_IDS = [
    "ARXIV_AI_SUBMISSIONS",
    "HF_NEW_MODELS",
    "GITHUB_AI_ACTIVITY",
    "EPOCH_GPQA_SOTA",
    "EPOCH_SWEBENCH_SOTA",
    "EPOCH_ARC2_SOTA",
    "EPOCH_LARGEST_FLOPS",
]


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def rows(conn: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# --- planning ---------------------------------------------------------------


def test_every_series_is_asked_at_every_horizon() -> None:
    planned = plan_numeric_questions(AS_OF)
    assert len(planned) == 21
    assert {(q.series_id, q.horizon_months, q.target_date) for q in planned} == {
        (s, h, month)
        for s in SERIES_IDS
        for h, month in [
            (1, date(2026, 11, 1)),
            (3, date(2027, 1, 1)),
            (6, date(2027, 4, 1)),
        ]
    }


def test_targets_match_the_series() -> None:
    targets = {q.series_id: q.target for q in plan_numeric_questions(AS_OF)}
    assert targets == {
        "ARXIV_AI_SUBMISSIONS": "arxiv_ai_submissions",
        "HF_NEW_MODELS": "hf_new_models",
        "GITHUB_AI_ACTIVITY": "github_ai_activity",
        "EPOCH_GPQA_SOTA": "benchmark_sota",
        "EPOCH_SWEBENCH_SOTA": "benchmark_sota",
        "EPOCH_ARC2_SOTA": "benchmark_sota",
        "EPOCH_LARGEST_FLOPS": "largest_training_run",
    }


def test_targets_use_the_ingested_series() -> None:
    # The question names the exact series its ingestion writes.
    ingested = {
        ingest_arxiv.SERIES.info,
        ingest_huggingface.SERIES.info,
        ingest_github.SERIES.info,
        *(s.series.info for s in ingest_epoch.SERIES.values()),
    }
    assert {t.series for t in NUMERIC_TARGETS} <= ingested
    assert ingest_github.STARS_SERIES not in {t.series for t in NUMERIC_TARGETS}


def test_the_target_month_is_counted_from_the_as_of_month() -> None:
    # Any day of a month asks the same questions; December rolls into the next year.
    assert plan_numeric_questions(date(2026, 10, 1)) == plan_numeric_questions(
        date(2026, 10, 31)
    )
    dec = plan_numeric_questions(date(2026, 12, 31), horizons=[1])
    assert {q.target_date for q in dec} == {date(2027, 1, 1)}


def test_resolution_rules_name_the_month_and_series() -> None:
    q = next(
        q
        for q in plan_numeric_questions(AS_OF, horizons=[3])
        if q.series_id == "EPOCH_SWEBENCH_SOTA"
    )
    assert q.resolution_rule == (
        "Best SWE-bench Verified score among models released by month end "
        "(Epoch AI), for 2027-01 (EPOCH_SWEBENCH_SOTA, Percent): the first live "
        "value stored after the month ends (not a reconstructed backfill). "
        "Later revisions don't change it."
    )


# --- storing ----------------------------------------------------------------


def test_questions_are_stored_as_numeric_ai_questions(migrated: Connection) -> None:
    result = generate_numeric_questions(migrated, AS_OF)
    migrated.commit()
    assert result.created == 21
    stored = rows(
        migrated,
        """
        SELECT target::text, question_type::text, domain, question_key,
               target_date, horizon_months, lead_days, resolution_rule
        FROM questions ORDER BY id
        """,
    )
    assert len(stored) == 21
    for (q, _), row in zip(result.questions, stored):
        assert row == (
            q.target,
            "numeric_interval",
            "ai",
            q.series_id,
            q.target_date,
            q.horizon_months,
            None,
            q.resolution_rule,
        )


def test_generating_twice_reuses_the_questions(migrated: Connection) -> None:
    first = generate_numeric_questions(migrated, AS_OF)
    migrated.commit()
    again = generate_numeric_questions(migrated, date(2026, 10, 28))
    assert again.created == 0
    assert [i for _, i in again.questions] == [i for _, i in first.questions]
    assert rows(migrated, "SELECT count(*) FROM questions") == [(21,)]


def test_next_month_asks_new_questions(migrated: Connection) -> None:
    generate_numeric_questions(migrated, AS_OF)
    nov = generate_numeric_questions(migrated, date(2026, 11, 2))
    # December h1, February h3 and May h6 are all new.
    assert nov.created == 21
    assert rows(migrated, "SELECT count(*) FROM questions") == [(42,)]


def test_ai_questions_need_a_key_and_economy_ones_none(migrated: Connection) -> None:
    with pytest.raises(ValueError, match="question_key"):
        get_or_create_question(migrated, "hf_new_models", date(2026, 11, 1), 1)
    with pytest.raises(ValueError, match="question_key"):
        get_or_create_question(
            migrated, "cpi_yoy", date(2026, 11, 1), 1, question_key="CPIAUCSL"
        )


def test_economy_resolution_ignores_ai_questions(migrated: Connection) -> None:
    # The economy job resolves from FRED series and would fail on an AI target.
    cpi, _ = get_or_create_question(migrated, "cpi_yoy", date(2026, 11, 1), 1)
    generate_numeric_questions(migrated, AS_OF)
    assert [q.id for q in unresolved_questions(migrated)] == [cpi]


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def pool(test_dsn: str) -> Iterator[str]:
    db.init_pool(test_dsn, maxconn=2)
    with db.connection() as conn:
        migrate(conn)
    try:
        yield test_dsn
    finally:
        db.close_pool()


def test_cli_dry_run_prints_without_a_database(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main(["numeric", "--date", "2026-10-15", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "2026-11  h1  arxiv_ai_submissions  ARXIV_AI_SUBMISSIONS"
    assert out[-1] == "21 questions"


def test_cli_creates_then_reuses(pool: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["numeric", "--date", "2026-10-15"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "numeric questions=21 created=21"
    assert main(["numeric", "--date", "2026-10-20"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "numeric questions=21 created=0"

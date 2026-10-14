"""Tests for the AI domain's schema additions (migrations 009 and 010)."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.db.forecasts import get_or_create_question
from eco_prediction.db.migrate import MIGRATIONS_DIR, migrate

NOW = datetime(2026, 10, 15, 13, 0, tzinfo=UTC)
SOTA_RULE = "Highest SWE-bench Verified score on the Epoch AI Benchmarking Hub"


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def add_question(conn: Connection, **cols: Any) -> int:
    cols = {
        "target": "ai_binary_event",
        "question_type": "binary",
        "domain": "ai",
        "target_date": date(2027, 6, 30),
        "lead_days": 90,
        "question_key": "swe_bench_verified>=80",
        **cols,
    }
    names = ", ".join(cols)
    marks = ", ".join(["%s"] * len(cols))
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO questions ({names}) VALUES ({marks}) RETURNING id",
            tuple(cols.values()),
        )
        return int(cur.fetchone()[0])  # type: ignore[index]


def add_resolution(conn: Connection, question_id: int, **cols: Any) -> None:
    cols = {"resolved_at": NOW, **cols}
    names = ", ".join(cols)
    marks = ", ".join(["%s"] * len(cols))
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO resolutions (question_id, {names}) VALUES (%s, {marks})",
            (question_id, *cols.values()),
        )


def rows(conn: Connection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def test_binary_question_with_text_and_lead_days(migrated: Connection) -> None:
    qid = add_question(
        migrated,
        question_text="Will SOTA on SWE-bench Verified exceed 80% by 2027-06-30?",
        resolution_rule=SOTA_RULE,
    )
    add_resolution(migrated, qid, actual_outcome="yes", actual_as_of=date(2027, 7, 1))
    migrated.commit()
    assert rows(
        migrated,
        "SELECT q.domain, q.lead_days, r.actual_outcome FROM questions q "
        "JOIN resolutions r ON r.question_id = q.id",
    ) == [("ai", 90, "yes")]


def test_numeric_ai_question(migrated: Connection) -> None:
    add_question(
        migrated,
        target="arxiv_ai_submissions",
        question_type="numeric_interval",
        horizon_months=3,
        lead_days=None,
        question_key=None,
    )
    migrated.commit()
    assert rows(migrated, "SELECT target, domain FROM questions") == [
        ("arxiv_ai_submissions", "ai")
    ]


def test_binary_questions_on_the_same_date_need_different_keys(
    migrated: Connection,
) -> None:
    add_question(migrated)
    add_question(migrated, question_key="gpqa_diamond>=90")
    migrated.commit()
    with pytest.raises(psycopg2.errors.UniqueViolation):
        add_question(migrated)


@pytest.mark.parametrize(
    ("cols", "constraint"),
    [
        # Each case breaks exactly one rule.
        (
            {"question_type": "probability", "question_key": None},
            "questions_target_type",
        ),
        ({"target": "fomc_decision", "domain": "economy"}, "questions_target_type"),
        (
            {"target": "hf_new_models", "horizon_months": 3, "lead_days": None},
            "questions_target_type",
        ),
        (
            {
                "target": "fomc_decision",
                "question_type": "probability",
                "question_key": None,
            },
            "questions_target_domain",
        ),
        ({"domain": "economy"}, "questions_target_domain"),
        ({"domain": "space"}, "questions_domain_check"),
        ({"question_key": None}, "questions_key_iff_binary"),
        (
            {
                "target": "cpi_yoy",
                "question_type": "numeric_interval",
                "domain": "economy",
            },
            "questions_key_iff_binary",
        ),
        ({"lead_days": None}, "questions_one_horizon"),
    ],
)
def test_invalid_questions_are_rejected(
    migrated: Connection, cols: dict[str, Any], constraint: str
) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation) as exc:
        add_question(migrated, **cols)
    assert exc.value.diag.constraint_name == constraint


@pytest.mark.parametrize("outcome", ["maybe", "YES"])
def test_invalid_outcomes_are_rejected(migrated: Connection, outcome: str) -> None:
    qid = add_question(migrated)
    with pytest.raises(psycopg2.errors.CheckViolation) as exc:
        add_resolution(migrated, qid, actual_outcome=outcome)
    assert exc.value.diag.constraint_name == "resolutions_actual_outcome_check"


def test_economy_defaults_are_unchanged(migrated: Connection) -> None:
    # The economy's own code path, which knows nothing about the new columns.
    qid, created = get_or_create_question(migrated, "cpi_yoy", date(2026, 11, 1), 1)
    assert created
    assert get_or_create_question(migrated, "cpi_yoy", date(2026, 11, 1), 1) == (
        qid,
        False,
    )
    with migrated.cursor() as cur:
        cur.execute("INSERT INTO sources (name) VALUES ('FRED') RETURNING id")
        source_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO series (series_id, source_id) VALUES ('CPIAUCSL', %s) RETURNING id",
            (source_id,),
        )
        series_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO observations (series_id, observed_at, as_of, value) "
            "VALUES (%s, '2026-09-01', '2026-10-15', 3.1)",
            (series_id,),
        )
    assert rows(migrated, "SELECT domain, question_key FROM questions") == [
        ("economy", None)
    ]
    assert rows(migrated, "SELECT domain FROM series") == [("economy",)]
    assert rows(migrated, "SELECT reconstructed FROM observations") == [(False,)]


def test_existing_economy_rows_survive_the_migration(
    conn: Connection, tmp_path: Path
) -> None:
    """A database at 008 with economy data, like the live one, migrates cleanly."""
    for path in sorted(MIGRATIONS_DIR.glob("00[1-8]_*.sql")):
        shutil.copy(path, tmp_path)
    migrate(conn, tmp_path)
    cpi, _ = get_or_create_question(conn, "cpi_yoy", date(2026, 11, 1), 3)
    fomc, _ = get_or_create_question(
        conn, "fomc_decision", date(2026, 10, 28), lead_days=7
    )
    add_resolution(conn, fomc, actual_outcome="hold")
    conn.commit()

    applied = migrate(conn)

    assert [m.version for m in applied][:2] == [9, 10]  # later ones may follow
    assert rows(conn, "SELECT id, domain, question_key FROM questions ORDER BY id") == [
        (cpi, "economy", None),
        (fomc, "economy", None),
    ]
    assert rows(conn, "SELECT actual_outcome FROM resolutions") == [("hold",)]
    # The economy's uniqueness still holds after the constraint was rebuilt.
    assert get_or_create_question(conn, "cpi_yoy", date(2026, 11, 1), 3) == (cpi, False)

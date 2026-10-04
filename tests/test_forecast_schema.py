"""Tests for the questions/resolutions schema (database fixtures in conftest.py)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.db.migrate import migrate

NOW = datetime(2026, 10, 15, 13, 0, tzinfo=UTC)
CPI_RULE = "CPIAUCSL YoY % change for the target month, first release"


@pytest.fixture
def migrated(conn: Connection) -> Connection:
    migrate(conn)
    return conn


def add_question(
    conn: Connection,
    target: str = "cpi_yoy",
    question_type: str = "numeric_interval",
    horizon: int = 1,
    target_date: date = date(2026, 9, 1),
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO questions
                (target, question_type, horizon_months, target_date, resolution_rule)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (target, question_type, horizon, target_date, CPI_RULE),
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


def test_questions_for_upcoming_releases(migrated: Connection) -> None:
    for horizon in (1, 3, 6):
        add_question(migrated, "cpi_yoy", horizon=horizon)
        add_question(migrated, "unemployment", horizon=horizon)
    add_question(migrated, "fomc_decision", "probability", 1, date(2026, 10, 28))
    migrated.commit()

    with migrated.cursor() as cur:
        cur.execute("SELECT target, count(*) FROM questions GROUP BY 1 ORDER BY 1")
        assert cur.fetchall() == [
            ("cpi_yoy", 3),
            ("unemployment", 3),
            ("fomc_decision", 1),
        ]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, psycopg2.errors.UniqueViolation),  # same target/date/horizon
        ({"horizon": 0}, psycopg2.errors.CheckViolation),
        ({"target": "gdp"}, psycopg2.errors.InvalidTextRepresentation),
        ({"question_type": "probability"}, psycopg2.errors.CheckViolation),
        ({"target": "fomc_decision"}, psycopg2.errors.CheckViolation),
    ],
)
def test_invalid_questions_are_rejected(
    migrated: Connection, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    add_question(migrated)
    with pytest.raises(error):
        add_question(migrated, **kwargs)


def test_numeric_and_categorical_resolutions(migrated: Connection) -> None:
    cpi = add_question(migrated)
    fomc = add_question(migrated, "fomc_decision", "probability", 1, date(2026, 10, 28))
    add_resolution(migrated, cpi, actual_value=2.9, actual_as_of=date(2026, 10, 15))
    add_resolution(migrated, fomc, actual_outcome="hold")
    migrated.commit()

    with migrated.cursor() as cur:
        cur.execute(
            """
            SELECT q.target, r.actual_value::float8, r.actual_outcome, r.actual_as_of
            FROM resolutions r JOIN questions q ON q.id = r.question_id
            ORDER BY q.id
            """
        )
        assert cur.fetchall() == [
            ("cpi_yoy", 2.9, None, date(2026, 10, 15)),
            ("fomc_decision", None, "hold", None),
        ]


@pytest.mark.parametrize(
    ("cols", "error"),
    [
        ({}, psycopg2.errors.CheckViolation),  # no actual at all
        (
            {"actual_value": 2.9, "actual_outcome": "hold"},
            psycopg2.errors.CheckViolation,
        ),
        ({"actual_outcome": "pause"}, psycopg2.errors.CheckViolation),
        (
            {"actual_value": 2.9, "resolved_at": None},
            psycopg2.errors.NotNullViolation,
        ),
    ],
)
def test_invalid_resolutions_are_rejected(
    migrated: Connection, cols: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        add_resolution(migrated, add_question(migrated), **cols)


def test_one_resolution_per_question(migrated: Connection) -> None:
    question = add_question(migrated)
    add_resolution(migrated, question, actual_value=2.9)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        add_resolution(migrated, question, actual_value=3.0)


def test_resolution_needs_existing_question(migrated: Connection) -> None:
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        add_resolution(migrated, 999, actual_value=2.9)


def test_deleting_question_deletes_its_resolution(migrated: Connection) -> None:
    question = add_question(migrated)
    add_resolution(migrated, question, actual_value=2.9)
    with migrated.cursor() as cur:
        cur.execute("DELETE FROM questions WHERE id = %s", (question,))
        cur.execute("SELECT count(*) FROM resolutions")
        assert cur.fetchone() == (0,)

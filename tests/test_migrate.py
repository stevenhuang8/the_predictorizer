"""Integration tests for the migration runner (database fixtures in conftest.py)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import psycopg2
import pytest
from psycopg2.extensions import connection as Connection

from eco_prediction.db.migrate import MIGRATIONS_DIR, MigrationError, discover, migrate


def test_discover_ignores_init_subdir() -> None:
    names = [m.path.name for m in discover()]
    assert names[0] == "001_initial_schema.sql"
    assert all(m.path.parent == MIGRATIONS_DIR for m in discover())


def test_migrate_creates_schema_and_is_idempotent(conn: Connection) -> None:
    applied = migrate(conn)
    assert [m.version for m in applied] == [m.version for m in discover()]
    assert migrate(conn) == []

    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = {row[0] for row in cur.fetchall()}
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'observations'")
        indexes = {row[0] for row in cur.fetchall()}
    assert {"sources", "series", "observations", "schema_migrations"} <= tables
    assert "idx_obs_as_of" in indexes


def test_vintage_point_in_time_query(conn: Connection) -> None:
    migrate(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sources (name) VALUES ('FRED') RETURNING id")
        source_id = cur.fetchone()[0]  # type: ignore[index]
        cur.execute(
            "INSERT INTO series (source_id, series_id, frequency) VALUES (%s, 'UNRATE', 'monthly') RETURNING id",
            (source_id,),
        )
        series_pk = cur.fetchone()[0]  # type: ignore[index]
        # March 2024 unemployment: first print, then a revision.
        cur.executemany(
            "INSERT INTO observations (series_id, observed_at, as_of, value) VALUES (%s, %s, %s, %s)",
            [
                (series_pk, date(2024, 3, 1), date(2024, 4, 5), 3.8),
                (series_pk, date(2024, 3, 1), date(2024, 5, 3), 3.9),
            ],
        )

        query = """
            SELECT o.value FROM observations o JOIN series s ON o.series_id = s.id
            WHERE s.series_id = 'UNRATE' AND o.observed_at = %s AND o.as_of <= %s
            ORDER BY o.as_of DESC LIMIT 1
        """
        results = {}
        for as_of in (date(2024, 4, 1), date(2024, 4, 20), date(2024, 6, 1)):
            cur.execute(query, (date(2024, 3, 1), as_of))
            row = cur.fetchone()
            results[as_of] = None if row is None else float(row[0])

        # Same (series, period, vintage) twice is rejected.
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO observations (series_id, observed_at, as_of, value) VALUES (%s, %s, %s, 4.0)",
                (series_pk, date(2024, 3, 1), date(2024, 4, 5)),
            )
    conn.rollback()

    assert results == {
        date(2024, 4, 1): None,
        date(2024, 4, 20): 3.8,
        date(2024, 6, 1): 3.9,
    }


def test_edited_migration_is_rejected(conn: Connection, tmp_path: Path) -> None:
    path = tmp_path / "001_first.sql"
    path.write_text("CREATE TABLE a (id INT);")
    migrate(conn, tmp_path)
    path.write_text("CREATE TABLE a (id BIGINT);")
    with pytest.raises(MigrationError, match="modified after being applied"):
        migrate(conn, tmp_path)


def test_failed_migration_leaves_no_trace(conn: Connection, tmp_path: Path) -> None:
    (tmp_path / "001_ok.sql").write_text("CREATE TABLE ok (id INT);")
    (tmp_path / "002_bad.sql").write_text(
        "CREATE TABLE half (id INT); SELECT nonsense_fn();"
    )
    with pytest.raises(MigrationError, match="002_bad.sql failed"):
        migrate(conn, tmp_path)
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations")
        assert [r[0] for r in cur.fetchall()] == [1]
        cur.execute("SELECT to_regclass('half')")
        assert cur.fetchone() == (None,)

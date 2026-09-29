"""Apply numbered SQL migrations from migrations/ and track them in schema_migrations.

Usage:
    uv run python -m eco_prediction.db.migrate           # apply pending migrations
    uv run python -m eco_prediction.db.migrate --status  # show applied/pending

Migration files are named NNN_description.sql. Each one runs in its own
transaction together with its schema_migrations row, so a failure leaves no
partial state. Applied files are checksummed; editing one afterwards is an
error. Add a new migration instead.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as Connection

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
_FILENAME_RE = re.compile(r"^(\d{3,})_([a-z0-9_]+)\.sql$")
# Arbitrary constant key so concurrent runners serialize instead of racing.
_LOCK_KEY = 7_345_001


class MigrationError(Exception):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text()

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Return migrations in version order. Subdirectories (e.g. init/) are ignored."""
    migrations = []
    for path in directory.glob("*.sql"):
        match = _FILENAME_RE.match(path.name)
        if not match:
            raise MigrationError(
                f"Bad migration filename: {path.name} (want NNN_name.sql)"
            )
        migrations.append(Migration(int(match[1]), match[2], path))
    migrations.sort(key=lambda m: m.version)
    versions = [m.version for m in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError(f"Duplicate migration versions in {directory}")
    return migrations


def _ensure_table(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                checksum   TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()


def applied_checksums(conn: Connection) -> dict[int, str]:
    _ensure_table(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations")
        rows = cur.fetchall()
    conn.commit()
    return {version: checksum for version, checksum in rows}


def pending(conn: Connection, migrations: list[Migration]) -> list[Migration]:
    """Return unapplied migrations, after checking applied ones haven't been edited."""
    applied = applied_checksums(conn)
    for m in migrations:
        if m.version in applied and applied[m.version] != m.checksum:
            raise MigrationError(
                f"{m.path.name} was modified after being applied. "
                "Revert it and add a new migration instead."
            )
    return [m for m in migrations if m.version not in applied]


def migrate(conn: Connection, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Apply all pending migrations. Returns the ones applied."""
    migrations = discover(directory)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
    try:
        todo = pending(conn, migrations)
        for m in todo:
            try:
                with conn.cursor() as cur:
                    cur.execute(m.sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                        (m.version, m.name, m.checksum),
                    )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise MigrationError(f"{m.path.name} failed: {exc}") from exc
        return todo
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
        conn.commit()


def connect() -> Connection:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise MigrationError("DATABASE_URL is not set (see .env.example)")
    return psycopg2.connect(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply numbered SQL migrations.")
    parser.add_argument(
        "--status", action="store_true", help="list migrations without applying"
    )
    parser.add_argument(
        "--dir", type=Path, default=MIGRATIONS_DIR, help="migrations directory"
    )
    args = parser.parse_args(argv)

    try:
        conn = connect()
        try:
            if args.status:
                todo = {m.version for m in pending(conn, discover(args.dir))}
                for m in discover(args.dir):
                    state = "pending" if m.version in todo else "applied"
                    print(f"{state:8} {m.path.name}")
            else:
                applied = migrate(conn, args.dir)
                for m in applied:
                    print(f"applied  {m.path.name}")
                if not applied:
                    print("Database is up to date.")
        finally:
            conn.close()
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

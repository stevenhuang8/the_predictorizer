"""Shared fixtures: a throwaway database per test on the docker-compose Postgres.

The real eco_forecast database is never touched. Tests needing a database are
skipped if Postgres isn't reachable.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg2
import pytest
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, make_dsn
from psycopg2.extensions import connection as Connection

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "")


@pytest.fixture
def test_dsn() -> Iterator[str]:
    """DSN of a freshly created, empty database; dropped after the test."""
    try:
        admin = psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    name = f"test_{uuid.uuid4().hex[:12]}"
    with admin.cursor() as cur:
        cur.execute(f"CREATE DATABASE {name}")
    params = admin.get_dsn_parameters()
    dsn = make_dsn(
        dbname=name,
        user=params["user"],
        host=params["host"],
        port=params["port"],
        password=admin.info.password,
    )
    try:
        yield dsn
    finally:
        with admin.cursor() as cur:
            # WITH (FORCE) drops it even if a pooled connection leaked.
            cur.execute(f"DROP DATABASE {name} WITH (FORCE)")
        admin.close()


@pytest.fixture
def conn(test_dsn: str) -> Iterator[Connection]:
    test_conn = psycopg2.connect(test_dsn)
    try:
        yield test_conn
    finally:
        test_conn.close()

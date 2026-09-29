"""Process-wide PostgreSQL connection pool.

Usage:
    from eco_prediction.db.connection import connection

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

`connection()` commits when the block exits normally and rolls back if it
raises, then returns the connection to the pool. The pool is created lazily
from DATABASE_URL on first use; call `init_pool(dsn)` to point it elsewhere
(e.g. a test database).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg2.extensions import connection as Connection
from psycopg2.extensions import cursor as Cursor
from psycopg2.pool import ThreadedConnectionPool

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


class ConfigError(Exception):
    pass


def database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError("DATABASE_URL is not set (see .env.example)")
    return url


def init_pool(
    dsn: str | None = None, minconn: int = 1, maxconn: int = 10
) -> ThreadedConnectionPool:
    """(Re)create the pool. Closes any existing pool first."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
        _pool = ThreadedConnectionPool(minconn, maxconn, dsn or database_url())
        return _pool


def get_pool() -> ThreadedConnectionPool:
    if _pool is None:
        return init_pool()
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def connection() -> Iterator[Connection]:
    """Borrow a pooled connection; commit on success, roll back on error."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except BaseException:
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        # A connection the server dropped is discarded rather than reused.
        pool.putconn(conn, close=bool(conn.closed))


@contextmanager
def cursor() -> Iterator[Cursor]:
    """Shortcut for `connection()` plus a cursor on it."""
    with connection() as conn, conn.cursor() as cur:
        yield cur

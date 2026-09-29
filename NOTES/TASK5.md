# Task 5: connection pool and point-in-time queries

Done on 2026-09-28.

## Quick reference

```python
from datetime import date
from eco_prediction.db.connection import connection
from eco_prediction.db.queries import (
    get_series_value_as_of, get_series_as_of,
    get_multiple_series_as_of, get_vintage_history,
)

with connection() as conn:                      # commits on success, rolls back on error
    cpi = get_series_as_of(conn, "CPIAUCSL", date(2024, 3, 20))
    df = get_multiple_series_as_of(conn, ["CPIAUCSL", "UNRATE"], date(2024, 3, 20))
```

```sh
uv run pytest tests/test_queries.py   # the tests for this task
```

---

## What I did

### `src/eco_prediction/db/connection.py`
- Keeps one connection pool for the whole process. It's created the first time it's needed, from `DATABASE_URL` in `.env`.
- `connection()` borrows a connection. When the block ends normally it saves the work (commit). If the block raises, it undoes the work (rollback). Either way, it gives the connection back to the pool.
- If the database dropped the connection, it's thrown away instead of going back into the pool.
- `cursor()` is a shortcut for a connection plus a cursor.
- `init_pool(dsn, minconn, maxconn)` points the pool at a different database, for example a test database. `close_pool()` shuts it down.
- `database_url()` reads `DATABASE_URL` and gives a clear error if it's missing. `migrate.py` now uses it too, so the setting is read in one place.

**Choice:** the task asked for psycopg2's `SimpleConnectionPool`. I used `ThreadedConnectionPool` instead, because the simple one breaks if two threads use it at once. Otherwise they work the same way.

### `src/eco_prediction/db/queries.py`
Every function takes a date and returns the data **as it was known on that date**. For each period it uses the latest version published on or before that date. Revisions published later are invisible, so backtests can't see the future.

| Function | Returns |
|---|---|
| `get_series_value_as_of(conn, series_id, observed_date, as_of_date)` | One value, or `None` |
| `get_series_as_of(conn, series_id, as_of_date, start=None, end=None)` | A pandas Series indexed by period date |
| `get_multiple_series_as_of(conn, series_ids, as_of_date, start=None, end=None)` | A DataFrame with one column per series |
| `get_vintage_history(conn, series_id, start=None, end=None)` | Every published version (`observed_at`, `as_of`, `value`), for looking at revisions |

**Things to remember**
- A value published *on* the query date counts as known.
- If the version in use is a missing value (FRED's `.`), you get `NaN` in a Series or DataFrame, or `None` from the single-value function. That means `None` from `get_series_value_as_of` can mean either "not published yet" or "published as missing".
- An unknown series code (a typo, for example) returns empty data, not an error.
- In `get_multiple_series_as_of`, a series with no data at all is left out of the columns.
- Values are cast to `float8` in SQL, so you get plain floats instead of `Decimal`.
- I build DataFrames from cursor rows instead of calling `pd.read_sql`. pandas only officially supports SQLAlchemy connections there, and warns when given a psycopg2 connection.

### Tests
- I moved the throwaway-database setup from `tests/test_migrate.py` into `tests/conftest.py` so both test files can share it:
  - `test_dsn` creates an empty database for each test and drops it afterwards.
  - `conn` is a plain connection to that database.
  - The drop uses `WITH (FORCE)`, so a connection left open by the pool can't block it.
- `tests/test_queries.py` fills the database with made-up UNRATE and CPI data and checks:
  - the value before the first release, on release day, and after a revision
  - a first release that was missing and filled in later
  - a date range
  - two series released on different days, side by side
  - empty and unknown series
  - the full list of versions
  - the pool saving work on success, undoing it on error, and replacing a dropped connection

---

## Problems
- **mypy didn't know pandas' types** (`Library stubs not installed for "pandas"`). Fix: `uv add --dev pandas-stubs`, the same way `types-psycopg2` was added in task 4.
- **mypy rejected `math.isnan(df.loc[...])`**: with the pandas type info, it can't tell that `df.loc` returns a number. Fix: used `pd.isna(...)` in those tests.

## Checks
- `uv run pytest`: 19 passed (5 migration tests, 14 new ones).
- `ruff check src tests` and `mypy src tests`: no problems.
- A query against the real `eco_forecast` database works and returns an empty Series, because no data has been loaded yet.

---

## What's next
- Task 6: the FRED/ALFRED API client. It needs a FRED API key in `.env.local`: https://fred.stlouisfed.org/docs/api/api_key.html
- Task 7 (loading the data) needs both task 5 and task 6.
- Task 5 isn't committed yet.

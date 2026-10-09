# Task 4: core database schema and migration system

Done on 2026-09-28. (These notes were written afterwards, on 2026-10-09, from the code and tests.)

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate            # apply pending migrations
uv run python -m eco_prediction.db.migrate --status   # list applied / pending
uv run pytest tests/test_migrate.py
```

To change the schema, add a new file `migrations/NNN_description.sql`. Never edit one that has already been applied.

---

## What I did

### `migrations/001_initial_schema.sql`
- **`sources`:** data providers, such as FRED. The name is unique.
- **`series`:** one row per series, such as `CPIAUCSL`. Holds the source's code, name, units, frequency and seasonal adjustment, and points to its source.
- **`observations`:** every published version of every value:
  - `observed_at`: the period the value describes, e.g. 2024-03-01 for March 2024.
  - `as_of`: the vintage date, the day this value was published (ALFRED's `realtime_start`).
  - `value`: `NULL` when the source reported the value as missing (FRED's `.`).
  - `UNIQUE (series_id, observed_at, as_of)`: one value per period per vintage.
  - **A point-in-time read** takes, for each period, the row with the latest `as_of` on or before the query date. A revision is a new row, so old versions are never overwritten.

### `src/eco_prediction/db/migrate.py`
- Finds `migrations/NNN_name.sql` files and applies them in order. A badly named file or a duplicate number is an error. Subfolders such as `init/` are ignored.
- Records each applied migration in a `schema_migrations` table (version, name, checksum, applied time), which it creates the first time it runs.
- **Each migration runs in one transaction** together with its `schema_migrations` row. If a migration fails, nothing from it is left behind.
- **Checksums:** a SHA-256 of each applied file is stored. If an applied file is edited later, the runner refuses to continue and says to add a new migration instead.
- **Advisory lock:** two runs at the same time wait for each other instead of both applying the same file.
- `--status` lists every migration as applied or pending. `--dir` points it at another folder, which the tests use.

### Where I changed the task's SQL
- **No separate `(series_id, observed_at)` index.** The `UNIQUE (series_id, observed_at, as_of)` constraint already creates an index starting with those two columns, so it serves the same lookups. A second index would only slow down inserts. `idx_obs_as_of` is kept.
- **`TIMESTAMPTZ NOT NULL DEFAULT NOW()`** instead of `TIMESTAMP DEFAULT NOW()`. Times are stored with their time zone and can't be null. Every later migration follows this.
- **`NOT NULL` on the foreign keys:** `series.source_id` and `observations.series_id`. A series without a source, or an observation without a series, makes no sense.
- **`ON DELETE CASCADE` on `observations.series_id`**, so deleting a series deletes its observations.
- **Comments in the SQL** explain `observed_at` vs `as_of` and the missing-value convention, since these are what everything else depends on.

### Later changes to this schema
- **`002_widen_series_metadata.sql`** (Task 7): `units`, `frequency` and `seasonal_adjustment` became `TEXT`. FRED's text was longer than the original limits, e.g. ICSA's frequency "Weekly, Ending Saturday" is 23 characters, over `VARCHAR(20)`.

### Tradeoffs
- **Hand-written SQL and a home-made runner instead of Alembic or an ORM.** This was a learning goal of the project: managing the schema, connections and indexes directly. It is about 150 lines, but it has no down-migrations. Undoing a change means writing a new migration.
- **Storing every vintage** makes the `observations` table much larger than storing only the latest values, and every read has to choose a version. It's the price of backtests that can't use revised data by accident.
- **`migrations/init/`** is mounted into Postgres's setup-scripts folder by `docker-compose.yml`, but those scripts only run when the database is first created. The schema itself lives in the numbered migrations, so the runner is the one source of truth. `init/` is empty.

---

## Checks
- `uv run pytest tests/test_migrate.py`: 5 passed (rerun 2026-10-09). The tests use a throwaway database (`tests/conftest.py`) and cover:
  - finding migrations, ignoring `init/`;
  - applying everything once, and a second run applying nothing; the tables and `idx_obs_as_of` exist;
  - **a point-in-time query:** March 2024 unemployment published as 3.8 and later revised to 3.9. Asking as of 2024-04-01 gives nothing, as of 2024-04-20 gives 3.8, and as of 2024-06-01 gives 3.9. Inserting the same (series, period, vintage) twice is rejected;
  - an edited, already-applied migration being refused;
  - a failing migration leaving no trace: when `002` fails, `001` stays applied, and neither the table `002` created nor a `schema_migrations` row for it remains.

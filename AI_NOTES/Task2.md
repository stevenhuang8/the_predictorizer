# AI Task 2: additive schema migration for the AI domain

Done on 2026-10-12. The live `eco_forecast` database was migrated to 010 that day, by the user, after a clean rehearsal on a copy.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate --status   # live DB: at 010 since 2026-10-12
uv run pytest tests/test_ai_schema.py
```

---

## Decisions made before starting (2026-10-12)

### Two migration files, not one
- `db/migrate.py` runs each file in one transaction. Postgres won't let a new enum value be used in the transaction that added it, and the new CHECK constraints use `'ai_binary_event'` and `'binary'`.
- So `009_ai_domain.sql` adds the enum values and columns, and `010_ai_domain_constraints.sql` adds the constraints that use them.

### Fixes to the task's SQL
- **Constraint names:** the task drops `resolutions_check1` and `questions_check` with `IF EXISTS`. Those were guesses. If a name is wrong, the drop quietly does nothing and the old cut/hold/hike-only check stays, so every yes/no resolution would be rejected. The real names are taken from the catalog (`pg_constraint`), and the drops have no `IF EXISTS`, so a wrong name fails loudly.
- **One type per target:** the task's check, `(fomc OR ai_binary) = (probability OR binary)`, would allow an FOMC question of type `binary`. Instead, FOMC must be `probability`, `ai_binary_event` must be `binary`, and every other target must be `numeric_interval`.
- **`domain` matches the target:** a CHECK ties `questions.domain` to the target, so an AI target can't be stored as `economy` or the other way round.

### Binary questions get a `question_key` (user's choice)
- **The problem:** every binary AI question uses the one target `ai_binary_event`, and questions are unique on (target, target_date, horizon). Two different binary questions resolving on the same date would collide.
- **The fix:** a nullable `question_key` (such as `metaculus:12345` or `swe_bench_verified>=80`) becomes part of that unique constraint, which keeps its name `questions_target_date_horizon_key`. Economy rows have `NULL` there (`NULLS NOT DISTINCT`), so their uniqueness and the `ON CONFLICT ON CONSTRAINT` in `db/forecasts.py` behave exactly as before.
- Binary questions must have a key. Other questions must not.
- **Rejected:** keying on `question_text`, because rewording a question would create a duplicate; and one target per kind of event, which still collides when two benchmark thresholds resolve on the same date.

### Binary questions reuse `lead_days`, not a new `horizon_days` (user's choice)
- `lead_days` already means "asked N days before the event". The FOMC week-before question uses it. A binary question asked 90 days before its resolution date means the same thing.
- **Why:** no second column with nearly the same meaning, and the existing exactly-one-horizon check works as is.
- **This departs from the PRD,** which says to add `horizon_days`.

### The live database: test on a clone first, then ask (user's choice)
- Enum values can't be removed, so migrating the real `eco_forecast` database is effectively one-way.
- Steps:
  1. Back it up with `pg_dump`.
  2. Restore the backup into a throwaway copy and migrate the copy.
  3. Run the read-only economy smoke check (deferred from AI task 1) on the copy.
  4. Report back. The user decides whether to migrate the real database.
- The cron job (`forecast_job --once-per-month`, daily at 09:00) doesn't run migrations, so the real database stays at 008 until it is migrated by hand.

---

## What I did

### `migrations/009_ai_domain.sql`
- **New `forecast_target` values:** `arxiv_ai_submissions`, `hf_new_models`, `github_ai_activity`, `benchmark_sota`, `largest_training_run` and `ai_binary_event`. **New `question_type` value:** `binary`.
- **`series.domain` and `questions.domain`:** `NOT NULL DEFAULT 'economy'`, limited to `economy` or `ai`. Existing rows get `economy`.
- **`questions.question_key` and `questions.question_text`:** both nullable.
- **`observations.reconstructed`:** `NOT NULL DEFAULT FALSE`.

### `migrations/010_ai_domain_constraints.sql`
- **`questions_target_type`** replaces `questions_check`. It is a `CASE` on the target: FOMC is `probability`, `ai_binary_event` is `binary`, everything else is `numeric_interval`.
- **`questions_target_domain`:** the three economy targets are `economy`, everything else is `ai`.
- **`questions_key_iff_binary`:** a question has a `question_key` exactly when it is binary.
- **`questions_target_date_horizon_key`** is rebuilt with `question_key` added, under the same name.
- **`resolutions_actual_outcome_check`** now also allows `yes` and `no`.

### `tests/test_ai_schema.py` (16 tests, not tagged `economy`)
- **Valid AI rows:** a binary question with text, `lead_days` and a yes resolution; a numeric AI question; two binary questions on the same date with different keys.
- **Every rejection names the constraint it should hit.** A test that only expects "some violation" could pass for the wrong reason, so each case breaks exactly one rule.
- **The economy is unchanged:** `get_or_create_question` (the economy's own code, which doesn't know the new columns) creates and then reuses a question. New series and observations get the economy defaults.
- **Upgrading from 008:** a database built from migrations 001–008 holds economy questions and an FOMC resolution. It migrates to 010 with those rows and their uniqueness intact. This is the live database's situation.

### `.gitignore`
- Ignores `backups/`, where the `pg_dump` taken before migrating goes.

## Where I changed the task's plan
- The decisions above: two files, real constraint names, one type per target, the domain check, `question_key`, and `lead_days` instead of `horizon_days`.
- **`NOT NULL` on `domain` and `reconstructed`:** the task only gave defaults. With a default already there, allowing `NULL` would only add a third state that means nothing.
- **Not done here:** which outcomes fit which question type (yes/no only for binary, cut/hold/hike only for FOMC). That spans two tables, so it belongs in `db/resolutions.py` (AI task 11), as the SQL comment says.

## Tradeoffs
- **Enum values are permanent.** Postgres can't drop an enum value. Undoing 009 means restoring the backup.
- **`questions_target_domain` lists the economy targets by name.** A new economy target would need this constraint updated, or it would be forced into `ai`. The other option was no check at all, which allows a mislabeled domain.
- **`domain` on `questions` is redundant with the target.** The PRD asked for it, and the check keeps the two from disagreeing.
- **`series.domain` has no matching check,** because a series has no target to check it against. Ingestion (AI tasks 4–7) has to set it.

## Checks
- `uv run pytest`: 332 passed, exit 0. That is the 316 economy tests, now running against the migrated schema, plus the 16 new tests. The task 1 safety net reported nothing missing or skipped.
- `ruff check`, `ruff format --check` and `mypy` pass on the new test file.
- **Rehearsal script, dry-run on a scratch database at 008** (not the live one):
  - Existing data was identical before and after migrating, and so were the economy CLI outputs.
  - The forecast job created questions on the migrated schema, all `domain = 'economy'`.
  - The copy was dropped afterwards.
- **Rehearsal on a copy of the live database** (the user ran it on 2026-10-12, because the permission system blocked me from reading the live database):
  - **Backup:** `backups/eco_forecast_pre009_2026-10-12.dump` (728K). It holds 8 questions, 22 forecasts, 0 resolutions, 0 post-mortems, 16 series and 90,993 observations.
  - **The copy migrated from 008 to 010** without errors.
  - **`DATA: identical`:** every table has the same row count and hash over its pre-migration columns.
  - **`OUTPUTS: identical`:** the economy CLI output was unchanged before and after (`inspect`, `scores --include-backtest`, and the cpi h3, unemployment h1 and fomc forecasts).
  - **The real jobs ran on the migrated copy:**
    - `forecast_job --no-ingest` exited 0 and inserted 22 forecasts (as of 2026-10-12) on the 8 existing questions, with 0 failed. It ran without `--once-per-month`, so it didn't reuse the live 2026-10-08 date.
    - `resolution_job --no-ingest` exited 0.
  - **After migrating:** questions are all `economy` (8), series all `economy` (16), and no observations are flagged `reconstructed`.
  - The copy was dropped. The live database was not changed.
- **Live migration (2026-10-12, run by the user):**
  - `migrate` printed `applied 009_ai_domain.sql` and `applied 010_ai_domain_constraints.sql`.
  - The `--status` that was meant to follow didn't run. It was typed into a normal zsh as `! cmd && ...`, and there `!` inverts the exit code, so the `&&` stopped. The migration itself is unaffected.
  - The daily job then ran on the migrated live database with `--once-per-month`: 0 failed, 22 forecasts already existed, nothing new to resolve.

## Found while migrating: the daily job loads pre-1990 history (economy, not caused by 009)
- **What happened:** the 2026-10-12 job run ingested 3,545 new DGS2 rows and 9,391 new DTB3 rows. Every other series got 0–3.
- **Cause:** economy task 21 loaded DFEDTAR, DFEDTARU, DGS2 and DTB3 from 1990 on purpose. But `forecast_job.refresh_data()` calls `ingest_many(client, V1_SERIES)` without that `observation_start`.
  - When FRED publishes a new vintage, the incremental fetch asks for the whole history.
  - `drop_unchanged` only drops values that are already stored, and the pre-1990 periods never were, so all of them are inserted.
  - The counts fit DTB3 starting in 1954 and DGS2 in 1976.
- **Effect:**
  - Those rows are stored with this week's `as_of`, so backtests as of earlier dates can't see them. There is no leakage.
  - Live fits now see a longer DGS2/DTB3 history than the backtests that validated them did.
  - It happens once per series, because the rows are stored now.
- **Fixed (2026-10-12, at the user's request), in economy code:**
  - **`data/ingest.py` `ingest_series`:** without an `observation_start`, an incremental run now starts at the earliest period already stored. It never extends a series' history backwards. That stays the job of `--full` and `scripts/backfill_data.py`. An explicit `observation_start` still wins.
    - **Rejected:** a hard-coded per-series start date in `refresh_data()`. Backfill uses 1990 for every series, so a list would have to be kept in sync by hand.
  - **`tests/test_ingest.py` `test_incremental_run_keeps_the_backfilled_range`:** a series backfilled from FEB gets a new vintage that re-reports JAN. JAN must not be inserted. The test fails without the fix: the fetch gets `observation_start=None`.
  - **`tests/economy_baseline.json`:** rewritten to include the new test (317 economy tests). This is the deliberate-change case the AI task 1 notes describe. The missing-test check passes either way, because it only looks for tests that disappeared.
  - **Cleanup SQL for the live rows:** run by the user, one transaction. It refuses if any pre-1990 DGS2/DTB3 row predates 2026-10-12, and rolls back unless the counts add up exactly.
    - **First attempt rolled back,** as designed. It expected the logged 3,545 and 9,391 to be all pre-1990 rows, but they are each one short (3,544 and 9,390). The rows span 1976-06-01 to 1989-12-29 for DGS2 and 1954-01-04 to 1989-12-29 for DTB3, which match each series' first date. The logged totals also include one ordinary new value per series from 1990 on.
    - **The corrected check:** deleted pre-1990 rows plus the post-1990 rows added today must be exactly 3544+1 and 9390+1. Tested on a scratch database shaped like the live one, for both the commit and the rollback (off by one) paths.
    - **Run by the user on 2026-10-12: committed.** It deleted DGS2 3,544 and DTB3 9,390 pre-1990 rows, and kept the one new post-1990 row per series.

## Moved the daily job from cron to launchd (2026-10-12, at the user's request)
- **Why:** cron skips a run the Mac sleeps through. The 09:00 runs on 2026-10-10, 10-11 and 10-12 never happened. launchd runs a missed `StartCalendarInterval` job when the Mac wakes; several missed days coalesce into one run. Neither catches up runs missed while the Mac is shut down.
- **`deploy/launchd/com.eco-prediction.forecast-job.plist`** is the repo copy of the agent, with install, run-now and status commands in its header comment. It's installed in `~/Library/LaunchAgents/` and bootstrapped into the GUI session. The command is the same as cron's (`forecast_job --once-per-month`, daily at 09:00). It logs to `logs/launchd.log`; `logs/cron.log` is kept as history.
- **The crontab line was removed;** the crontab is now empty. A copy of the old line was saved before removing it. The crontab write sat waiting for several minutes before it finished, which fits a macOS permission prompt.
- **README:** the Scheduling row and step 4 now describe launchd.
- **Still needs Docker (OrbStack) running at 09:00,** as it did under cron. If Postgres isn't up, that day's run fails and is logged.

# Task 9: questions and resolutions schema

Done on 2026-10-04.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate          # applies 003
uv run pytest tests/test_forecast_schema.py
```

---

## What I did

### `migrations/003_forecast_schema.sql`
- Two enum types:
  - `question_type`: `numeric_interval`, `probability`
  - `forecast_target`: `cpi_yoy`, `unemployment`, `fomc_decision`
- **`questions`**: one row per (target, target_date, horizon), with `UNIQUE (target, target_date, horizon_months)`.
- **`resolutions`**: the actual outcome of a question. There is at most one per question (`question_id` is `UNIQUE`), and it is deleted along with its question.

### Where I changed the task's SQL
- **The file is 003, not 002.** `002_widen_series_metadata.sql` already exists from Task 7. Task 10's model-tracking migration will be 004.
- **I left `score_rmse` and `score_brier` off `resolutions`.** A question has one resolution but gets forecasts from several models, so a single score per resolution has no meaning. Scores should go on the `forecasts` table in Task 10. Task 10's SQL doesn't include score columns yet, so they need adding there. (Task 23 already expects `update_forecast_score(forecast.id, ...)`.)
- **`target_date` is the period or event the question is about**, not the release date. It's the first of the month for CPI and unemployment, matching `observations.observed_at`, and the meeting date for FOMC. The task called it "when this resolves", but release dates move around and wouldn't join to the data.
- **New `actual_as_of DATE` column**: the data vintage used to resolve the question. Task 20's "bad data" post-mortem has to compare the actual at resolution time with later revisions, and this column makes that possible.
- **More constraints:**
  - `horizon_months > 0`. I didn't limit it to 1/3/6, so adding horizons later won't need a migration.
  - `fomc_decision` questions must be `probability`, and the other targets must be `numeric_interval`.
  - A resolution must have exactly one of `actual_value` or `actual_outcome`.
  - `actual_outcome` must be `cut`, `hold`, or `hike`.
  - `question_id` and `resolved_at` are `NOT NULL`.
- **Timestamps are `TIMESTAMPTZ NOT NULL DEFAULT NOW()`**, as in 001.

---

## Checks
- `uv run pytest`: 67 passed (14 new). They cover:
  - CPI and unemployment questions at 1, 3 and 6 months, plus one FOMC question
  - the uniqueness constraint, plus rejection of a bad horizon, an unknown target, and a type that doesn't match its target
  - numeric and FOMC resolutions, and rejection of a resolution with no actual, with both, with an unknown outcome, or with no `resolved_at`
  - one resolution per question, the foreign key, and cascade delete
- `ruff check src tests` and `mypy src tests`: no problems.
- Real `eco_forecast` database: migration 003 applied.

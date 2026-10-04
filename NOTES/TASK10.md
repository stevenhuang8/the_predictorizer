# Task 10: model versioning and feature snapshot schema

Done on 2026-10-04.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate          # applies 004
uv run pytest tests/test_model_tracking.py
```

---

## What I did

### `migrations/004_model_tracking.sql`
- **`model_versions`** has one row per (model_type, version_tag). It holds the hyperparameters as JSONB, the training period, and the code hash.
- **`feature_snapshots`** holds the feature values a forecast saw, as a JSONB object, together with the date the data was known as of.
- **`forecasts`** links a question to the model version and feature snapshot that produced it.
  - It stores a point estimate and interval, or class probabilities for FOMC.
  - It also has SHAP values, a rationale, and `previous_forecast_id` for revisions.
  - Indexed on `question_id` and `model_version_id`.

### Where I changed the task's SQL
- **The file is 004, not 003.** 003 was used by Task 9. Task 20's post-mortem migration will be 005.
- **Score columns on `forecasts`:** `error` (prediction − actual), `in_interval`, `brier_score`, `scored_at`. These are the scores taken off `resolutions` in Task 9, and Task 23's `update_forecast_score(forecast.id, ...)` writes them. Absolute and squared error can be derived from `error`.
- **`forecast_date DATE NOT NULL`** is the date a forecast was made as of. `created_at` is only when the row was inserted, and backtest forecasts are inserted long after their forecast date.
- **`is_backtest BOOLEAN`** (default false). Without it, the track-record page in Task 24 couldn't tell live forecasts from ones generated in hindsight.
- **Deletes:**
  - Deleting a question also deletes its forecasts, the same as its resolution.
  - A model version that has forecasts can't be deleted.
  - Deleting a forecast that was later revised sets the revision's `previous_forecast_id` to NULL.
- **Checks:**
  - A forecast needs a point prediction, probabilities, or both.
  - Interval bounds must be given together, with lower ≤ upper.
  - `parameters`, `features`, `probabilities` and `shap_values` must be JSON objects.
  - `training_start` ≤ `training_end`.
  - A forecast can't point to itself as its previous version.
- **`question_id`, `model_version_id` and `forecast_date` are `NOT NULL`.** `feature_snapshot_id` stays optional, because baselines don't use features.
- **Timestamps are `TIMESTAMPTZ NOT NULL DEFAULT NOW()`**, as in 001 and 003.

---

## Checks
- `uv run pytest`: 135 passed (16 new). They cover:
  - a forecast linked to a model version and a snapshot, with the JSON coming back intact and queryable (`features->>'oil_price'`)
  - an FOMC forecast with probabilities only
  - a revision chain, with scores filled in afterwards
  - unique model versions
  - rejecting a forecast with a bad link, no forecast date, nothing predicted, half an interval, a reversed interval, or non-object JSON
  - rejecting a bad training period or a non-object feature snapshot
  - the delete rules
- `ruff check src tests` and `mypy src tests`: no problems.
- Real `eco_forecast` database: migration 004 applied.

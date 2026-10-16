# AI Task 9: numeric AI question generator

Done on 2026-10-15. **Live database (run by the user on 2026-10-15):** `migrate` applied 011 and 012. The first run created the 21 questions for 2026-11, 2027-01 and 2027-04 (ids 82–102). The ids skip ahead because Postgres uses up a sequence value even on `ON CONFLICT DO NOTHING`, and the daily economy job's reuse lookups go through that path.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate              # applies 012_ai_numeric_question_keys
uv run python -m eco_prediction.ai.questions --dry-run  # print this month's 21 questions
uv run python -m eco_prediction.ai.questions            # create them (safe to repeat)
uv run python -m eco_prediction.ai.questions --date 2026-11-01
uv run pytest tests/test_ai_questions.py tests/test_ai_schema.py
```

---

## Decisions made before starting (2026-10-15)

### The user's choice: numeric AI questions carry their series id as `question_key`
- **The problem:** three Epoch series (`EPOCH_GPQA_SOTA`, `EPOCH_SWEBENCH_SOTA`, `EPOCH_ARC2_SOTA`) share the target `benchmark_sota`.
  - Questions are unique on (target, target_date, horizon_months, lead_days, question_key), and 010 only allowed a key on binary questions.
  - So only one benchmark could have had a question per month and horizon. Nothing on a numeric AI question said which series resolves it.
- **The fix (migration 012):** every AI question has a key, and economy questions have none.
  - For numeric questions, the key is the series id.
  - `questions_key_iff_binary` is replaced by `questions_key_iff_ai`: `(question_key IS NOT NULL) = (domain = 'ai')`.
  - Binary questions are all AI, so they still need a key. Economy uniqueness is unchanged.
- **Rejected:**
  - One enum value per benchmark: enum values are permanent, and `benchmark_sota` would be left unused.
  - Only one benchmark as a numeric target.

### My defaults (stated, not asked)
- **Same calendar as the economy.**
  - A run in month M asks about months M+1, M+3 and M+6.
  - `target_date` is the first day of the target month, and `horizon_months` is the horizon.
  - That's 7 series × 3 horizons = **21 questions a month**.
- **Step series are asked the same way:** the best value so far at the target month's end.
  - The task's alternative, "whether it will change", is a yes/no question, so it's left to the binary questions (AI task 10).
- **Resolution = the first live value stored for the month.** This is the AI counterpart of the economy's first release.
  - Reconstructed backfill rows don't count.
  - Later revisions don't change it: late Epoch evaluations, or Hugging Face models deleted after the count.
  - This is only rule text here. AI task 11 implements resolving.
- **Each rule is written per question,** naming the month, the series id and its units. It's built from the series' own `AISeriesInfo`, taken from each ingest module, so the wording can't drift from what ingestion stores.
- **`GITHUB_AI_STARS` isn't a target.** It has no `forecast_target` value and is a feature series.
- **Questions don't depend on data being present.** Creating them needs no ingested series.
- **A small CLI** (`--date`, `--dry-run`), as each ingest has. Hooking it into a daily job is AI task 21.

---

## What I did

### `migrations/012_ai_numeric_question_keys.sql`
- Drops `questions_key_iff_binary` and adds `questions_key_iff_ai`.
- Adding the constraint validates existing rows. A keyless AI question would fail the migration rather than be let through. The live database has no AI questions.

### `db/forecasts.py` `get_or_create_question` (economy code, additive)
- **New keyword arguments:** `question_key` and `question_text`. `QUESTION_TYPES` gains the AI targets.
- **`domain` is now written explicitly,** from `ECONOMY_TARGETS`.
- **A `ValueError` is raised** for an AI question without a key, or an economy question with one.
- **The reuse lookup also matches `question_key`.**
- **Economy calls are unchanged:** they pass no key and get `domain = 'economy'`, the old default.
- **The function now needs 009's columns.** The live database is at 011, so this is fine. It does mean the code can no longer write to a database still at 008.

### `db/resolutions.py`: economy readers skip AI questions
- **The bug this prevents:**
  - `unresolved_questions` feeds `resolution_job.answer()`, which looks up `TARGETS[question.target]`.
  - One AI question in the table would raise `KeyError` and break the daily economy job.
- **The fix:**
  - `unresolved_questions` and `numeric_resolutions` now filter on `q.domain = 'economy'`.
  - AI task 11 adds the AI resolution path.

### `ai/questions.py`
- `NUMERIC_TARGETS` pairs each target with its series.
- `plan_numeric_questions(as_of_date)` lists the questions without touching the database.
- `generate_numeric_questions(conn, as_of_date)` creates or reuses them and doesn't commit.
- The CLI: `--date` and `--dry-run`. The dry run needs no database.

### Tests
- **`tests/test_ai_questions.py`: 12 tests, not tagged `economy`.**
  - **Planning:**
    - 21 questions with the right months, including the December→January rollover.
    - Targets match their series.
    - The series are the ingest modules' own. Stars are excluded.
    - Any day of a month gives the same questions.
    - The exact rule text.
  - **Storing (throwaway database):**
    - Rows are `numeric_interval` and `ai`, keyed by series, with `lead_days` NULL.
    - A second run in the same month reuses all 21.
    - The next month adds 21 new questions.
    - The key/domain `ValueError`s.
    - `unresolved_questions` ignores AI questions.
  - **CLI:**
    - The dry run works with no `DATABASE_URL`.
    - Create, then reuse.
- **`tests/test_ai_schema.py` (updated for 012):**
  - A numeric AI question now has a key.
  - New test: two benchmarks on the same month and horizon need different keys.
  - New rejection case: a numeric AI question without a key.
  - Rejection cases now name `questions_key_iff_ai`. A few cases were adjusted so each still breaks exactly one rule.
  - The 008-upgrade test seeds its economy rows with raw SQL in 008's columns, because `get_or_create_question` now writes 009's.

## Checks
- `uv run pytest`: **464 passed**, including the economy safety net. No economy test changed.
- `ruff check`, `ruff format` and `mypy src` are clean.
- The dry run prints the 21 questions for 2026-11, 2027-01 and 2027-04.

## Where I changed the task's plan
- **Migration 012** (the user's choice). The task assumed the existing schema was enough.
- **`get_or_create_question` was extended** rather than copied. The task said to reuse its pattern, and AI task 10 needs the same key and text arguments for binary questions.
- **The economy resolution readers were filtered by domain.** This wasn't in the task, but the daily job would break without it.
- **Step series get numeric questions only.** "Whether it changes" goes to AI task 10.

## Left for later tasks
- **The CLI and dashboard (AI tasks 23 and 24):**
  - `eco-forecast inspect` lists AI questions too, without the key, so the three `benchmark_sota` rows look identical.
  - The dashboard's track-record queries join forecasts without a domain filter. That's harmless until AI forecasts exist.
- **The job (AI task 21):** `forecast_job.month_forecast_date` looks at every live monthly forecast. Once AI forecasts exist, it needs to filter by domain, or AI forecasts will set the economy's month date.
- **Resolving (AI task 11):** the rule says "first live value". The resolver should read the first non-reconstructed vintage with `observed_at` = the target month.

## Tradeoffs
- **First-live resolution depends on live pulls actually running.** If the user only backfills a month after it ends, that month has no live row and its questions can't resolve until a live pull stores one.
- **Epoch's first value is often incomplete.** Models are frequently evaluated weeks after release (Gemini 4 Argon: released 2026-09-30, evaluated 2026-10-13). A first-release rule scores the forecast against what was known then, not the eventual record.
- **Saturating benchmarks** (GPQA 96.4%, ARC-AGI-2 95%) still get numeric questions. They'll mostly test whether a model predicts "no change".
- **The economy code now writes `domain`,** so it needs migration 009 or later.

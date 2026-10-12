# AI Task 1: regression safety net for the economy tests

Done on 2026-10-10.

## Quick reference

```sh
uv run pytest                     # full run; also enforces the economy safety net
uv run pytest -m economy          # just the economy tests (316)
uv run pytest tests/test_regression_safety.py   # the missing-test check (~5 s)
UPDATE_ECONOMY_BASELINE=1 uv run pytest tests/test_regression_safety.py   # rewrite the baseline
```

**Baseline on 2026-10-10:** 313 existing tests passed, 0 skipped, in about 2.5 minutes. With this task's 3 new tests it is 316 economy tests, all passing.

These notes are for the `ai` Taskmaster tag. It reuses task ids 1–25, so its notes live here in `AI_NOTES/` and not in `NOTES/`, which holds the economy (`master`) tasks.

---

## Decisions made before starting (2026-10-10)

### Where the AI tasks live
- The `ai` tag was moved out of `.taskmaster/tasks/tasks.json` into `.taskmaster/tasks/ai_tasks.json`. `tasks.json` now holds only the economy (`master`) tasks.
- **Tradeoff:** the Taskmaster tools that take a `file` argument (`next_task`, `update_subtask`, `expand_task`, `get_tasks`, `parse_prd`) work on the new file. `get_task` and `set_task_status` have no `file` argument (v0.43.1) and only read `tasks.json`. For the `ai` tasks, those two jobs are done by editing `ai_tasks.json` directly.

### How the economy tests are tagged: from `conftest.py`, not per file
- `tests/conftest.py` gives the `economy` marker to every test in a fixed list of existing test files. `uv run pytest -m economy` runs just those.
- **Chosen because** none of the 22 existing test files has to be edited.
- **Tradeoffs:**
  - Reading a test file doesn't show that it's an economy test. You have to know about `conftest.py`.
  - A new economy test file must be added to the list, or it goes untagged. The other option, a `pytestmark` line in each file, has the opposite risk: a forgotten line leaves a file out.

### How a regression is detected: saved baseline, not pytest inside pytest
- The task as written wanted `test_regression_safety.py` to check that "all existing tests pass" by running the suite from inside a test. Instead:
  - A baseline file records every economy test's name and how many economy tests skip.
  - `tests/test_regression_safety.py` checks that every test in the baseline is still collected. Passing or failing is left to the normal run.
- **Chosen because:**
  - It catches a deleted or renamed test. A nested run would only see "fewer tests, all green".
  - The suite runs once, not twice, and failures show up as normal pytest output.
- **Tradeoff:** when you deliberately add or remove an economy test, the baseline must be updated.

### Skips count as a regression
- The database tests skip themselves when Postgres isn't reachable, and the live FRED tests skip without `FRED_API_KEY`. A skip isn't a failure, so with Docker down most of the safety net would quietly turn off and still look green.
- The safety net fails the run if any baseline economy test skips and isn't on the baseline's `allowed_skips` list. It checks named tests rather than a count, so one test starting to skip can't hide behind another one being fixed.

### Which database the forecast job check uses: a throwaway test database
- The check that the forecast job runs on economy targets uses a throwaway test database, never the real `eco_forecast` one. This keeps the promise in `conftest.py` that the real database is never touched. It also means a test can't add forecasts to the live track record, which only the daily job should write.
- `tests/test_forecast_job.py` already runs the whole job end to end on a throwaway database, so it is tagged and counted rather than duplicated.
- **Tradeoff:** fixture data misses problems that only appear with real data.
- **Deferred to AI task 2:** a manual, read-only smoke check against the real database. It would plan and fit but not store anything, and it would run before the schema migration.

---

## What I did

### `tests/conftest.py`
- **`ECONOMY_TEST_FILES`:** the 22 existing test files plus `test_regression_safety.py`. `pytest_collection_modifyitems` gives every test in them the `economy` marker, registered in `pytest_configure` so pytest doesn't warn about it.
- **Skip check:** `pytest_runtest_logreport` records every skipped test. `pytest_sessionfinish` then fails the run (exit code 1) if a baseline test skipped and isn't in `allowed_skips`. It prints the list in red, with a hint to check Postgres and `FRED_API_KEY`.
  - It only runs when every baseline test was collected, which means a full run or `-m economy`. Running one file or one test with Postgres down still behaves as before.

### `tests/economy_baseline.json`
- **`tests`:** the node id of every economy test (316).
- **`allowed_skips`:** tests that may skip without failing the run. It starts empty. The live FRED tests currently run because `FRED_API_KEY` is set. On a machine without the key, add their ids here.

### `tests/test_regression_safety.py`
- **`test_every_baseline_economy_test_is_still_collected`:** runs `pytest --collect-only -m economy` in a subprocess and fails if any baseline test is gone. With `UPDATE_ECONOMY_BASELINE=1` it rewrites the baseline instead, keeping `allowed_skips`.
- **`test_every_economy_test_file_exists`:** a renamed economy file would otherwise silently fall out of `-m economy` and out of the baseline on its next rewrite.
- **`test_allowed_skips_are_baseline_tests`:** catches stale entries in `allowed_skips`.

## Where I changed the task's plan
- **No nested test run.** Covered in the decisions above. The collection check does start a second pytest process, but it only collects (about 4 seconds) and doesn't run anything.
- **"Economy database queries return expected results"** is already covered by `test_queries.py`, and **"forecast_job runs on economy targets"** by `test_forecast_job.py`. Both are tagged and in the baseline, so I didn't write duplicates.
- **No coverage number.** The test strategy says "document baseline test count and coverage", but `pytest-cov` isn't a dependency. I recorded the count and didn't add a dependency for one number.

## Tradeoffs
- **Pass/fail and skips are checked differently.** Pass/fail is pytest's normal job. Skips are enforced by a session hook, so they only show at the end of a full or `-m economy` run, not when running `test_regression_safety.py` alone.
- **On a skip failure,** pytest's summary line still reads like "229 passed, 87 skipped". The red section above it and the exit code are what show the failure.
- **The baseline is a list of test names,** so it notices removed tests but not weakened ones (a test whose assertions were loosened). Review covers that.
- **New economy test files must be added to `ECONOMY_TEST_FILES` by hand.** New AI test files must stay out of it.

## Checks
- `uv run pytest`: 316 passed, exit 0. `-m economy` collects all 316.
- **Deleted test:** I renamed `test_plan_questions_for_october_2026` so pytest no longer collects it. The collection check failed with "1 economy test(s) in the baseline are gone". Restored afterwards.
- **Postgres down** (`DATABASE_URL` pointed at a dead port): `pytest -m economy` gave 229 passed and 87 skipped, listed the 87 skipped tests, and exited 1. Running only `tests/test_queries.py` the same way gave 14 skipped and exit 0, as intended.
- **Broken query:** I flipped the point-in-time filter in `db/queries.py` (`as_of <=` to `>=`). `-m economy tests/test_queries.py` gave 4 failed. Restored afterwards.
- `ruff check`, `ruff format --check` and `mypy` pass on the changed test files.

# AI Task 11: binary resolution, Brier scoring and resolving AI questions

Done on 2026-10-15. No migration and nothing written to the live database. The first AI questions resolve once 2026-11 has ended and been pulled live. AI task 21 schedules the resolver.

## Quick reference

```python
from eco_prediction.ai.resolutions import resolve_and_score_ai
result = resolve_and_score_ai(conn, date.today())   # resolved / still_open / scored
conn.commit()

from eco_prediction.db.resolutions import resolve_binary_question, scored_forecasts
resolve_binary_question(conn, question_id, "yes", date(2026, 12, 3))
scored_forecasts(conn, domain="ai", include_backtest=True)

from eco_prediction.db.forecasts import save_binary_forecast
save_binary_forecast(conn, question_id, model_version_id, date(2026, 10, 15), 0.35)

from eco_prediction.backtest import metrics
metrics.brier_score_binary(0.8, "yes")                 # 0.04
metrics.brier_decomposition(p_yes, outcomes)           # reliability/resolution/uncertainty
```

```sh
uv run pytest tests/test_binary_resolution.py
```

---

## Decisions made before starting (2026-10-15)

### The user's choice: a live pull confirms backfilled months
- **The gap:** numeric and threshold questions resolve to the month's first live value (Tasks 9 and 10). A live pull used to store a month only when it was new or its count had changed.
  - So a month backfilled after it ended, then re-counted unchanged, never got a live row, and its questions would never resolve.
- **The fix, in `ai/monthly_ingest.py`:** a live pull also stores a row for any month that has no live row yet, even when the value is unchanged.
  - "First live value" then always exists after one live pull.
  - The rule text on the 21 questions already in the live database stays true.
- **Cost:** a one-time extra row per re-counted month after a backfill. That's 3 months for arXiv, HF and GitHub, and 12 for Epoch.
- **Rejected:** resolving to the earliest stored value, backfill included. A reconstructed value (today's count, dated as if known at month end) could then resolve a question, and the stored rule text would be wrong.

### My defaults (stated, not asked)
- **Resolving AI questions is done here, not only storing resolutions.** The task only asked for `resolve_binary_question` and the metrics. But the Task 9 and 10 notes left the resolution rules to this task, and Task 10's test plan asks that sample questions "can be resolved correctly with yes/no outcomes".
  - `ai/resolutions.py` finds answers, stores them, and scores forecasts.
  - Task 21 adds the scheduled job, lock and CLI around it.
- **How each kind resolves:**

  | Kind | Answer | Known as of |
  |---|---|---|
  | Numeric (key = series id) | the first live value for the target month | that row's `as_of` |
  | Threshold (`SERIES>=T`) | yes if that same value ≥ T (equal counts as yes) | that row's `as_of` |
  | Curated (`platform_question_id`) | the market's `resolved_outcome`, once yes or no | the market's resolution date |

  - Only answers known by the given `as_of` count, so resolving as of a past date is point-in-time.
  - 50/50 and disputed markets stay open.
  - Resolutions are stored once and never revised. That matches the economy's first-release rule, and Polymarket flipping an outcome later doesn't change ours.
- **Binary forecasts are stored as `{"yes": p, "no": 1 - p}`** in `forecasts.probabilities` (`save_binary_forecast`), like any categorical forecast. They're scored on p alone.
- **Two Brier scales in one column:**
  - Binary questions get (p − outcome)², from 0 to 1. That's the task's formula and the one Polymarket and Metaculus comparisons use.
  - FOMC keeps the 3-category sum, from 0 to 2.
  - The binary score is exactly half the 2-category one, and a test pins that. They must not be averaged together, and the domain filters below keep them apart.
- **Binary calibration metrics** in `backtest/metrics.py`:
  - `log_loss_binary`, with probabilities clipped to 1e-15.
  - `expected_calibration_error`, over `calibration_curve` bins.
  - `brier_decomposition`: Murphy's reliability, resolution and uncertainty, plus the actual mean. It's exact only when forecasts in a bin are equal.
  - `brier_skill_score` against a reference (the market, or the base rate).
  - These take yes/no or 1/0 outcomes. The existing `calibration_curve` and `plot_calibration_curve` already handle binary forecasts.
- **Answers are checked against the question type** in `save_resolution`, not only in `resolve_binary_question`.
  - A value for numeric questions, cut/hold/hike for FOMC, and yes/no for binary ones.
  - Task 2 left this here because it spans two tables.
  - It costs one `SELECT` per resolution, and the economy's resolutions already satisfy it.

---

## What I did

### `ai/monthly_ingest.py`
- **`live_rows(..., confirmed)` and `live_months_stored(conn, series_pk)`:** a month with no live row gets one. Documented in the module docstring.

### `db/resolutions.py`
- **`save_resolution` validates the answer against `questions.question_type`** and raises `ValueError` on a mismatch or an unknown question.
- **New `resolve_binary_question(conn, question_id, outcome, as_of_date)`.**
- **The "guarded by domain check" part of the task:**
  - `unscored_forecasts(conn, *, domain="economy")` now joins `questions`, filters by domain, and returns `question_type`. `UnscoredForecast` gained a trailing `question_type` field with a default, so existing constructors still work.
  - `scored_forecasts(conn, *, include_backtest, domain="economy")` filters by domain and adds the `question_type` and `question_key` columns.
  - **Without this,** the economy job would score AI binary forecasts with the 3-category formula, and AI rows would mix into `eco-forecast scores`.

### `db/postmortems.py`
- **`forecasts_to_review` only reads economy forecasts.** The economy classifier knows only CPI, unemployment and FOMC. AI post-mortems are Task 22.

### `db/forecasts.py`
- **New `save_binary_forecast`.** It checks that p is in [0, 1].

### `backtest/metrics.py`
- **New functions:** `brier_score_binary`, `mean_brier_score_binary`, `log_loss_binary`, `expected_calibration_error`, `brier_decomposition` and `brier_skill_score`.
- The module docstring explains the two Brier scales.

### `ai/resolutions.py` (new)
- `unresolved_ai_questions`, `first_live_value`, `parse_threshold_key` and `answer`.
- `score(forecast)`: binary questions get a Brier score; numeric ones get `error` and `in_interval`.
- **`resolve_and_score_ai(conn, as_of)`** doesn't commit.

### Tests
- **`tests/test_binary_resolution.py`: 26 tests, not tagged `economy`.**
  - **Metrics, checked by hand:**
    - The binary Brier at p = 0, 0.5, 1 and 0.8, for both outcomes.
    - Binary is half the 2-category score.
    - Bad input raises.
    - Mean Brier: (0.1² + 0.2² + 0.6²)/3.
    - Log loss at 0.5 is ln 2, and a confident miss stays finite.
    - ECE is 0 for a calibrated set and 0.4 for an overconfident one.
    - Murphy's decomposition adds up exactly (REL 0, RES 0.09, UNC 0.25, Brier 0.16).
    - The skill score.
  - **Storing:**
    - A binary resolution is stored, and a second one fails on uniqueness.
    - Seven type mismatches each raise `ValueError` and store nothing.
    - FOMC and CPI resolutions still work.
    - The format `save_binary_forecast` stores.
  - **Resolving:**
    - Numeric and threshold questions resolve to the first live value. A backfill row and a later revision are ignored, and nothing resolves before the value is known.
    - End to end: a month backfilled after it ended resolves only after a live pull confirms it.
    - Curated questions follow their market. 50/50 and open markets stay open, and nothing resolves before the resolution date.
    - Threshold keys parse.
  - **Scoring:**
    - A numeric forecast gets error −1,000 and is in its interval.
    - A binary forecast at 0.8 resolved yes gets Brier 0.04.
    - `scored_forecasts()` (economy) doesn't see them.
    - The economy's `unscored_forecasts` returns only the FOMC forecast.
- **Ingest tests updated for the confirming rows:**
  - `test_arxiv_client.py`: the test is renamed `test_live_stores_new_changed_and_unconfirmed_months`.
  - `test_huggingface_client.py` and `test_epoch_client.py`: each expects one more live row for the unchanged backfilled month.
  - "Same counts the next day: nothing new" still holds.

## Checks
- `uv run pytest`: **531 passed**, including the economy safety net. No economy test changed.
- `ruff check`, `ruff format` and `mypy src` are clean.

## Where I changed the task's plan
- **Added `ai/resolutions.py`** (finding answers, resolving, scoring) and the confirming live rows. The task covered storing a resolution and the metrics only.
- **"Extend existing Brier score calculation":** `brier_score` keeps its multi-category meaning, and binary questions go through `brier_score_binary`. Folding binary into `brier_score` would have changed the FOMC scale.
- **Validation lives in `save_resolution`,** so every path checks it, not only `resolve_binary_question`.

## Tradeoffs
- **`forecasts.brier_score` mixes two scales across question types.** Any summary must group by `question_type`, or by domain at least. `scored_forecasts` now returns `question_type` for that.
- **The first live value can be early or incomplete.** Late Epoch evaluations and arXiv cross-lists aren't counted. That's the intended first-release rule, and Task 9's notes discuss it.
- **Frozen curated outcomes:** if Polymarket reverses a resolution after we store it, ours keeps the first one. That's rare, but it would make the market benchmark and our answer disagree.
- **One-time extra rows.** The first live pull after any backfill writes up to 12 confirming rows per series. It's harmless, but it shows up as live vintages that equal the backfilled values.

## Left for later tasks
- **Task 13:** baselines save binary forecasts with `save_binary_forecast`. The Polymarket price on the ask date (`platform_probabilities`) is the "community" baseline.
- **Task 20:** reports should group Brier by question type and by curated category (Task 10's ranking skew), and use `brier_skill_score` against the market.
- **Task 21:** the AI resolution job calls `resolve_and_score_ai` daily after the AI ingests and `ingest_polymarket`, with its own lock and `eco-forecast ai-resolve`.

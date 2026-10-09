# Task 20: post-mortems, classifying why forecasts missed

Done on 2026-10-08.

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                                   # applies 007
uv run python -m eco_prediction.postmortem.classifier                        # classify, list the misses
uv run python -m eco_prediction.postmortem.classifier --all                  # include expected variance
uv run python -m eco_prediction.postmortem.classifier --set 42 regime_change --notes "COVID"
uv run pytest tests/test_postmortem.py
```

**No cron change needed.** The daily forecast job runs resolve → score → classify, then forecasts. `--no-resolve` skips all three.

---

## What I did

### `migrations/007_postmortems.sql`
- **`miss_category`** enum: `bad_data`, `bad_model`, `regime_change`, `expected_variance`.
- **`postmortems`:** one row per scored forecast, with `forecast_id` unique and deleted along with its forecast.
  - `miss_category`, `notes`
  - `revised_data_impact`: |error vs first release| − |error vs latest value|, i.e. how much of the error later revisions took away
  - `evidence` (JSONB): every number behind the call
  - `classified_by`: `auto` or `manual`
  - `created_at`, `updated_at`

### `src/eco_prediction/postmortem/classifier.py`
Every scored forecast gets a category, including the ones that didn't miss.

**Numeric (CPI YoY, unemployment)**, first match wins:
1. **expected_variance:** |error| ≤ 1.5 × the interval's half-width, about a 95% band for an 80% normal interval. Calibrated 80% intervals miss 1 time in 5, so a near miss isn't a failure.
2. **bad_data:** the target was revised, and against the latest vintage the error is inside that band. The revision must also remove at least 25% of the error. (With no interval: at least half.)
3. **regime_change:** the actual move from the last value known at forecast time exceeds the **99th percentile of every same-length move** in the history known then. This is computed from the data, not a list of crises, and needs at least 36 past moves.
4. **bad_model:** everything else. A normal-sized move, no revision to blame.

**FOMC:**
1. **expected_variance:** the outcome had at least 20% probability.
2. **bad_model:** under 20%, but the 3-month bill had priced it. "Priced" means bill minus target ≥ +0.10 for a hike, ≤ −0.10 for a cut, or within ±0.10 for a hold. This is the task's "the model ignored a strong signal", made concrete with the model's own top feature (Task 21).
3. **regime_change:** under 20%, and the bill hadn't priced it either: a genuine surprise.
4. `bad_data` never applies, because decisions aren't revised. With no bill data, the call defaults to `bad_model`.

**How runs work:**
- `run_postmortems(conn, store, as_of)` reclassifies automatic rows on every run, because a revision months later can turn a miss into bad data.
- A **manual** classification (`--set`) is never overwritten. That is enforced in the upsert itself, so no caller can get it wrong.
- It uses only data published by `as_of`.

### `src/eco_prediction/db/postmortems.py`
- `forecasts_to_review`, `save_postmortem`, `list_postmortems`.
- `db.forecasts._json_value` became the public `json_value`, since both modules use it.

### Where I changed the task's plan
- **Migration 007, not 004.** 004 to 006 were taken by Tasks 10 and 22.
- **"Bad model: SHAP shows the model ignored a strong signal"** can't be tested as written. SHAP shows what the model used, not what it should have used. I made it concrete for each kind of forecast:
  - **numeric:** what's left after ruling out uncertainty, data and regime;
  - **FOMC:** the market had priced the decision and the model didn't.
- **"Regime change: unprecedented event (COVID, financial crisis)"** is measured against the history known at forecast time, not a list of named events, so it covers future shocks without edits.
- **"Expected variance: error within predicted interval"** became 1.5× the half-width. Using the interval edge would label every honest tail miss of a calibrated model as a failure.
- **Added** `evidence`, `classified_by` and `updated_at`, so calls can be checked, overridden, and refreshed safely.

---

## Real data
- **Your database:** migration 007 applied. Nothing is scored yet, so nothing is classified; the first post-mortems come in mid-December.
- **Smoke test in a throwaway database** (dropped afterwards): random-walk and persistence backtests, 1-month questions monthly from 2019-06 to 2021-06, resolved and classified as of today. 66 forecasts: 47 expected variance, 9 regime change, 10 bad model.
  - **Unemployment, April to November 2020: regime change.** Moves of +11.1 down to −1.7 against a historical 99th percentile of 1.4–1.6. March and December 2020 (moves of 0.9 and 1.2) are bad model.
  - **CPI 2020–21 swings of 2–3 points: bad model.** Over CPI's full history (the 1970s–80s, 2008) the 99th percentile of 3-month moves is about 3.9, so these weren't unprecedented by that standard. A random walk can't foresee anything, so for baselines "bad model" just means "missed, and not excused".
  - **The September 2019 FOMC cut: bad model.** The bill spread (−0.33) priced it and persistence gave it 6%.
  - **The July 2019 FOMC cut: regime change.** That call is wrong: fed funds futures had fully priced the cut, but T-bills that year traded cheap on heavy Treasury issuance, so the bill spread was only −0.04. This is the T-bill proxy's known weakness (Task 21); proper futures data would fix it.
  - **One rule changed after seeing this run:** March 2021 unemployment was at first labelled bad data because a 0.1 revision moved a 0.7 error just inside the band. That's a knife-edge, not the data's fault, so I added the 25% materiality condition and it is now bad model. This adjusts the classifier's definition, not a forecasting model, but the threshold was picked with this case in view.

## Tradeoffs and open questions
- **The thresholds are judgment calls:** 1.5× band, 25% revision share, 99th percentile, 20% FOMC probability, 0.10 bill spread. They're module constants with the reasoning beside them. Expect to tune them once real misses accumulate; manual overrides are there for the cases they get wrong.
- **The regime check uses all known history.** CPI's 1970s volatility makes recent swings look normal. A rolling window (say 20 years) would flag 2021–22 inflation as a regime change. Either is defensible; full history is the conservative choice.
- **Baselines get post-mortems too.** That keeps the table complete and lets you compare categories across models. For filtering, `list_postmortems` returns `model_type`.
- **No input-data revisions.** "Bad data" only considers revisions to the target. A forecast can also go wrong because a feature it used was later revised. Detecting that means re-running the model on revised inputs, which is possible later with the stored feature snapshots.

---

## Checks
- `uv run pytest`: 303 passed (18 new in `tests/test_postmortem.py`). They cover:
  - **every numeric rule:** a near miss inside the band; a revision that rescues the forecast; a small revision that only tips the band (not bad data); an April-2020-sized move; an ordinary move missed; too little history for a regime threshold; the no-interval revision rule
  - **every FOMC rule:** had a chance; priced hike, cut and hold; unpriced hike and hold; no bill data
  - **on the database:**
    - classification after resolution
    - a revision published later flipping a miss to bad data, with `revised_data_impact` and `updated_at` updated
    - the T-bill evidence stored
    - manual rows surviving reruns, and blocking automatic saves but not manual ones
    - unscored forecasts left alone
    - bad categories refused in code and by the enum
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.
- CLI: lists nothing yet on the real database, and `--set` with an unknown category exits with a clear error.

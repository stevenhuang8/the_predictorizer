# Task 21: FOMC cut/hold/hike forecaster

Done on 2026-10-08.

## Quick reference

```python
from eco_prediction.backtest.data import VintageStore
from eco_prediction.backtest.fomc import FOMCBacktest, summarize_fomc
from eco_prediction.data.fomc import RELEASE_LAGS, FOMC_MEETINGS, decision
from eco_prediction.models.fomc_model import FOMCForecaster, Persistence

store = VintageStore.from_db(backfill=True, release_lags=RELEASE_LAGS)  # RELEASE_LAGS matters
model = FOMCForecaster(lead_days=7)          # forecast 7 days before the decision day
model.fit(store, cutoff=date(2024, 12, 31))
model.predict_proba(store, date(2025, 1, 29))   # {'cut': .., 'hold': .., 'hike': ..}
model.feature_importance()

bt = FOMCBacktest(date(2005, 1, 1), date(2026, 9, 30), "annually", store=store)
summarize_fomc(bt.run(model))   # brier, accuracy, n_changes, brier_on_changes
```

```sh
uv run python -m eco_prediction.data.ingest DFEDTAR DFEDTARU DGS2 DTB3   # already loaded
uv run pytest tests/test_fomc.py
```

---

## What I did

### Data
- **Four new series in `V1_SERIES`**, loaded from 1990: `DFEDTAR` (the target rate to 2008-12-15), `DFEDTARU` (the top of the target range since), `DGS2` (2-year Treasury) and `DTB3` (3-month bill). This added 32,660 rows. The ingest and backfill defaults now cover 16 series.
- **`src/eco_prediction/data/fomc.py`**:
  - **`FOMC_MEETINGS`**: decision days of the 271 scheduled meetings from 1994 to 2027, 8 a year (7 in 2020).
    - Parsed from the raw HTML of federalreserve.gov's calendar pages, not a summary, so the dates are exactly as the Fed lists them.
    - Left out: unscheduled meetings and calls (e.g. March 2020), the cancelled March 2020 meeting, the August 2025 notation vote, and 2003-09-15, which has an agenda but no statement.
  - **`fed_funds_target(store, as_of)`**: one continuous daily series. It is DFEDTAR to 2008, then the midpoint of the range (upper − 0.125).
  - **`decision(store, meeting, as_of)`**: cut, hold or hike, from comparing the target the day before the decision day with the day after. It returns `None` if that isn't known by `as_of`.
  - **`RELEASE_LAGS`**: DFEDTAR has a single ALFRED vintage (2008-12-15). Without a 1-day lag passed to `VintageStore`, a backtest can't see any of its history before then.
- **Labels checked against known history:** the 2001 cuts, the December 2015 hike, the 2022–23 hiking cycle, the 2024 and 2025 cuts. Overall, 1994 to September 2026: 178 holds, 51 hikes, 32 cuts. The 9 moves between meetings (1994, 1998, 2001 ×3, 2008 ×2, 2020 ×2) appear in the target but aren't labels.

### `src/eco_prediction/models/fomc_model.py`
- **`FOMCForecaster(lead_days=7)`** makes a forecast `lead_days` before the decision day, using data through the day before (`forecast_cutoff`).
  - **Training rows:** every meeting from `train_start` (1994) whose decision was known at the fit cutoff.
  - **Features**, as known on each row's own forecast date:
    - the Task 14 macro vector, minus the month dummies and `trend_months` (Tasks 16 and 17 suggested dropping the trend);
    - the target and its 3/6/12-month changes;
    - the last two decisions and the meetings since the last change;
    - the 3-month bill and 2-year yields minus the target, and their 1-month changes.
  - **Tuning:** a 6-point grid scored by multi-class log loss on `TimeSeriesSplit` folds, with early stopping, then a refit on every row.
  - **`calibrate=True`** applies Task 18's `ProbabilityCalibrator`, fit on the out-of-fold probabilities. It is off by default; see the results.
- **Baselines**, with the same interface:
  - `AlwaysHold`: the task's baseline.
  - `Climatology`: how often each decision happened in training.
  - `Persistence`: P(decision | previous decision), a Markov chain.
  - Both counting baselines are add-one smoothed.

### `src/eco_prediction/backtest/fomc.py`
- **`FOMCBacktest(start, end, retrain_frequency)`**: one row per meeting with `p_cut`/`p_hold`/`p_hike`, the outcome and the Brier score. Refits are `"every_meeting"`, `"quarterly"` or `"annually"`.
- **`summarize_fomc`** reports mean Brier, accuracy, the number of non-hold decisions, and Brier on those alone. Always-hold scores well on overall Brier simply because most meetings are holds, so the score on changes is the one that separates models.
- **No lookahead, enforced:** models only get `store.until(cutoff)`, a new `VintageStore` method that returns a store with nothing published after the cutoff. This is stronger than the numeric harness's after-the-fact check: later data isn't reachable at all.

### Where I changed the task's code
- **Not `fit(X, y)` / `predict_proba(X)`.** As in Task 16, the model builds its own point-in-time rows from the store: `fit(store, cutoff)`, `predict_proba(store, meeting)`.
- **The booster API (`lgb.train`), not `LGBMClassifier`.** A training window can have no cuts at all (e.g. 2015–2018). `LGBMClassifier` would then silently become a 2-class model, and `probs[0]` would no longer mean "cut". `lgb.train` with `num_class=3` always returns three probabilities in a fixed order.
- **No CME FedWatch benchmark.** CME has no free API, and its terms forbid scraping. Fed funds futures aren't on FRED. The 3-month bill and 2-year spreads are the free stand-in for what markets expect, and they turned out to be the strongest features. A real FedWatch comparison needs paid futures data.
- **The task's 2015–2023 → 2024 test is reported but isn't the main evidence.** 2024 has 8 meetings and 3 decisions that weren't holds; one surprise moves the score by a lot. The 2005–2026 walk-forward is what I'd judge by.

---

## Results on real data

### The task's test: train 2015–2023, predict the 8 meetings of 2024 (lead 7 days)
| Model | Brier | Brier on the 3 cuts |
|---|---|---|
| Always hold | 0.750 | 2.000 |
| Climatology | 0.654 | 1.414 |
| Persistence | 0.370 | 0.758 |
| **LightGBM** | **0.185** | **0.107** |

LightGBM is 75% better than always-hold. Its only miss was July 2024, where it gave a cut 72%, one meeting early. Training from 1994 instead scored 0.254: less overconfident in July, but too hesitant on the September–December cuts. The 2015-start model's own CV log loss (1.59) was worse than a uniform guess (1.10). With 8 meetings, read this row as a check that everything runs, not as a ranking.

### Walk-forward 2005-01 to 2026-09: 173 meetings, 50 of them non-hold, annual refits, training from 1994
| Model | Brier, lead 7 days | on changes | Brier, lead 42 days | on changes |
|---|---|---|---|---|
| Always hold | 0.578 | 2.000 | 0.578 | 2.000 |
| Climatology | 0.466 | 1.131 | 0.466 | 1.131 |
| Persistence | 0.331 | 0.743 | 0.299 | 0.666 |
| **LightGBM** | **0.239** | **0.508** | 0.279 | 0.628 |
| LightGBM + calibration | 0.248 | 0.525 | 0.296 | 0.647 |

- **At 7 days, LightGBM beats the best baseline, and the margin holds up.**
  - Its Brier is 0.092 lower than persistence's. A bootstrap that resamples whole years puts the 90% interval for that gap at [−0.162, −0.027].
  - It wins in 18 of 22 years.
  - Reliability is good: in five probability bins, the predicted and observed frequencies are within 0.06 of each other.
- **At 42 days (forecasting right after the previous meeting), it isn't clearly better than persistence:** −0.020, interval [−0.050, +0.010]. The gain at 7 days comes from the T-bill spread. The bill spread is the top feature by a wide margin, and a week out it has already priced the decision. Six weeks out, it hasn't.
- **Calibration made it slightly worse** at both leads. The raw probabilities are already close to the diagonal, and isotonic regression fit on about 170 out-of-fold rows adds noise. That's why `calibrate` defaults to False.

### The largest misses (lead 7 days)
| Meeting | Forecast | Outcome |
|---|---|---|
| 2019-09-18 | hold 99% | cut |
| 2005-09-20, 2005-12-13 | hold 96–98% | hike |
| 2018-05-02, 2018-08-01, 2018-11-08 | hike 87–98% | hold |
| 2019-07-31, 2019-10-30 | hold 91–92% | cut |

- **2018: the Fed only hiked at meetings with a press conference** (March, June, September, December), and the model has no way to know which meetings those are. A press-conference/SEP flag (every meeting since 2019, quarterly from 2011 to 2018) is the obvious next feature. I didn't add it, because choosing features by looking at the test years' errors would flatter this backtest.
- **2019 cuts and 2005 hikes:** markets priced both, but the model still said hold with over 90%. Its training data had few cuts from a low rate (2019), or only 11 years in total (2005).

## Tradeoffs and open questions
- **One model per lead time.** Task 22's scheduler will forecast meetings at varying distances. Options: train one model per lead (simple, as here), or one model with days-to-meeting as a feature and rows at several leads. The second shares data, but its rows are correlated.
- **Decisions are cut/hold/hike only.** A 50 bp cut and a 25 bp cut are both "cut". The `questions` schema only has three outcomes, so I kept it.
- **The T-bill spread has quirks.** It is quoted on a discount basis and shifts with bill supply and Treasury cash management. A cleaner signal is the 1-month or 3-month OIS, but that isn't on FRED.
- **1994–2004 features come from backfilled first vintages** (Task 16's approach). That's exact for market series and leaks later revisions of CPI and unemployment into early training rows.
- **Runtime:** a 22-year walk-forward with annual refits takes about 2.5 minutes at lead 7 and 4 minutes at lead 42. Every fit rebuilds its training rows from a fresh `store.until()`, which has an empty feature cache. Caching rows by (meeting, as-of) across fits would make it several times faster.

---

## Checks
- `uv run pytest`: 252 passed (14 new in `tests/test_fomc.py`). The tests use the real calendar from 2009, random decisions, and a T-bill that prices the next decision. They cover:
  - the calendar: 8 meetings a year, 7 in 2020, sorted, the left-out dates absent
  - splicing DFEDTAR with the DFEDTARU midpoint, and the December 2008 cut
  - labels matching the generated decisions, and not known until the day after the meeting is published
  - each rate feature computed by hand, and missing market series giving NaN
  - `store.until` hiding later vintages
  - every baseline's probabilities checked by hand
  - training rows lining up with meetings and labels, with no calendar or trend features
  - the T-bill spread as the top feature, and the top class correct for every 2019 meeting
  - calibrated output summing to 1
  - a spy model confirming the backtest never hands over data past each cutoff, refits on schedule, and scores correctly
  - future meetings left unresolved
  - `summarize_fomc` computed by hand
  - LightGBM under half the best baseline's Brier in a walk-forward
  - bad input refused
- `ruff check`, `ruff format --check` and `mypy src tests`: no problems.

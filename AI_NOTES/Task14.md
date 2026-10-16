# AI Task 14: ARIMA and ETS for the AI series

Done on 2026-10-15. No migration and nothing written to the live database.

## Quick reference

```python
from eco_prediction.ai.statistical_models import AIARIMA, AIETS
from eco_prediction.ai.targets import AI_TARGETS

bt.run(AIARIMA(), AI_TARGETS["ARXIV_AI_SUBMISSIONS"], horizon_months=3)
bt.run(AIETS(), AI_TARGETS["EPOCH_SWEBENCH_SOTA"], horizon_months=6)
```

```sh
uv run pytest tests/test_ai_statistical.py   # ~25 s (statsforecast compiles on first use)
```

---

## Decisions (my defaults, not asked)

### Wrap the economy models; don't extend them
- **How it works:** `AIStatsModel` holds an economy `ARIMAModel` or `ETSModel`, unchanged. It hands that model a view of the data whose target history is on a transformed scale, then transforms the forecast and interval back.
- **Why a wrapper:** the economy's `models/statistical.py` isn't touched, and its fit/forward behavior carries over:
  - It refits at harness retrains.
  - Between retrains it runs `forward` from the latest published month.
  - It interpolates interior gaps.
- **Look-ahead check:** the transformed view reads through the original `PointInTimeData`, so the harness's `max_as_of_seen` check still sees every read. A test pins this.
- **The task's two options** were a new `ai/statistical_models.py` or a domain parameter on the economy classes. The new module keeps economy code untouched.

### One scale per series kind

| Kind | Series | Modelled as | Why |
|---|---|---|---|
| count | arXiv, HF, GitHub | log(y + offset) | Growth is multiplicative: a drift is a growth rate, and the interval scales with the level. |
| percent | GPQA, SWE-bench, ARC-AGI-2 | log(100.1 − y), the log of the headroom | Closing a share of the remaining gap is a constant step, and the back-transform can't pass 100. |
| log10 | largest training run | as is | Already log FLOP. |

**Notes on the choices:**
- **Zero counts (the task asked for this):** the offset is 0, or 1 if the training history has any zero or negative month. It's fixed at fit time, so fit and predict use the same scale.
  - Ingestion already refuses zero counts, so in practice it's 0.
  - The 0.1 headroom epsilon keeps a score of exactly 100 finite.
- **Running maxima (percent, log10) are floored at their last value** after back-transforming, as in the baselines. ARIMA and ETS don't know a record can't fall.
- **The percent scale is decreasing,** so the interval's ends swap when transformed back.
- **The task's "state-space model or just use baselines" for step series:** I used the same ARIMA and ETS on the headroom scale, plus the floor. Both stay strictly within [last, 100] in the tests. Whether they beat the random walk on real step series is for Task 20. A flat-for-months series favors the random walk.

### Other defaults
- **Seasonality:** 12 months for counts (arXiv has conference-deadline and December cycles), none for step series. Either can be set with `season_length`.
- **Back-transformed points are medians, not means:** exp of a symmetric log forecast sits below the mean. The gap is about half the log-scale variance, under 1% at typical 1–3 month spreads, so no bias correction is applied.
- **Minimum history: the economy's 24 months.**
  - ARC-AGI-2 (from 2025-03) can't be fit until 2027-03, and a fit raises `ValueError`.
  - SWE-bench (from 2024-08) is just past the limit now.
  - Fitting season-12 models on less isn't worth it. The baselines cover those series meanwhile.

---

## What I did
- **`src/eco_prediction/ai/statistical_models.py` (new):**
  - `Scale` and `scale_for(kind, history)`.
  - `AIStatsModel`, the wrapper.
  - `AIARIMA` and `AIETS`.
- **`tests/test_ai_statistical.py`: 10 tests, not tagged `economy`.** They use a synthetic arXiv-like series (3%/month growth, a December dip, seeded noise) and hand-made step series.
  - **Scales:**
    - The count log round trip, with an offset only when there's a zero.
    - The percent headroom: decreasing, 100 is finite, it can't pass 100.1, and the round trip.
    - log10 is the identity.
  - **Counts:**
    - The wrapper equals exp of the economy ARIMA run directly on the logs (the back-transform is right).
    - The interval is symmetric in ratio terms.
    - Reads go through the original view.
    - ETS forecasts continued growth.
    - A zero month switches to log(y + 1) and still forecasts.
  - **Step series:**
    - ARIMA and ETS on a saturating percentage stay within [last, 100], with the interval in order.
    - A log10 record isn't forecast below its last value.
  - **Errors:** a history under 24 months raises, and an unfitted model raises.
  - **Against the random walk:** ARIMA's RMSE is under half the random walk's in a 30-forecast walk-forward backtest, with interval coverage of at least 60%.

## Checks
- `uv run pytest`: **573 passed**, including the economy safety net.
- `ruff check`, `ruff format` and `mypy src` are clean.
- **Every numeric model on the synthetic series:** a walk-forward backtest from 2023-01 to 2025-06, refit quarterly, 3 months ahead, 30 forecasts.

  | Model | RMSE | 80% interval coverage |
  |---|---|---|
  | AIRandomWalk | 1,630 | 70% |
  | AIHistoricalMean | 3,247 | 0% |
  | AILinearTrend | 503 | 77% |
  | AIARIMA | 238 | 70% |
  | AIETS | 206 | 87% |

  - **This shows the log-scale models work as intended, not that they'll win on real data.** The synthetic series is exactly the constant growth they assume.
  - **Not run on real data:** the AI backfills haven't been run on the live database yet.

## Where I changed the task's plan
- **A wrapper module instead of extending `models/statistical.py`** (reasoning above).
- **The log of the headroom for SOTA series,** and ARIMA/ETS plus a floor rather than a separate state-space model.

## Tradeoffs
- **AutoARIMA and AutoETS re-select their orders at every fit,** so the model can change between refits. That's the same as the economy.
- **The floor clips, it doesn't reshape.** Clipped intervals on step series are narrower than the model believes, so their coverage can look better than it is.
- **The headroom scale assumes progress slows near 100.** A benchmark that jumps from 60 to 95 in one release breaks that, and no smooth model will see such a jump coming.
- **On short real histories** (HF 54 months, SWE-bench 26), AICc selection among seasonal models will be unstable. ETS with `season_length=1` may do better there, and Task 20 can compare.

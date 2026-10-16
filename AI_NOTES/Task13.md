# AI Task 13: AI baselines

Done on 2026-10-15. No migration and nothing written to the live database. The baselines are code only. Task 21 runs them live and Task 20 compares them.

## Quick reference

```python
from eco_prediction.ai.baselines import (
    AIRandomWalk, AIHistoricalMean, AILinearTrend,                # numeric
    BaseRate, MarketPrice, BinaryData, binary_questions, evaluate_binary,  # binary
)
from eco_prediction.ai.targets import AI_TARGETS
from eco_prediction.backtest.harness import WalkForwardBacktest, summarize

bt = WalkForwardBacktest(date(2024, 1, 1), date(2026, 6, 1), "monthly")
summarize(bt.run(AILinearTrend(), AI_TARGETS["HF_NEW_MODELS"], horizon_months=3))

data = BinaryData.from_db(conn)
evaluate_binary(MarketPrice(), binary_questions(conn), data)   # p_yes, outcome, brier
```

```sh
uv run pytest tests/test_ai_baselines.py
```

---

## Decisions (my defaults, not asked)

### Numeric: AI series as harness targets, not a new harness
- **`ai/targets.py` gives each of the 7 numeric series a `Target`** (the level, no transform). The economy's `WalkForwardBacktest.run(model, target, horizon)` already accepts a `Target`, so the harness runs AI series unchanged.
- **Targets are named by series id.** The three benchmarks share the enum value `benchmark_sota`, and `Target.name` is only used in messages and as the series name.
- **`kind_of(series_id)`** returns `count`, `percent` or `log10`, from the series' units.
- **Backtests resolve with the first stored vintage** (the harness's rule). Over history that's the reconstructed backfill, while live questions resolve to the first live value (Task 11). The two agree once live pulls have run.

### The three numeric baselines

| Model | Point | Interval (80%) | Counts | Running maxima |
|---|---|---|---|---|
| `AIRandomWalk` | last value | quantiles of past changes over the same number of months | changes are log ratios | floored at the last value; percentages capped at 100 |
| `AIHistoricalMean` | mean of the last 12 months | mean ± z·sd | geometric (on logs) | percentages capped at 100 |
| `AILinearTrend` | OLS line through the last 12 months, projected | OLS prediction interval, normal z | fit on logs (constant growth) | floored at the last value; percentages capped at 100 |

**Notes on the choices:**
- **Why log scale for counts:** the series grow multiplicatively. An arXiv change of +300 meant something different in 2015 than it does now, but +2% means the same.
- **Running maxima can't fall,** so a forecast below the last value is impossible. Clipping keeps the baselines from losing points on something trivial. The historical mean isn't floored, because it's meant to be naive.
- **The trend is refit from the forecast's own data** inside `predict`, so it always ends at the latest known month, even between quarterly refits.
- **The window is 12 months for both the mean and the trend** (the task said 6–12). It's configurable through `window_months`.
- **The economy's `RandomWalkModel` and `HistoricalMeanModel` are untouched.** The AI versions reuse `NotFittedError` and `_check_coverage`.

### Binary baselines

| Model | Applies to | Probability of yes |
|---|---|---|
| `ThresholdBaseRate` | threshold questions (`SERIES>=T`) | how often the series has cleared the same relative step over the same gap |
| `CuratedBaseRate` | curated questions | the yes rate of markets in the same category that resolved before the ask date |
| `BaseRate` | either | routes to the right one of the two above |
| `MarketPrice` | curated questions | the market's price the day before the ask date (the task's CommunityForecast) |

#### How the threshold base rate works
- A threshold was set from the month before the asking month. For example, T is 10% above September's value for a question asked in October about January.
- **The rule is restated relative to that base:**
  - Counts: a ratio, T/base.
  - Percentages: a share of the gap to 100.
  - log10: a difference.
- **The rate is the share of past month pairs** h+1 months apart where the later value cleared the same rule from the earlier one.
- This needs no step label (Task 10 only stores the threshold), and it works for every kind. Equality counts as yes, within a 1e-9 relative tolerance, because 200 × (110/100) isn't exactly 220 in floating point.

#### The curated base rate
- **It pools by category** (benchmark, ranking, release) from `curated_questions.classify`, over every market in `platform_questions`, not only the ones we ask about. That's more history.
- **Rankings come out near 1 in 10,** which is right for "will company X have the best model" legs.

#### Shared rules
- **Smoothing:** (yes + 1) / (n + 2), so a short record never gives 0 or 1, and no record gives 0.5.
- **Ask dates and cutoffs, following the harness's rule** (data published by the end of D − 1):
  - A curated question is asked `lead_days` before its close.
  - A threshold question is asked on the day it was created (`questions.created_at`), which is when its base value was known.
- **A model that can't answer raises `NoForecast`.** `MarketPrice` has nothing for a threshold question, and a threshold question can't be answered before its base month is known. `evaluate_binary` records that as a missing `p_yes` rather than inventing 0.5.

### A binary interface, not the task's `predict_probability(view, target_period)`
- **Binary questions need more than a series view:** a threshold, a market, a category and an ask date.
- **So the interface is `predict_probability(question: BinaryQuestion, data: BinaryData) -> {"yes": p, "no": 1 - p}`.**
  - `BinaryData` serves the AI series (a `VintageStore`), the Polymarket markets and their daily prices, each as known on a cutoff.
  - `binary_questions(conn)` loads every binary question with its ask date and outcome.
- **`evaluate_binary` is the binary counterpart of the walk-forward run:** one row per question, with `p_yes`, the outcome and the binary Brier score. Tasks 15 and 19 can plug classifiers and the LLM into the same interface.

---

## What I did
- **`ai/targets.py` (new):** `AI_TARGETS`, `KINDS` and `kind_of`.
- **`ai/baselines.py` (new):**
  - The numeric baselines `AIRandomWalk`, `AIHistoricalMean` and `AILinearTrend`.
  - The binary data and questions: `BinaryQuestion`, `binary_questions` and `BinaryData` (with `from_db`).
  - The binary models: `ThresholdBaseRate`, `CuratedBaseRate`, `BaseRate` and `MarketPrice`.
  - The helpers: `cleared`, `smoothed_rate`, `NoForecast` and `evaluate_binary`.
- **`tests/test_ai_baselines.py`: 16 tests, not tagged `economy`.** Most use an arXiv series growing exactly 2% a month, so expected values have closed forms:
  - **Targets and kinds.**
  - **Random walk:**
    - It returns the last value.
    - Its interval is exactly the 3-month growth.
    - A running max stays between its last value and 100.
  - **Historical mean:** geometric for counts (1000 × 1.02^29.5) and arithmetic for a step series.
  - **Linear trend:**
    - It projects exponential growth exactly, with zero width.
    - A percentage is capped at 100, and log10 isn't capped.
  - **Errors:** unfitted models raise, and so does an empty series.
  - **The walk-forward harness:**
    - All three run.
    - The trend's RMSE is 0 on a pure trend.
    - The random walk beats the mean.
  - **Threshold base rate:**
    - `cleared` for each kind, including equality under rounding.
    - +10% never and +3% always over a 2-month gap gives 1/36 and 35/36 after smoothing.
    - It raises before the base month is known.
  - **Curated base rate:** it counts only markets of the same category resolved before the ask date (3 yes, 1 no gives 4/6).
  - **Market price:**
    - It uses the price the day before the ask date.
    - With no price yet it raises, and it has nothing for a threshold question.
  - **`evaluate_binary`:** Brier, NaN for a question it can't answer, and categories.
  - **From the database:**
    - Ask dates: close − lead, and the created date.
    - Outcome and category.
    - `BinaryData.from_db` reads prices.

## Checks
- `uv run pytest`: **563 passed**, including the economy safety net.
- `ruff check`, `ruff format` and `mypy src` are clean.
- **Not run on real data:** the AI backfills and the first Polymarket run haven't been done on the live database yet.
  - Once they have, `evaluate_binary(MarketPrice(), …)` versus `BaseRate()` on the ~2,300 resolved curated questions is the first real comparison.

## Where I changed the task's plan
- **`ai/baselines.py` and `ai/targets.py`,** not changes to `models/baselines.py`. The economy file is unchanged.
- **The binary interface takes a question and a data object,** not `(view, target_period)` (reasoning above). The binary evaluator is `evaluate_binary`, not the harness.
- **The threshold base rate comes from the series' own history** of clearing the same relative step. The task's "similar question types" has no history of resolved threshold questions yet.
- **No Metaculus community forecast** (deferred, see Task 8).

## Tradeoffs
- **The threshold base rate is hindsight-heavy over the backfilled period.** It's computed from reconstructed history, the same caveat as the features.
- **Overlapping pairs:** consecutive month pairs share most of their span, so the effective sample is smaller than n. The smoothing only helps a little.
- **The curated base rate ignores the lead time and the event.** Ranking legs of one event are scored as independent, so a model that knows "exactly one leg is yes" can beat it easily. That's fine for a baseline.
- **The market price on the day before the ask date** can be stale for a thin market. Polymarket history is daily (Task 8), so it's at most a day old in normal trading.
- **Historical mean and linear trend on short SOTA histories** (as little as 3–4 months for ARC-AGI-2) are fragile. `MIN_TREND_POINTS` is 3.

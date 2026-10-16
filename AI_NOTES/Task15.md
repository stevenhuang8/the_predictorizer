# AI Task 15: LightGBM on the AI features

Done on 2026-10-16. No migration and nothing written to the live database. The live database has no AI observations yet, so everything was checked on synthetic data.

## Quick reference

```python
from eco_prediction.ai.lightgbm_model import AILightGBMModel, AIThresholdClassifier
from eco_prediction.models.explainability import SHAPExplainer

bt.run(AILightGBMModel(3), AI_TARGETS["ARXIV_AI_SUBMISSIONS"], horizon_months=3)
bt.run(AILightGBMModel(3, pool=True), AI_TARGETS["EPOCH_GPQA_SOTA"], horizon_months=3)
SHAPExplainer(model).explain(view, target_period)     # on the model's scale

data = BinaryData.from_db(conn)
clf = AIThresholdClassifier().fit(data, date(2026, 10, 14))
evaluate_binary(clf, binary_questions(conn), data)
```

```sh
uv run pytest tests/test_ai_lightgbm.py   # ~18 s
```

---

## Decisions (my defaults, not asked)

### Numeric: subclass the economy's `LightGBMModel`
- **`AILightGBMModel` subclasses `models/lightgbm_model.LightGBMModel`.** The economy file is unchanged. These carry over:
  - Direct forecasting, one model per horizon.
  - Change labels.
  - The grid search with early stopping.
  - Residual intervals.
- **Features:** `AIFeatureEngineer` with the target's `target_*` aliases, through a small adapter shaped like `FeatureEngineer.build_matrix`.
- **Scale: Task 14's `scale_for`.** The label is a change on that scale:
  - Counts: the log change (a growth rate). This is the task's "log-change" for counts.
  - Percent: the change in log headroom, i.e. the share of the gap to 100 that gets closed.
  - log10: the change in log10 FLOP.
  - Forecasts are transformed back and clipped to [last, 100] like the baselines (this reuses `_bounded`).
- **Look-ahead:** reads go through the harness's view, so the look-ahead check still sees them. A test pins this.
- **No harness change was needed.** The task said to "extend harness to domain='ai'", but `WalkForwardBacktest.run` already takes AI `Target`s (Task 13).

### Training rows are dated on the cutoff's day of the month
- **A bug in the economy model that I found here and fixed for AI:**
  - The economy model dates every training row at a month end. At a month end, the previous month isn't published yet.
  - A live run mid-month already knows that month. Its labels would then cover one month more than its forecasts do.
  - On arXiv that's about +3% bias on every live forecast.
  - `_months_behind` can't fix it, because it reads 1 in both cases.
- **The fix:** rows are dated on the fit cutoff's day of the month (`row_offset`).
  - At the harness's month-end cutoffs, that's exactly the economy's rule.
  - Mid-month (Task 21's live job), rows know the previous month, just as the forecast does.
- **The economy model has the same issue.** It matters less there (CPI YoY and unemployment drift slowly), and I didn't touch economy code. Worth a look when the economy forecast job runs mid-month.

### Trending features are left out by default (`trending=False`)
- **What's dropped:** the levels, logs and lags of the counts and of compute, the `log_ratio_*` features and `trend_months`.
- **Why:** these grow with time, so trees use them as a calendar ("after 2020 growth was slow") and carry that past the training period.
- **What I saw:** with them in, the classifier gave p ≈ 1e-11 to the fast years in the test and got them all wrong (Brier 0.44, worse than the base rate's 0.25).
- **What's kept:** growth rates, changes, headroom, months since a change, the cross-series correlations, calendar months, and the percent levels (bounded).
- **Result:** dropping them also improved the regressor, from RMSE 1,535 to 1,181.
- **`trending=True` turns them back on.**

### `pool=True` trains across a kind
- **How it works:**
  - One row per series and month, for every AI target of the target's kind.
  - `series_<prefix>` indicator columns tell the rows apart.
  - CV folds split by date, so all series' rows for a month land in the same fold (`TimeSeriesSplit` over distinct dates, with gap = horizon).
- **Why:** the benchmarks are too short alone. SWE-bench has 26 months, and `MIN_ROWS` needs 36 labelled. The three together have about 60.
- **Unpooled, nothing changes:** one row per date, the same folds as the economy.
- **This is the task's "cross-series signals" in a second sense.** The first: every series' features are already in every row, e.g. HF growth when predicting arXiv.

### Binary: `AIThresholdClassifier` for the threshold questions only
- **One LightGBM classifier per kind** (`objective='binary'`), with the same `predict_probability(question, data)` interface as Task 13's baselines.
- **Training rows are synthetic, because almost no threshold questions have resolved:**
  - Each row is a question that could have been asked: series × past base month × horizon (1/3/6) × the generator's three steps.
  - The label: was the value h + 1 months later at or above the step?
  - Features are as of the base month's first release, plus `step` (restated relative to the base: log ratio, gap share or log10 difference) and `horizon_months`. These are the task's "time to resolution" and "recent trends".
- **Tuned on log loss.** Folds drop training rows whose outcome was published after the validation period starts.
- **Calibrated:** Platt scaling (`ProbabilityCalibrator('sigmoid')`) on the winning candidate's out-of-fold predictions, when there are at least 30 of them and both outcomes appear. Probabilities are then clipped to [0.01, 0.99].
- **Edge cases:**
  - A kind with fewer than 36 rows is skipped (listed in `skipped`), and its questions raise `NoForecast`.
  - A kind whose outcomes all agree gets the smoothed rate as a constant.
- **It's fit once at a cutoff.** A question asked before that cutoff raises `LookaheadError`, because the model has seen later data.
- **Curated (Polymarket) questions raise `NoForecast`.** `MarketPrice` is the baseline to beat there, and Tasks 18–19 (the LLM) are aimed at them. A LightGBM over price history and category is possible later, but it would be a model of the market, not of AI progress.

---

## What I did
- **`src/eco_prediction/ai/lightgbm_model.py` (new):**
  - `AILightGBMModel` and `AIThresholdClassifier`.
  - Helpers: `pool_for`, `relative_step`, `row_offset` and `is_trending`.
- **`tests/test_ai_lightgbm.py`: 14 tests, not tagged `economy`.**
  - **The synthetic store has growth regimes:** HF grows 1% or 6% a month, switching at random each year, and arXiv copies HF's growth six months later.
  - **Log scale:** pure 2%/month growth is forecast as exactly last × 1.02⁴ with zero spread, and reads go through the view.
  - **Row dating:** month-end cutoffs give the economy's dates. A cutoff on the 14th gives rows on the 14th that know the previous month.
  - **Trending features:** each is classified right, and none reach the model.
  - **Tuning:** the best parameters come from the grid, and the CV RMSE is finite. The top feature by gain is an HF one (the cross-series signal).
  - **Walk-forward backtest:** 42 forecasts, quarterly refits. LightGBM's RMSE is under half the random walk's, and its MAE is lower.
  - **SHAP:** the offset is the log of the last value, exp(prediction) equals the forecast, and there's one contribution per feature.
  - **Pooling:**
    - SWE-bench alone raises. Pooled, it fits, with indicators, and the forecast stays within [last, 100].
    - Pooled rows are date-ordered and one per series and month.
    - Labels are ≤ 0 on log headroom.
  - **Errors:** unfitted, the wrong series, the wrong horizon.
  - **Classifier:**
    - The relative steps.
    - Training rows use only outcomes published by the cutoff.
    - Outcomes that all agree give 1/(n + 2).
    - It refuses unfitted use, questions asked before its cutoff, curated questions and kinds without data.
    - On 41 questions asked 2021-01 to 2024-06 (two slow years, then two fast), its Brier is under half the base rate's.

## Checks
- `uv run pytest`: **587 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean. The new files are formatted.
  - `ruff format --check` flags 15 other files. That was already the case before this task, and I left them alone.
- **Every numeric model on the regime series:** a walk-forward backtest from 2022-01 to 2025-06, refit quarterly, 3 months ahead, 42 forecasts.

  | Model | RMSE | MAE | 80% interval coverage |
  |---|---|---|---|
  | AIRandomWalk | 15,087 | 11,748 | 74% |
  | AIHistoricalMean | 28,958 | 21,962 | 0% |
  | AILinearTrend | 2,762 | 1,740 | 40% |
  | AIARIMA | 1,812 | 1,186 | 88% |
  | AIETS | 2,027 | 1,484 | 71% |
  | **AILightGBMModel** | **1,181** | **737** | 79% |

  - **Threshold questions** (+10% in 2 months, 41 asked, 44% yes): the classifier's Brier is **0.054**, against **0.254** for `ThresholdBaseRate`.
  - **This series was built to have a cross-series lead** that LightGBM can use and ARIMA can't. So it shows the model works as intended, not that it will win on real data. Task 20 makes the real comparison once the backfills have run.

## Where I changed the task's plan
- **A subclass in `ai/`, not edits to `models/lightgbm_model.py`.**
- **No harness change:** it already supports AI targets.
- **The binary classifier covers threshold questions only.** Curated questions are left to the market price and the LLM.
- **I added to the plan:**
  - Rows dated on the cutoff's day of the month (the live-run bias).
  - Trending features dropped by default.
  - Pooling.
  - Probability clipping.

## Tradeoffs
- **Synthetic threshold rows overlap heavily.** Three steps × three horizons share each base month, and consecutive months share most of their span. So the effective sample is much smaller than the row count, and CV log loss is optimistic. Calibration on out-of-fold predictions only partly corrects this.
- **Training rows come from reconstructed history,** the same hindsight caveat as Tasks 12–13.
- **The live question's base value can differ slightly.** It's the value known at the question's cutoff (it may have been revised), while training uses first releases as known at the fit cutoff.
- **The grid is 27 candidates × 5 folds per fit, per kind for the classifier.** That's fine at these sizes (a few seconds), but a monthly refit across 7 series × 3 horizons in Task 21 is 21 regressor fits. A smaller grid may be wanted there.
- **Without trending features, the model can't learn level effects** (e.g. "growth slows as the count gets huge"). On real data that may cost a little. `trending=True` lets Task 20 check.
- **Pooled intervals come from residuals pooled across series,** so they're too wide for the steady series and too narrow for the jumpy ones.
- **SHAP contributions are on the model's scale** (log change for counts, log headroom for percentages). Turning them into "+N papers" needs the back-transform, which isn't additive. Task 22's post-mortems should describe them as growth contributions.

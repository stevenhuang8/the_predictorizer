# Eco Prediction

A forecasting system for the US economy that makes **specific, scoreable predictions**, keeps an honest record of how they turned out, and explains why it was wrong when it misses.

It forecasts three things:

| Target | What it predicts | Horizons |
|---|---|---|
| **CPI inflation** | The year-over-year % change in consumer prices, plus an 80% range | 1, 3 and 6 months ahead |
| **Unemployment rate** | The rate, plus an 80% range | 1, 3 and 6 months ahead |
| **Fed rate decisions (FOMC)** | The probability of a **cut**, **hold** or **hike** at each scheduled meeting | 7 days before each meeting, and at 1, 3 and 6 months ahead |

### The ground rules

1. **Every forecast can be scored.** Each one is a clear question with a resolution date and a resolution rule, so it can be checked later.
2. **No hindsight.** Models only see data as it was published on the day the forecast was made. Economic data gets revised, and the system stores every published version.
3. **Forecasts are never edited.** They're saved before the outcome is known, along with the model version and the exact inputs it saw.
4. **Models must beat simple baselines.** A model that can't beat "next month will look like this month" doesn't earn its place.
5. **Judge calibration, not single misses.** A "70%" call should come true about 70% of the time. One miss on its own doesn't mean the system is broken.

---

## How it works

```
   FRED / ALFRED API
   (every published version of 16 series)
          │
          ▼
 ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
 │ 1. Ingest       │ ──▶ │ 2. Point-in-time     │ ──▶ │ 3. Features         │
 │ data/ingest.py  │     │ view of the data     │     │ 41 numbers per date │
 │ → PostgreSQL    │     │ backtest/data.py     │     │ features/           │
 └─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                 │
          ┌──────────────────────────────────────────────────────┘
          ▼
 ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
 │ 4. Models       │ ──▶ │ 5. Store forecasts   │ ──▶ │ 6. Resolve + score  │
 │ baselines,      │     │ with model version,  │     │ against the first   │
 │ ARIMA, LightGBM │     │ inputs and SHAP      │     │ published value     │
 └─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                 │
                         ┌──────────────────────┐                ▼
                         │ 8. Dashboard + CLI   │ ◀── ┌─────────────────────┐
                         │ web/, eco-forecast   │     │ 7. Post-mortem:     │
                         └──────────────────────┘     │ why did it miss?    │
                                                      └─────────────────────┘
```

One daily job (`scheduler/forecast_job.py`) runs the whole loop:
1. Refreshes the data.
2. Resolves and scores any questions whose answers are now out.
3. Classifies the misses.
4. Makes the new month's forecasts.

---

## The data

All data comes from **FRED** (St. Louis Fed). Its archive, **ALFRED**, has every past version of each value. Each value is stored with an `as_of` date (when it was published), so the system can always reconstruct what was known on any given day.

| Series | What it is | Why it's useful |
|---|---|---|
| `CPIAUCSL` | Consumer Price Index | **Forecast target** (as year-over-year inflation) |
| `UNRATE` | Unemployment rate | **Forecast target** |
| `T10Y2Y`, `T10Y3M` | Long-term minus short-term Treasury yields (the "yield curve") | When it inverts, markets expect a slowdown |
| `DCOILWTICO`, `GASREGW` | Oil and gasoline prices | Energy shows up in CPI quickly |
| `PPIACO` | Producer prices | Business costs reach consumer prices later |
| `CES0500000003` | Average hourly earnings | Wage growth puts pressure on prices |
| `ICSA` | Weekly initial jobless claims | The fastest labor-market signal |
| `MICH` | Inflation that consumers expect | Expectations can become self-fulfilling |
| `CUSR0000SEHA`, `CUUR0000SEHA` | Rent (CPI component) | Shelter is the largest part of CPI |
| `DFEDTAR`, `DFEDTARU` | Fed funds target rate | Used to label each meeting cut / hold / hike |
| `DGS2`, `DTB3` | 2-year and 3-month Treasury yields | Show what markets expect the Fed to do |

These are turned into **41 features** per date (`features/engineering.py`):
- recent levels and % changes over 1, 3 and 12 months
- the targets' own recent values
- 4-week averages of jobless claims
- calendar terms

---

## The models

### Inflation and unemployment

| Model | In plain terms | Role |
|---|---|---|
| **Random walk** | "Next value = latest value" | The baseline to beat |
| **Historical mean** | "It'll be the long-run average" | A naive baseline |
| **ARIMA** | A classic time-series model that only looks at the target's own history | Statistical model |
| **ETS** | Exponential smoothing: recent values count more | Statistical model |
| **LightGBM** | Gradient-boosted decision trees that use all 41 features to predict the *change* from today | Machine-learning model |

### Fed decisions

| Model | In plain terms |
|---|---|
| **Always hold** | Most meetings are holds |
| **Climatology** | How often each decision has happened historically |
| **Persistence** | "The Fed tends to repeat its last move" (a Markov chain) |
| **FOMC LightGBM** | A 3-class tree model. Its best inputs are short-term Treasury yields minus the current Fed rate. When those yields sit well below the Fed's rate, markets are expecting cuts. |

**Running live:** random walk, ARIMA and LightGBM for the numeric targets, and persistence and the LightGBM classifier for the Fed.

### How the models are trained and tested

- **Walk-forward backtesting** (`backtest/harness.py`). This replays history as if the system were running at the time:
  1. Pick a date.
  2. Train only on data published by then.
  3. Forecast.
  4. Step forward a month and repeat, refitting the models periodically.
- **Every training row uses the data as published on that row's own date.** The model learns from the same unrevised, early data it will see when it forecasts for real.
- **LightGBM tuning:**
  - It tries a grid of settings: tree depth, learning rate and minimum leaf size.
  - Each setting is scored on time-ordered folds (always train on the past, test on the future), with a gap between them so nothing leaks across.
  - Training stops early once the test score stops improving.
- **Ranges (prediction intervals)** come from each model's own past errors. They can be corrected afterwards with conformal calibration (`models/calibration.py`).
- **Explanations:** every LightGBM forecast stores its **SHAP values**, which show how much each input pushed the forecast up or down.
- **Scoring:**
  - Numeric forecasts are scored against the *first* published value, because that's what was actually known when the question resolved. Metrics: RMSE, MAE, bias and interval coverage.
  - Fed forecasts use the 3-class **Brier score**. It runs from 0 (perfect) to 2 (confidently wrong).

---

## How well it works

The full report is in `results/model_comparison.html`. To regenerate it, run `uv run eco-forecast backtest`.

### Inflation and unemployment (backtest, 2020–2023, 48 monthly forecasts)

RMSE is the typical size of the error in percentage points; lower is better.

| Model | CPI 1m | CPI 3m | CPI 6m | Unemp. 1m | Unemp. 3m | Unemp. 6m |
|---|---|---|---|---|---|---|
| **ARIMA** | **0.83** | **1.24** | **1.87** | 3.35 | 4.40 | 4.74 |
| ETS | 1.16 | 1.68 | 2.37 | 3.05 | 3.86 | 3.78 |
| Random walk | 1.27 | 1.84 | 2.60 | 2.67 | 3.17 | 2.52 |
| LightGBM | 1.04 | 1.98 | 2.82 | 2.84 | 3.28 | 4.18 |
| Historical mean | 2.86 | 2.85 | 2.74 | **2.62** | **2.62** | **1.90** |

- ✅ **ARIMA is the best inflation model at every horizon.** Its errors are 28–35% smaller than the random walk's, and it tracked the 2021–22 surge and the decline that followed.
- ❌ **LightGBM missed the project's main success goal**, which was to beat the random walk on 3-month inflation. It only wins at 1 month.
  - Its top input was a time-trend feature. That helps in calm years and hurts in turbulent ones.
- ⚠️ **The COVID spike dominates the unemployment results.** No model saw April 2020 coming.
  - The flat historical average "wins" on RMSE only because it never moves.
  - By typical error (MAE), the random walk is best.
- ⚠️ **The ranges were too narrow in this period.** The 80% ranges contained the actual value less often than promised, and LightGBM's did so only 35–42% of the time.

### Fed decisions (backtest, 2005–2026, 173 meetings, 7 days before each)

| Model | Brier score (lower is better) | On the 50 meetings where rates changed |
|---|---|---|
| Always hold | 0.578 | 2.000 |
| Climatology | 0.466 | 1.131 |
| Persistence | 0.331 | 0.743 |
| **LightGBM** | **0.239** | **0.508** |

✅ **This is where machine learning clearly pays off.** LightGBM's score is 59% lower than always-hold's and 28% lower than persistence's, because bond yields reveal what markets expect.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Data and models** | Python 3.13, pandas, statsforecast (ARIMA/ETS), LightGBM, SHAP, scikit-learn (CV, calibration), SciPy, matplotlib |
| **Database** | PostgreSQL 16 in Docker (OrbStack), psycopg2, hand-written SQL migrations (`migrations/`) |
| **Data source** | FRED / ALFRED REST API |
| **Frontend** | Next.js 16 (App Router), TypeScript, Tailwind CSS 4, ESLint + Prettier |
| **Tooling** | uv (packages), pytest (250+ tests), ruff, mypy |
| **Scheduling** | cron, plus a Postgres advisory lock so runs never overlap |

---

## Project layout

```
eco_prediction/
├── src/eco_prediction/
│   ├── data/          # FRED client, ingestion, FOMC meeting calendar + decision labels
│   ├── db/            # connection pool, migration runner, read/write helpers per table
│   ├── backtest/      # point-in-time data views, walk-forward harness, scoring metrics
│   ├── features/      # the 41-feature pipeline
│   ├── models/        # baselines, ARIMA/ETS, LightGBM, FOMC model, SHAP, calibration
│   ├── scheduler/     # daily forecast job + resolution/scoring job
│   ├── postmortem/    # classifies each miss: bad data / bad model / regime change / bad luck
│   ├── scripts/       # history backfill, model comparison report
│   └── cli.py         # `eco-forecast` command
├── migrations/        # numbered SQL migrations (001–007)
├── tests/             # pytest suite
├── results/           # model comparison report (HTML + JSON)
├── web/               # Next.js dashboard
└── NOTES/             # per-task notes: what was built and why
```

### Database tables

| Table | Holds |
|---|---|
| `sources`, `series` | Where each data series comes from |
| `observations` | Every published version of every value: `(series, period, as_of, value)` |
| `questions` | What's being forecast: target, date, horizon |
| `model_versions` | Model type, tuned parameters, training period, git hash |
| `feature_snapshots` | The exact inputs a forecast saw |
| `forecasts` | Predictions, ranges and probabilities, SHAP values, plus scores once resolved |
| `resolutions` | The actual outcome, and which data version decided it |
| `postmortems` | Why a forecast missed, with the evidence |

---

## Getting started

**Prerequisites:** Docker (or OrbStack), [uv](https://docs.astral.sh/uv/), Node.js, a free [FRED API key](https://fredaccount.stlouisfed.org/apikeys), and on macOS `brew install libomp` (LightGBM needs it).

```sh
# 1. Configure and start the database
cp .env.example .env.local          # fill in FRED_API_KEY and a Postgres password
set -a; . ./.env.local; set +a
docker compose up -d                # Postgres on port 5433

# 2. Install, migrate, load data
uv sync
uv run python -m eco_prediction.db.migrate
uv run python -m eco_prediction.scripts.backfill_data    # history from 1990 (slow the first time)

# 3. Use it
uv run eco-forecast forecast --target cpi --horizon 3    # forecast now, every model
uv run eco-forecast forecast --target fomc               # next Fed meeting
uv run eco-forecast inspect                              # list stored questions
uv run eco-forecast scores                               # live track record
uv run eco-forecast backtest                             # regenerate the comparison report

# 4. Run the daily pipeline (put the --once-per-month form in cron)
uv run python -m eco_prediction.scheduler.forecast_job --once-per-month

# 5. Dashboard
cd web && cp .env.local.example .env.local && npm install && npm run dev   # http://localhost:3000
```

Checks: `uv run pytest`, `uv run ruff check src`, `uv run mypy src`.

---

## Tradeoffs and design decisions

| Decision | Why | Cost |
|---|---|---|
| **Store every data vintage** instead of just the latest values | Backtests can't secretly use revised data, which makes them honest | Much more data, and every query has to pick the right version |
| **Score against the first release**, not revised data | That's what was known when the forecast resolved | A forecast can be "wrong" only because the first number was later revised. The post-mortem tags these as `bad_data`, and score summaries also report `rmse_latest` / `mae_latest` / `bias_latest` against the revised value as a diagnostic. |
| **Self-hosted Postgres with raw SQL migrations** instead of a managed service or an ORM | Full control and a learning goal: connections, indexes and backups handled directly | More code to maintain, and no hosted backups |
| **Models build their own point-in-time training rows** instead of a plain `fit(X, y)` | Lookahead is impossible by design, and backtests and live runs share one code path | A custom interface that's slower than training once on one big table |
| **LightGBM predicts the *change* from today**, not the level | Trees can't extrapolate beyond levels they've seen, but changes stay in a familiar range | Still hasn't beaten the random walk at 3 months |
| **ARIMA updates on new data between refits without re-estimating** | Much cheaper than refitting every month, and still starts from the latest data | The model's parameters only change at refits |
| **Ranges from the models' own past errors** (empirical / conformal) | No assumption that errors follow a normal distribution | Ranges built in calm years are too narrow in turbulent ones (2020–23) |
| **No CME FedWatch benchmark** | No free API, and scraping is against their terms | Treasury yields stand in for market expectations, so there's no direct "beat the market" comparison |
| **The CLI never writes forecasts** | The scheduled job stays the only source of the live track record | Forecasts run by hand aren't saved |
| **Forecasts are never edited**; each new version is a separate row | The track record can't be quietly rewritten | More rows, and the latest forecast has to be looked up |
| **Interpolating gaps** (e.g., October 2025 CPI, never published because of the government shutdown) | ARIMA and ETS need an unbroken series | The filled-in month is invented. It's only used for interior gaps, never the latest month. |

## Known limitations and next steps

- **LightGBM doesn't yet beat the random walk on inflation at 3 months.** The first thing to try is removing the time-trend feature: in a quick test without it, LightGBM beat the random walk on 2020–23 inflation.
- **The ranges are too narrow during turbulent periods.** Conformal recalibration exists (`models/calibration.py`) but isn't in the live job yet.
- **ALFRED's archive starts late for many series:** 2009–2011 for jobless claims, oil, wages and rent. Earlier backtests have fewer features.
- **There's no live track record yet.** Live forecasts began in October 2026, and none have resolved.

Detailed notes for each piece of work are in `NOTES/TASK<N>.md`.

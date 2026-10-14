# Eco Prediction

A forecasting system that makes **specific, scoreable predictions**, keeps an honest record of how they turned out, and explains why it was wrong when it misses. It covers two domains:

- **US economy** (live): CPI inflation, unemployment and Fed rate decisions.
- **Direction of AI** (in progress): research output, open-source activity, benchmark progress and compute, with an **LLM forecaster (Claude)** competing against the statistical and ML models under the same rules.

### Ground rules (both domains)

1. **Every forecast can be scored**: a clear question, a resolution date and a resolution rule.
2. **No hindsight**: models only see data as it was published on the forecast date. Every published version of a value is stored with its `as_of` date.
3. **Forecasts are never edited**: they're saved before the outcome is known, with the model version and exact inputs.
4. **Models must beat simple baselines** like "next month will look like this month".
5. **Judge calibration, not single misses**: a "70%" call should come true about 70% of the time.

---

## How it works

```
 data sources ──▶ PostgreSQL ──▶ point-in-time ──▶ features ──▶ models ──▶ stored forecasts
 (every vintage,   observations    view of data                              (inputs, SHAP)
  as_of dated)                                                                     │
                                                                                   ▼
        dashboard + CLI ◀── post-mortem: why did it miss? ◀── resolve + score against
                                                              the first published value
```

A daily job (`scheduler/forecast_job.py`) refreshes data, resolves and scores due questions, classifies misses (bad data / bad model / regime change / bad luck), and makes the new month's forecasts.

---

## Economy domain

| Target | Prediction | Horizons |
|---|---|---|
| **CPI inflation** (YoY %) | Value + 80% range | 1, 3, 6 months |
| **Unemployment rate** | Value + 80% range | 1, 3, 6 months |
| **Fed decisions (FOMC)** | P(cut / hold / hike) per meeting | 7 days before, and 1, 3, 6 months |

**Data:** 16 series from FRED and its vintage archive ALFRED: the targets, yield curve, oil and gas prices, producer prices, wages, jobless claims, inflation expectations, rent, the Fed funds target and short-term Treasury yields. These become 41 features per date (`features/engineering.py`).

**Models:** random walk and historical mean (baselines), ARIMA and ETS (statistical), and LightGBM predicting the *change* from today. For the Fed: always-hold, climatology, persistence (Markov chain) and a 3-class LightGBM classifier. Live: random walk, ARIMA, LightGBM, persistence and the FOMC classifier.

**Testing:** walk-forward backtests (`backtest/harness.py`) train only on data published by each date. LightGBM is tuned on time-ordered folds with a gap. Ranges come from each model's past errors (conformal calibration in `models/calibration.py`), and every LightGBM forecast stores SHAP values. Numeric forecasts are scored with RMSE, MAE, bias and interval coverage; Fed forecasts with the 3-class Brier score (0 perfect, 2 confidently wrong).

### Results

Full report: `results/model_comparison.html` (regenerate with `uv run eco-forecast backtest`).

**Inflation and unemployment** (2020–2023, 48 monthly forecasts, RMSE in pp, lower is better):

| Model | CPI 1m | CPI 3m | CPI 6m | Unemp. 1m | Unemp. 3m | Unemp. 6m |
|---|---|---|---|---|---|---|
| **ARIMA** | **0.83** | **1.24** | **1.87** | 3.35 | 4.40 | 4.74 |
| ETS | 1.16 | 1.68 | 2.37 | 3.05 | 3.86 | 3.78 |
| Random walk | 1.27 | 1.84 | 2.60 | 2.67 | 3.17 | 2.52 |
| LightGBM | 1.04 | 1.98 | 2.82 | 2.84 | 3.28 | 4.18 |
| Historical mean | 2.86 | 2.85 | 2.74 | **2.62** | **2.62** | **1.90** |

- ✅ ARIMA is the best inflation model at every horizon, 28–35% better than the random walk.
- ❌ LightGBM missed the main goal of beating the random walk on 3-month inflation; its top feature was a time trend.
- ⚠️ The COVID spike dominates unemployment; the flat mean "wins" only because it never moves (by MAE the random walk is best).
- ⚠️ 80% ranges were too narrow in this turbulent period (LightGBM's covered only 35–42%).

**Fed decisions** (2005–2026, 173 meetings, 7 days before each, Brier score):

| Model | All meetings | 50 meetings where rates changed |
|---|---|---|
| Always hold | 0.578 | 2.000 |
| Climatology | 0.466 | 1.131 |
| Persistence | 0.331 | 0.743 |
| **LightGBM** | **0.239** | **0.508** |

✅ ML clearly pays off here: short-term yields minus the Fed rate reveal what markets expect.

---

## AI domain (in progress)

The second domain is purely additive: the economy tables, jobs, CLI and dashboard keep their behavior, and a regression safety net (`tests/test_regression_safety.py`) fails if any economy test goes missing. Full spec: `.taskmaster/docs/prd_ai.md`.

**Targets**

| Target | Source | Status |
|---|---|---|
| arXiv AI submissions (cs.AI + cs.LG + cs.CL, monthly) | arXiv API, from 2010 | ✅ Ingestion built |
| Hugging Face new public models (monthly) | Hugging Face Hub API | ✅ Ingestion built |
| GitHub AI activity (new repos monthly, stars snapshot) | GitHub Search API, from 2017 | ✅ Ingestion built |
| Benchmark SOTA (GPQA Diamond, SWE-bench Verified, ARC-AGI-2; best so far, monthly) | Epoch AI | ✅ Ingestion built |
| Largest training run (log10 FLOP, best so far, monthly) | Epoch AI | ✅ Ingestion built |
| Binary events ("Will SOTA on X exceed Y% by D?") | Generated thresholds + curated markets | Planned |
| Market benchmark: AI-tagged Polymarket markets (≥ $10k traded) with daily prices | Polymarket Gamma + CLOB APIs | ✅ Ingestion built |

**What's different from the economy:**
- **Reconstructed vintages are labeled.** AI sources don't keep revision history, so live pulls are snapshotted with `as_of` = fetch date, and backfills built from timestamped records (e.g., arXiv submission dates) are flagged `reconstructed = true`.
- **Binary questions** resolve yes/no and are scored with the Brier score, benchmarked against Polymarket prices (stored point-in-time in `platform_questions` / `platform_probabilities`). Metaculus is on hold: its API terms forbid evaluating AI models without written permission.
- **The LLM forecaster** gets a point-in-time context bundle (question, recent values, stats-model forecasts, retrieved documents published before the forecast date) and returns structured JSON with a rationale. Its prompt, model ID and raw response are stored for every forecast. Backtests only use questions resolving after the model's training cutoff, so it can't recall the answer.

**Roadmap:** foundation (safety net, migrations 009–010, `ai/` package) ✅ → data (arXiv, HF, GitHub, Epoch, Polymarket ✅; Metaculus pending permission) → questions, features, baselines, ARIMA/ETS and LightGBM → LLM forecaster, retrieval corpus and ensemble → live job, `--domain ai` CLI and dashboard section. Per-task notes are in `AI_NOTES/`.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Data and models** | Python 3.13, pandas, statsforecast (ARIMA/ETS), LightGBM, SHAP, scikit-learn, SciPy, matplotlib; Claude via the `anthropic` SDK (planned) |
| **Database** | PostgreSQL 16 in Docker, psycopg2, hand-written SQL migrations |
| **Data sources** | FRED / ALFRED; arXiv, Hugging Face Hub, GitHub, Epoch AI, Polymarket |
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS 4 |
| **Tooling** | uv, pytest (400+ tests), ruff, mypy; launchd + Postgres advisory lock for the daily job |

## Project layout

```
eco_prediction/
├── src/eco_prediction/
│   ├── data/          # FRED client, ingestion, FOMC calendar + decision labels
│   ├── ai/            # AI domain: sources/ (arXiv, HF, GitHub, Epoch, Polymarket clients), ingestion
│   ├── db/            # connection pool, migration runner, per-table helpers
│   ├── backtest/      # point-in-time views, walk-forward harness, metrics
│   ├── features/      # economy feature pipeline
│   ├── models/        # baselines, ARIMA/ETS, LightGBM, FOMC model, SHAP, calibration
│   ├── scheduler/     # daily forecast job + resolution/scoring job
│   ├── postmortem/    # classifies each miss
│   └── cli.py         # `eco-forecast` command
├── migrations/        # numbered SQL migrations (001–011)
├── tests/  results/  web/
├── NOTES/             # economy task notes
└── AI_NOTES/          # AI domain task notes
```

Key tables: `observations` (every vintage: `series, period, as_of, value, reconstructed`), `questions` (now with `domain` and binary questions), `forecasts`, `model_versions`, `feature_snapshots`, `resolutions`, `postmortems`.

---

## Getting started

**Prerequisites:** Docker (or OrbStack), [uv](https://docs.astral.sh/uv/), Node.js, a free [FRED API key](https://fredaccount.stlouisfed.org/apikeys), and on macOS `brew install libomp`. The AI ingestion runs without any keys; adding `GITHUB_TOKEN` or `HF_TOKEN` to `.env` only raises rate limits.

```sh
# 1. Configure and start the database
cp .env.example .env.local          # fill in FRED_API_KEY and a Postgres password
set -a; . ./.env.local; set +a
docker compose up -d                # Postgres on port 5433

# 2. Install, migrate, load data
uv sync
uv run python -m eco_prediction.db.migrate
uv run python -m eco_prediction.scripts.backfill_data    # economy history from 1990

# 3. Economy forecasts
uv run eco-forecast forecast --target cpi --horizon 3    # every model, now
uv run eco-forecast forecast --target fomc               # next Fed meeting
uv run eco-forecast scores                               # live track record

# 4. AI data (--backfill for history, --dry-run to print only; no flag = last 3 months)
uv run python -m eco_prediction.ai.ingest_arxiv --backfill
uv run python -m eco_prediction.ai.ingest_huggingface --backfill
uv run python -m eco_prediction.ai.ingest_github --backfill
uv run python -m eco_prediction.ai.ingest_github stars   # monthly stars snapshot
uv run python -m eco_prediction.ai.ingest_epoch --backfill   # benchmark SOTA + compute
uv run python -m eco_prediction.ai.ingest_polymarket         # markets + daily prices (run daily)

# 5. Daily pipeline (by hand, or at 09:00 via launchd) and dashboard
uv run python -m eco_prediction.scheduler.forecast_job --once-per-month
cp deploy/launchd/com.eco-prediction.forecast-job.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.eco-prediction.forecast-job.plist
cd web && cp .env.local.example .env.local && npm install && npm run dev   # localhost:3000
```

Checks: `uv run pytest`, `uv run ruff check src`, `uv run mypy src`.

---

## Key design decisions

| Decision | Why | Cost |
|---|---|---|
| Store every data vintage | Backtests can't secretly use revised data | More data; every query picks a version |
| Score against the first release | That's what was known at resolution | Revisions can make a forecast look wrong (tagged `bad_data`; revised-value metrics reported as diagnostics) |
| Models build their own point-in-time training rows | Lookahead is impossible; backtest and live share one code path | Slower than one big `fit(X, y)` |
| LightGBM predicts the change, not the level | Trees can't extrapolate past seen levels | Still hasn't beaten the random walk at 3 months |
| Raw SQL migrations on self-hosted Postgres | Full control, learning goal | More code, no hosted backups |
| Only the scheduled job writes forecasts | One source of truth for the track record | Manual CLI forecasts aren't saved |
| AI domain is additive, guarded by a safety net | The live economy pipeline can't regress | Some shared code needs guarded, economy-default branches |
| LLM backtests only after its training cutoff | The model may have memorized earlier outcomes | A smaller LLM evaluation set |

## Known limitations

- LightGBM doesn't yet beat the random walk on 3-month inflation; dropping the time-trend feature looked promising in a quick test.
- Ranges are too narrow in turbulent periods; conformal recalibration isn't in the live job yet.
- ALFRED's archive starts 2009–2011 for several series, so early backtests have fewer features.
- Live economy forecasts began in October 2026 and none have resolved yet. The AI domain has no forecasts yet.

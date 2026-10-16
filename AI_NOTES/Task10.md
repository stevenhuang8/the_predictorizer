# AI Task 10: threshold and curated binary questions

Done on 2026-10-15. Nothing has been written to the live `eco_forecast` database yet. Migration 013, a Polymarket refresh and the first run are the user's to do (see Quick reference).

## Quick reference

```sh
uv run python -m eco_prediction.db.migrate                     # applies 013_question_platform_link
uv run python -m eco_prediction.ai.ingest_polymarket           # refresh markets (stores closed_at; first run ~7 min)
uv run python -m eco_prediction.ai.questions --dry-run         # all three kinds, writes nothing
uv run python -m eco_prediction.ai.questions                   # numeric + thresholds + curated (run daily)
uv run python -m eco_prediction.ai.questions thresholds        # or: numeric, curated
uv run pytest tests/test_ai_binary_questions.py
```

**Order matters:**
- **Threshold questions need last month's value for each series.** Run the AI ingests (Tasks 4–7) first. A series without one is skipped with a message, and a later run picks it up.
- **Curated questions come from `platform_questions`.** Those rows need the `closed_at` this task adds to their metadata, so run `ingest_polymarket` once before the first load.

---

## Decisions made before starting (2026-10-15)

### The user's choices
- **Curation is by keyword rules on the market title** (`ai/curated_questions.py` `classify`). Only AI-progress markets become questions.
  - **Kept:**
    - `benchmark`: benchmark and Arena scores, HLE, IMO, AGI.
    - `ranking`: "best AI model at the end of M", "#3 AI model", "top 3 AI models".
    - `release`: "GPT-6 released by D".
  - **Dropped:**
    - `exact_day`: "released on Feb 12", one day of a many-way split.
    - `finance`: valuations, IPOs, price indexes, and AI trading contests (Alpha Arena, NOF1, okbet).
    - `people_legal`, `outage_app` (outages and app-store ranks) and `hardware` (devices, robots, cars, chips, data centers).
    - Anything that matches no rule (`other`).
  - **Drop rules are checked first,** so "Anthropic's valuation" is finance even though it names a lab.
  - New markets are picked up automatically.
  - **Rejected:** loading every market of at least $10k (a third aren't about AI progress), and a hand-picked id list (new markets would need manual additions).
- **Curated leads: 7, 30 and 90 days before the market's close,** one question per lead. They fit market lifetimes (median 45 days). The task's 180 days would cover only about 380 markets.
- **Thresholds = the latest value plus fixed steps,** each question the binary twin of a Task 9 numeric question.
  - **Rejected:** thresholds around a trend projection. They're more informative near 50/50, but put a model inside the question, and they don't suit step series.

### My defaults (stated, not asked)

#### Threshold questions
- **Calendar:** each threshold question uses `horizon_months` (1/3/6) and the numeric question's month and resolution, not `lead_days`.
  - Task 2 planned `lead_days` for binary questions, but those were event questions with arbitrary dates.
  - These are "is the numeric answer ≥ T?", so they share the numeric calendar, and Task 11 can resolve both from one value.
- **The base is the series' value for month M−1, the month before the ask month,** as known on the ask date (point-in-time: `as_of <= ask date`).
- **Steps:**
  - Counts: +10%, +25% and +50%, to 3 significant figures.
  - SOTA percentages: closing 25%, 50% and 75% of the gap to 100%, to 0.1.
  - Compute (log10 FLOP): 2×, 3× and 10× the record, to 0.01.
  - Thresholds that round to the base or repeat are dropped. So a saturated SOTA gets 0 or 1 question.
  - The kind comes from the series' units: `Percent` → gap, `log10(FLOP)` → multiples, anything else → counts.
- **The key is `<series>>=<threshold>`,** e.g. `EPOCH_SWEBENCH_SOTA>=87.6` or `ARXIV_AI_SUBMISSIONS>=33000`.
- **One set per series, month and horizon.** Once any exists, it's reused as is.
  - Without this, a base revised mid-month (late arXiv papers, an Epoch re-evaluation) would add a second set with different thresholds.
- **The text includes the rule,** as the task asks: "Will … be at least 87.6% for 2027-01? It was 83.5% for 2026-09 (threshold: 25% of the gap to 100%)." followed by the full resolution rule.

#### Curated questions
- **`target_date` is the close's UTC date.**
- **A lead is asked only once its ask date (close − lead) has come,** and only if the market was open on that day: opened by then, and not yet resolved or closed.
  - "Released by D" markets often resolve yes early: 420 resolved more than a week before their scheduled close. A question asked after that would leak the answer.
  - A daily run adds each lead on the day it's due, like the FOMC week-before question.
- **Skipped:**
  - Markets settled without a yes/no (50/50).
  - Rows stored before `closed_at` was recorded, because whether they're still open is unknown. One `ingest_polymarket` run fixes that.
- **Link:** `question_key` is `polymarket:<market id>`, unique per lead. A new `platform_question_id` foreign key (migration 013) points at the market, so benchmark prices join without parsing keys. The task offered "mapping table or extend questions table"; one nullable column is the simpler of the two.
- **Text and rule:** `question_text` is the market title, and `resolution_rule` is the market's criteria plus "Resolves as Polymarket market N does (url)". Both are as first seen; later edits on Polymarket don't change the question.

#### CLI
- **`python -m eco_prediction.ai.questions [numeric|thresholds|curated|all]`,** with `all` as the default, so one daily command covers all three.
  - `--dry-run` runs everything inside a transaction and rolls it back. A numeric-only dry run still needs no database.
- **What a run prints:**
  - Numeric and threshold questions: one line each.
  - Curated: the new questions only, plus the market count per category. A dry run lists every curated question that's due.

---

## What I did

### `migrations/013_question_platform_link.sql`
- `questions.platform_question_id` references `platform_questions(id)`.
- **`questions_platform_link_binary`:** only binary questions can have a link.
- A partial index on the link.
- Purely additive.

### `ai/sources/polymarket_client.py` (one field)
- **`metadata["closed_at"]`** records when trading stopped. Before this, `closed_at` was only kept as `resolved_at` for clean yes/no resolutions, so a market closed with a proposed or disputed resolution had no close time.

### `db/forecasts.py` `get_or_create_question`
- **New `platform_question_id` argument.** Economy calls are unchanged.

### `ai/questions.py`
- **Threshold functions:**
  - `thresholds(series, base)`.
  - `ThresholdQuestion`, with its key, rule and text.
  - `latest_known(conn, series_id, month, as_of)`.
  - **`generate_threshold_binary_questions(conn, as_of_date)`.** It returns the questions asked, the number created, and the series with no base yet.
- **The CLI** now has the kind argument.

### `ai/curated_questions.py` (new)
- `classify(title)`.
- `MarketRow` (a `platform_questions` row).
- `plan_curated_questions(markets, today)`, which is pure.
- **`load_curated_questions(conn, today)`.** It returns every question that's due, the new ones, and the market count per category.

### Tests
- **`tests/test_ai_binary_questions.py`: 41 tests, not tagged `economy`.**
  - **Thresholds:**
    - Counts to 3 significant figures (29,800 → 32,800 / 37,200 / 44,700).
    - SOTA gap steps (83.47 → 87.6 / 91.7 / 95.9).
    - A saturated SOTA (99.96 → one question, 100 → none).
    - Compute (27.97 → 28.27 / 28.45 / 28.97).
    - Every target series gets three thresholds.
  - **Threshold questions (throwaway database):**
    - Rows twin the numeric questions: `ai_binary_event`, binary, `horizon_months`, no `lead_days`.
    - The exact text and rule.
    - The base is point-in-time: a later revision isn't seen.
    - A month keeps its first set after a revision.
    - A missing base is reported, then picked up once stored.
  - **Classification:** 20 real market titles, covering every category.
  - **Curated planning:**
    - Leads are asked once due.
    - A lead before the market opened is skipped.
    - Early resolution or close skips the later leads.
    - 50/50, stale and no-close markets are skipped.
    - Dropped categories are never asked.
  - **Curated loading (throwaway database):**
    - Key, foreign key, text, rule and lead.
    - A daily load adds each lead once.
    - A numeric question can't link a market (the constraint).
    - The economy resolver ignores binary questions.
  - **CLI:**
    - The dry run writes nothing.
    - A real run creates all three kinds and reports a series with no base.
- **`tests/test_ai_questions.py`:** the Task 9 CLI tests now pass `numeric` and expect the `numeric questions=` summary.
- **`tests/test_polymarket_client.py`:** `closed_at` is in the metadata. It's `None` while open, and kept for a closed market with only a proposed resolution.

## Checks
- `uv run pytest`: **505 passed**, including the economy safety net. No economy test changed.
- `ruff check`, `ruff format` and `mypy src` are clean.
- **The classifier on today's 2,537 markets:**

  | Category | Markets | Kept? |
  |---|---|---|
  | ranking | 996 | ✅ |
  | release | 378 | ✅ |
  | benchmark | 237 | ✅ |
  | finance | 338 | dropped |
  | exact_day | 201 | dropped |
  | other | 178 | dropped |
  | hardware | 85 | dropped |
  | outage_app | 70 | dropped |
  | people_legal | 55 | dropped |

- **The planner on today's markets** (from the API, nothing stored):
  - **2,440 questions** from 1,338 markets: 7 days 1,116, 30 days 1,020 and 90 days 304.
  - **2,276 are on markets that have already resolved,** which the backtest can use, and 164 are on open ones.
  - By close year: 29 in 2024, 551 in 2025, 1,793 in 2026 and 67 in 2027.

## Where I changed the task's plan
- **Threshold questions use `horizon_months` (1/3/6), not `horizon_days` 30/90/180.** They share the numeric calendar, as explained above.
- **Curated leads are 7/30/90, not 30/90/180** (the user's choice).
- **Curation is a separate module, `ai/curated_questions.py`,** rather than more code in `questions.py`.
- **One column (`platform_question_id`) instead of a mapping table,** plus the `polymarket:` key Task 2 planned.
- **No Metaculus questions** (deferred, see Task 8).

## Tradeoffs and things to watch
- **Rankings dominate:** 1,906 of 2,440 curated questions (78%).
  - Each "best AI model at the end of M" event is split into one market per company, roughly ten, and the legs are strongly correlated (exactly one is yes).
  - Brier averages over curated questions will mostly measure rankings.
  - **Options for later:** weight by event (`metadata.event_id`), report by category, or keep only the front-runner legs. Task 20's comparison report is the natural place.
- **Many threshold questions will be near-certain.** For example, +50% arXiv in one month, or a saturated benchmark. They mostly test calibration on easy questions. Base-rate baselines (Task 13) will score well on them.
- **Keyword rules misfile some titles.** "Will Sam Altman say 'Robot' on The Tonight Show" lands in `hardware`, and only drop rules caught it. A future market with unusual wording might be kept or dropped wrongly.
  - The dry run prints every kept question, and `classify` is easy to extend with a test.
- **"Released by D" markets that resolve early** leave only their earliest leads. That's correct (no leakage), but such markets contribute fewer questions.
- **The question text is frozen at first sight.** If Polymarket later clarifies the criteria, the stored rule keeps the old wording. That's point-in-time correct for a forecast, but resolution follows the market.
- **Daily runs are needed for curated leads.** The 7-day lead of a market closing next week is only created on or after its ask date. A run that's days late still creates it, with the original ask date.

## Left for later tasks
- **Task 11, resolving:**
  - Threshold questions resolve from the same first-live value as their numeric twin.
  - Curated questions resolve from `platform_questions.resolved_outcome`, when not NULL.
  - Both must stay out of the economy resolver. It filters on `domain = 'economy'` (Task 9).
- **Tasks 13, 18 and 19, forecasting:** the forecast for a curated question is made as of `target_date − lead_days`. The Polymarket price on that day (`platform_probabilities`) is the benchmark.
- **Task 21, the daily job:** run `ai.questions` daily (curated leads come due daily), after `ingest_polymarket`.

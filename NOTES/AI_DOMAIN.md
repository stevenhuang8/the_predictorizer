# The AI domain: design, results and costs

Written 2026-10-16, when the domain went live. This is the one-document view; the reasoning behind each piece, and its tests, are in `AI_NOTES/Task1.md` to `Task25.md`. The spec is `.taskmaster/docs/prd_ai.md`.

---

## What it does

- **Forecasts how fast AI is moving,** under the economy's rules:
  - Every question is scoreable.
  - Models see only data published before the forecast.
  - Forecasts are stored before the outcome and never edited.
  - Simple baselines are scored alongside.
- **Four kinds of question:**
  - **Monthly counts:** arXiv AI papers, new Hugging Face models and new GitHub AI repos, as a value and an 80% range 1, 3 and 6 months ahead.
  - **Records:** GPQA Diamond, SWE-bench Verified and ARC-AGI-2 best scores, and the largest training run (Epoch AI), the same way.
  - **Thresholds:** "will a series clear a bar?", three bars per series and horizon.
  - **Polymarket questions:** AI releases, benchmark records and model rankings, asked 90, 30 and 7 days before each market closes.
- **The models:** baselines, ARIMA/ETS, LightGBM, the market price, base rates, a threshold classifier, and an LLM forecaster (Claude), plus an ensemble.

## The decisions that shape it

### Data
- **Everything is a vintage.** Live pulls are stored with `as_of` = the fetch date.
- **History is reconstructed from timestamped records and flagged `reconstructed`.** That includes arXiv submission dates, Hugging Face `createdAt`, GitHub repo creation dates, and Epoch evaluation dates.
  - Reconstructed history carries hindsight: GitHub counts repos by their topics *today*, Hugging Face only counts models not since deleted, and late benchmark evaluations are dated as if known at the time. (Tasks 4–7)
- **Questions resolve to the first *live* value,** never a backfilled one. So no forecast is scored against hindsight. (Tasks 9, 11)
- **Polymarket only.** Metaculus's API terms forbid evaluating AI models without written permission. Polymarket prices are stored daily, point-in-time, as the benchmark to beat. (Task 8)

### Models
- **Counts are modelled on a log scale,** because growth is multiplicative.
- **Records are modelled on the log of the headroom to 100%.** Closing a share of the gap is a constant step, and forecasts can't pass 100. Records never fall, so forecasts are floored at the last value. (Tasks 13–14)
- **LightGBM** predicts the change on that scale from 109 point-in-time features.
  - It's pooled across the three benchmarks, which are too short to train alone.
  - It leaves out features that trend with time. With them in, it learned "the date" and extrapolated it badly. (Task 15)
- **The threshold classifier** is trained on every past month as if a question had been asked then, since few threshold questions have resolved. (Task 15)

### The LLM forecaster
- **What it's given:** a point-in-time context bundle.
  - The question and its resolution rule.
  - The series' last 12 months.
  - The baselines' forecasts, including the market price for Polymarket questions.
  - About 11 documents from a dated corpus: arXiv abstracts on the forecast topics, model cards from 20 labs, release notes of 10 core libraries, and Polymarket questions. Each document is published before the forecast. (Tasks 16–18)
- **How it answers:** structured JSON, rationale first, from `claude-opus-5-5` with adaptive thinking at high effort. The answer is validated (an interval must contain its point, a probability must be in [0, 1]). A refusal goes to Anthropic's fallback model server-side. (Task 18)
- **Leakage:**
  - Every current Claude model's training data runs to June 2026. So the LLM backtest only uses forecasts *made* from 2026-07-01, not merely questions *resolving* after it: a question asked in May and resolving in August would still leak May–June news.
  - The bundle's own documents are filtered by publication date.
  - Model cards are dated 30 days after their repo was created, because labs create repos privately before a launch. (Tasks 17, 19)
- **The ensemble** blends the LLM with the market and the base rate, or with the statistical models. Its weights are learned walk-forward from forecasts resolved before each new one, and it starts with equal weights until 10 have resolved. (Task 19)

### Prompt engineering
- **The system prompt is short.** It states the date, says to reason only from what's given, to start from base rates and baselines and adjust for specific evidence, and to be calibrated ("an 80% interval should contain the outcome four times in five"). It asks for a rationale under 200 words.
- **No prompt caching:** at about 3,000–4,000 tokens, a request is below the minimum cacheable length.
- **Documents are retrieved by full-text search, with a quota per source.** Dozens of near-identical Polymarket markets otherwise filled the context. Titles are weighted above bodies, because long texts repeating the query words outranked the right title. (Tasks 16–18)
- **The model is shown the market price,** which makes "LLM vs market" partly circular. It measures what the LLM adds to the price. A no-price variant is the obvious next experiment. (Tasks 18, 20)
- **Versioning:** each prompt version is recorded in `model_versions`, with a hash of the prompt and schemas. Changing the prompt means bumping `PROMPT_VERSION`.

### Operations
- **Separate locks:** the AI job (`ai/jobs/ai_forecast_job.py`, launchd at 09:30) runs under its own advisory lock, so it never blocks the economy job or waits on it.
- **Point-in-time binary questions:** each is forecast as of its own ask date. A late catch-up is stored as a backtest.
- **A model that can't fit yet is "unavailable", not a failure.** For example, ARIMA needs 24 months, and ARC-AGI-2 has 19. (Task 21)
- **Post-mortems use the economy's four categories.** Counts are judged on the log scale. Yes/no forecasts use the FOMC rule, with the market or base rate as the reference. An optional Haiku-drafted explanation waits for a person to confirm it. (Task 22)
- **The economy stays untouched:** a regression safety net fails if any economy test goes missing.
  - It caught one change of mine, in the CLI's target parsing.
  - The dashboard's economy queries were missing a domain filter, and showed AI data once the AI job ran. That was found and fixed in Task 24.

## Results so far

**Live:** forecasts began on 2026-10-16:
- 541 forecasts on the first run.
- 2,276 Polymarket questions resolved from the backlog. They had no forecasts, so they're unscored.
- The first numeric questions (1 month ahead) resolve in early December 2026. There's no live track record yet.

**Synthetic** (these show the models behave as designed, not that they'll win on real data):

| Test | Result |
|---|---|
| arXiv-like series with a 6-month lead from HF growth, 42 walk-forward forecasts (Task 15) | RMSE: LightGBM 1,181 · ARIMA 1,812 · ETS 2,027 · linear trend 2,762 · random walk 15,087 |
| Threshold questions on growth regimes, 41 asked (Task 15) | Brier: classifier 0.054 · base rate 0.254 |
| Pure 3%/month growth, 30 forecasts (Task 14) | RMSE: ETS 206 · ARIMA 238 · linear trend 503 · random walk 1,630 |

**The LLM:** not yet run on real questions (the budget is deferred).
- The post-cutoff backtest is built and dry-run: 574 Polymarket questions plus 14 numeric forecasts.
- Its verdict, whether it beats the base rate after the training cutoff, waits for that run. If it doesn't, the PRD says to report it and keep the LLM out of live use.

## Cost analysis

| | Per forecast | The 588-forecast backtest |
|---|---|---|
| `claude-opus-5-5` ($4 / $20 per M tokens) | ~$0.076 | ~$44 |
| `claude-haiku-5-5` ($0.10 / $0.50) | ~$0.002 | ~$1.10 |

- **The basis:** prompts measured on the live data (mean about 3,900 tokens), plus an *assumed* 3,000 output and thinking tokens. The first real calls will replace that assumption with logged numbers.
- **Live use, if switched on:** each day's newly due Polymarket questions plus 21 numeric and about 63 threshold questions a month.
  - That's on the order of 100–300 Opus forecasts a month, about $8–25, or under $1 on Haiku.
  - Post-mortem drafts on Haiku cost about $0.0004 each.
- **Controls:**
  - Nothing spends without an explicit budget (`--budget`, `--llm-budget`), and every run stops before the next call once new spending reaches it.
  - Identical calls are reused from `llm_calls`, so re-runs are free.
  - Every call's tokens and cost are logged, failed calls included.

## Future improvements

1. **Run the LLM backtest** (Haiku first, about $1), then decide on Opus and on live use from its verdict.
2. **A prompt with no market price,** to measure the LLM's independent skill, and a market-only ensemble for comparison.
3. **Metaculus,** if permission comes through the Data Needs Form. The tables already allow it.
4. **Semantic retrieval** (embeddings): lexical search misses paraphrases ("frontier model launch" doesn't find "GPT-5 released").
5. **Several samples per question** (self-consistency), if the LLM's calibration is poor. That multiplies the cost.
6. **Recalibrating intervals:** conformal calibration from the economy, once enough AI forecasts have resolved.
7. **New benchmarks** as the current ones saturate. ARC-AGI-2 was already at 95% in September 2026. That means adding Epoch series and topic queries.

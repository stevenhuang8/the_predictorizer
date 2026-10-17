# AI Task 20: the LLM comparison report

Done on 2026-10-16. **The report has no real data yet.** No LLM backtest has been run, because the API budget is deferred until all AI tasks are done (the user's call). `results/llm_comparison.html` currently says so and gives the command. Everything was checked on synthetic backtest results.

## Quick reference

```sh
uv run python -m eco_prediction.ai.llm.backtest --budget N            # first: a real run, saved to results/llm_backtest/
uv run python -m eco_prediction.ai.llm.backtest --model claude-haiku-5-5 --budget N   # optional second model, for the cost chart
uv run python -m eco_prediction.ai.llm.comparison_report             # -> results/llm_comparison.html + .json
uv run pytest tests/test_llm_comparison.py
```

---

## Decisions (my defaults, not asked)

### The report reads saved runs; it never calls the API
- **The backtest CLI now pickles each real run** to `results/llm_backtest/<model>.pkl` (`save_result` / `load_results`; the latest run per model wins).
- **The report builds from those files,** so rebuilding it costs nothing and takes seconds.
- **The pickles are git-ignored** (`results/llm_backtest/`): they hold prompts and rationales. The HTML and JSON are tracked, like `model_comparison.*`.
- **The first `.gitignore` entry I wrote didn't work.** A comment after a pattern becomes part of the pattern. I caught that with `git check-ignore`; the comment is now on its own line.
- **One run is the primary:** the Opus run if there is one, else the first. The cost section compares every run (Opus vs Haiku).

### Sections (the task's six, plus a verdict)
1. **Binary questions:**
   - The Brier score of the LLM, the Polymarket price, the base rate and the ensemble, on the questions all four answered.
   - Skill against the market and against the base rate.
   - A 95% interval for each difference.
   - Breakdowns by category and by lead (90/30/7 days).
2. **Numeric targets:** relative RMSE (forecast / actual − 1, so series pool) by horizon, and the ratio to the random walk.
3. **Cost:** calls (new and reused), spend, cost per call, failures, and the LLM's Brier per model. A cost-against-Brier chart appears once two models have run.
4. **Calibration:** a reliability diagram of the LLM's and the market's probabilities (`metrics.calibration_curve`), with the tables.
5. **Sample rationales:** the LLM's 5 most confident hits and 5 most confident misses, with the market price beside each.
6. **Key phrases:** the commonest word pairs and words in the rationales. The task's "word cloud or key phrase extraction"; I chose phrases, which need no new dependency.
7. **Caveats:**
   - The training cutoff and how many questions it dropped.
   - The market price in the prompt.
   - Clustering.
   - Reconstructed data.
   - Metaculus not compared.
- **The verdict at the top is the PRD's criterion:** the LLM's Brier is below the base rate's after the cutoff. It's "significant" when the bootstrap interval of the difference lies below 0.

### Bootstrap by market, not by question
- **The 574 questions aren't independent.** Each market is asked at three leads, and an event's legs ("which company has the best model") resolve together.
- **So the intervals resample markets.** A test shows the difference: two markets of 50 identical questions give an interval of [−1, 1] by market, but under 0.5 wide if every question were treated as independent.
- **2,000 resamples, seed 0,** so the report is reproducible.

### Charts
- **Static PNGs in a self-contained HTML page, like `model_comparison.html`,** reusing its CSS and chart helpers (`_style`, `_png`, `_legend`, `_json_safe`). The two reports look like one system.
- **Colours:**
  - The statistical models keep their colours from the economy report (LightGBM blue, random walk orange, ARIMA aqua).
  - The new components take the reference palette's remaining slots: LLM violet, ensemble green, market yellow, base rate magenta.
  - Both sets that appear together pass the dataviz validator: lightness, chroma, colour-blind separation and the normal-vision floor.
  - Yellow, magenta and aqua are under 3:1 contrast on the background, so every chart has a legend and a table of the same numbers.
- **Rendering and looking at the charts** (the dataviz procedure's last step) caught three problems the tests couldn't:
  - The cost chart's log axis labelled its minor ticks over each other. It now shows only $0.001 / $0.01 / $0.1.
  - The ensemble's bar in the numeric chart was invisible at an error of 0. The bars now carry value labels.
  - The phrase chart mixed words and pairs. It now charts only pairs, and the table has both.

### Key phrases: pairs and words ranked apart
- **A word is always at least as common as any pair containing it,** so a single ranked list never showed a pair. A test caught this.
- **Now** half the slots go to pairs that occur more than once, and the rest to words.

---

## What I did
- **`src/eco_prediction/ai/llm/comparison_report.py` (new):**
  - The tables: `binary_table`, `binary_by`, `numeric_table`, `cost_table`, `calibration`, `sample_rationales` and `key_phrases`.
  - `cluster_bootstrap` and `verdict`.
  - Five charts, `render_html`, `build_report`, and the CLI.
- **`src/eco_prediction/ai/llm/backtest.py`:**
  - `save_result` / `load_results`, and the CLI saves each real run (`--out`).
  - Binary rows now carry the title, market id, lead and the LLM's rationale.
- **`.gitignore`:** `results/llm_backtest/`.
- **`results/llm_comparison.html` and `.json`:** generated from the (empty) real results, so for now they say how to get some.
- **`tests/test_llm_comparison.py`: 9 tests, not tagged `economy`.** They use a synthetic run: 40 markets × 3 leads, an LLM that leans towards the outcome, a noisier market, a flat 0.3 base rate, and 6 numeric forecasts.
  - **The binary table:**
    - Every component on 120 common questions from 40 markets.
    - The expected order.
    - Positive skill, with an interval below 0.
    - No self-comparison.
  - **The bootstrap** resamples markets (the two-market case above). It's undefined for one market.
  - **Breakdowns:** by category and by lead; an unknown column gives an empty frame.
  - **The numeric table:** exact relative RMSEs (2% and 5%), the ratio to the random walk (0.4), and the ensemble row.
  - **Rationales:** hits come most confident first. **Phrases:** the pair "release cadence" and the word "release" both appear, and stopwords don't.
  - **The verdict:** passed and significant for a skilled LLM; not passed for an LLM that always says about 0.5.
  - **The full report with two models:**
    - Every section heading is present.
    - The HTML is balanced (a parser checks every tag closes).
    - There are 5 charts, each a real PNG.
    - Opus is the primary, and the dropped count and verdict are in the text.
    - The JSON has the right keys and values, Haiku's cost per call included.
  - **No results:** the report says how to get some, and the JSON is empty.
  - **Save, load and the CLI:** a run round-trips through pickle, and the CLI writes the report and prints the verdict.

## Checks
- `uv run pytest`: **648 passed**, including the economy safety net.
- `ruff check` and `mypy src` are clean.
- **The report was rendered from synthetic Opus and Haiku runs, and each chart inspected** (fixes above).

## Where I changed the task's plan
- **A new module, `ai/llm/comparison_report.py`,** not an extension of `scripts/model_comparison.py`. It reuses that module's styling.
- **The input is saved backtest runs.** The report never runs a backtest.
- **Benchmarks:** Polymarket only (Metaculus deferred).
- **Key phrases, not a word cloud.**
- **Added:** the clustered bootstrap, the verdict, the breakdowns by category and lead, and the empty-results page.

## Tradeoffs
- **The numeric section will have very few forecasts** for months (14 at first). Its text says to treat it as indicative.
- **Pickle ties saved runs to the code's classes.** A future change to `LLMBacktestResult` can make old pickles unreadable. Re-running a backtest is free for calls already logged (Task 18's reuse), so the cost is time, not money.
- **The LLM-vs-market comparison is partly circular** while the prompt shows the market price (a caveat in the report). A no-price run would need a CLI flag; not added.
- **Key phrases are plain frequency,** and the same phrase repeated in one rationale counts each time. The axis says "occurrences" for that reason.

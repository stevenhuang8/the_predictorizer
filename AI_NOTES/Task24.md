# AI Task 24: the dashboard's AI section

Done on 2026-10-16. Built, checked against the live database, and inspected in screenshots. No migration. The dashboard only reads.

## Quick reference

```sh
cd web && npm run dev            # then /ai, /ai/questions, /ai/forecasts/<id>, /ai/track-record
cd web && npm run build          # passes: no TypeScript or prerender errors
cd web && npm run lint
```

| Page | What it shows |
|---|---|
| `/ai` | Tiles: open questions by kind, live forecasts, scored, LLM spend. One chart and table per series: three years of history, and every model's forecast 1, 3 and 6 months ahead. |
| `/ai/questions` | Open questions, filterable by kind (`?kind=numeric\|threshold\|curated`). Each model's latest live forecast, with Polymarket links. |
| `/ai/forecasts/[id]` | One question: its resolution, a chart (the series, or the market's daily price), and every forecast, with the LLM's rationale expandable, LightGBM's drivers, the score and the post-mortem or LLM draft. |
| `/ai/track-record` | Numeric accuracy per series, horizon and model; binary Brier by kind; the LLM vs the market on the same questions; calibration; why forecasts missed; LLM cost. `?backtest=1` includes backtests. |

- **Navigation:** the header gets "AI progress", and the footer names the AI data sources.

---

## Decisions (my defaults, not asked)

### The economy pages were showing AI data, now fixed
- **No economy query filtered by domain.** Since the AI job's first run (Task 21), the economy pages had been mixing in AI data:
  - **`/forecasts`** got **245 open forecasts, 223 of them AI.** Now it gets 22.
  - **`/track-record`** counted **94 open questions.** The economy has 8.
  - **"Data through" and "last run"** took the maximum over every series and forecast, AI included.
  - **Once AI forecasts are scored,** they'd have joined the economy score tables, which only excluded FOMC.
- **The fix:** `loadDashboard(today, domain = "economy")` and `loadTrackRecord(domain = "economy")` filter every query by `q.domain`. The two queries that didn't join `questions` now do, and "data through" counts only the domain's series. Economy pages are back to economy data.

### Structure
- **The loaders are in `web/lib/ai-dashboard-data.ts`,** each limited to `q.domain = 'ai'`:
  - `loadAIOverview`, `loadAIQuestions`, `loadAIQuestion` and `loadAITrackRecord`.
  - `loadLLMCosts` (the task's `getLLMCosts`).
- **The pages follow the existing ones:** async server components that call `await connection()` inside `<Suspense>`.
- **They read `params` and `searchParams` inside the Suspense boundary too.** This project has Cache Components enabled, and the first build failed because my pages awaited them at the top. Next 16's bundled docs and the build error both give the fix.
- **`TimeChart` gets an optional `format` prop** for counts, percentages and log10, so an arXiv count doesn't read "11245.00".
  - It's a format *name*, not a function, because a server page can't pass a function to a client component.
  - Without it, the economy charts render exactly as before.

### Colours
- **The statistical models keep their economy colours** (LightGBM blue, random walk orange, ARIMA aqua).
- **ETS and the market price share yellow,** and **the linear trend and the base rate share magenta.** The models in each pair never appear in the same chart.
- **The LLM gets a new token, violet** (`--series-violet`: #6250d6, dark #9085e9).
- **Validated with the dataviz validator** in light and dark mode: lightness, chroma, colour-blind separation and the normal-vision floor all pass.
  - In light mode yellow, magenta and aqua are under 3:1 against the background. The charts carry direct labels and tooltips, and every chart has a table of the same numbers.
- **The calibration chart has a hover tooltip per bin,** plus a table.

### Found by looking at the rendered pages (screenshots, light and dark)
1. **"Exponential smoothing" was clipped** in the chart's right-hand labels. It's now "ETS", with its blurb explaining.
2. **Five overlapping 80% bands smeared together.** Each chart now shades one (LightGBM's) and says so; the table under it lists every model's range.
3. **Series headings were the full database definitions.** They're now short names, with the definition as a subtitle.
4. **The threshold rows on `/ai/questions` all read the same.** The truncated question text lost the threshold and month. Labels are now built from the key ("arXiv AI papers ≥ 12,400 in Nov 2026").
5. **A question with only backtest forecasts looked empty.** Its forecasts were behind a collapsed toggle. They now open when they're all there is.
6. **A real model bug: the random walk's interval could exclude its own point.**
   - ARC-AGI-2 at 95% showed "No change 95.0% (100.0%–100.0%)". Its past 3-month gains are bigger than the 5 points of headroom left, so the 100 cap pushed both ends of the interval above the point.
   - **Fixed in `ai/baselines._bounded`** (Task 13), which now keeps lower ≤ point ≤ upper. There's a regression test using ARC-AGI-2's shape.
   - **8 forecasts stored on 2026-10-16 have a point outside their interval:**
     - **7 random-walk forecasts** (ARC-AGI-2, GPQA, SWE-bench) come from this bug. Stored forecasts are never edited, so they stay as made and will be scored as they are. New runs are coherent.
     - **1 LightGBM forecast (Hugging Face) is different.** LightGBM's interval comes from its past out-of-fold errors. A model that has consistently under-forecast has even its 10th-percentile error above zero, so the whole interval sits above the point. That's the economy model's design (an empirical bias correction), not this bug; left as is.

### Things the task asked for that differ
- **"Scores vs Metaculus/Polymarket":** Polymarket only, as Metaculus is deferred.
- **The track record compares each model with the market on the same Polymarket questions,** from the same day.
- **Empty states say when data will arrive:** the first numeric forecasts resolve in early December; the LLM hasn't run, so it's cost nothing.

---

## What I did
- **`web/lib/ai-dashboard-data.ts` (new).**
- **`web/lib/dashboard-data.ts`, `web/lib/track-record-data.ts`:** the domain parameter (economy by default).
- **`web/app/ai/` (new):**
  - `models.ts`: names, blurbs, colours, short series names, labels and formats.
  - `page.tsx` and `overview.tsx`.
  - `questions/page.tsx` and `forecasts/[id]/page.tsx`.
  - `track-record/page.tsx` and `track-record/calibration-chart.tsx`.
- **`web/app/_components/time-chart.tsx`:** the `format` prop. **`tones.ts`:** violet.
- **`web/app/globals.css`:** the violet token, light and dark.
- **`web/app/layout.tsx`:** the nav link, and the footer's data sources.
- **`src/eco_prediction/ai/baselines.py`:** `_bounded` keeps the point inside its interval. **`tests/test_ai_baselines.py`:** the regression test.

## Checks
- **`npm run build`, `npm run lint` and `tsc --noEmit` all pass.**
- **`uv run pytest`: 688 passed.** `ruff check` and `mypy src` are clean.
- **Every page served from the production build against the live database** (`next start` on port 3100, since port 3000 had a server already running):
  - **All 200, with no error digests.** That covers `/`, `/forecasts`, `/track-record`, `/ai`, `/ai/questions` (and a kind filter), `/ai/track-record` (and `?backtest=1`), and `/ai/forecasts` for 2585, 10104 and 82.
  - **A bad id** (999999, `abc`) renders the 404 content. The status is 200, because the lookup happens inside a streamed boundary, after the status is sent.
  - **The economy pages mention AI only in the shared footer.**
- **Screenshots inspected:**
  - `/ai` in light and dark.
  - `/ai/questions`, filtered to thresholds.
  - The AGI market's page.
  - The fixes above were re-checked in fresh screenshots.

## Not checked
- **True phone width.** Headless Chrome has a minimum window width: at 390 px the existing economy page clips the same way as the AI ones, so the screenshot can't judge phone layout. The AI pages use the same responsive classes as the economy pages (grids that collapse, tables that scroll inside their container). Check on a phone, or with devtools' device mode.
- **The calibration chart, the vs-market table and the rationale sections with real data.** Nothing is scored yet, and the LLM hasn't run. They show their empty states. The synthetic and real-data paths are typed and built, but have only been seen empty.

## Tradeoffs
- **`/ai/questions` shows the 200 questions resolving soonest** out of about 250 open; the kind filter reaches the rest. No pagination yet.
- **Charts show one model's band.** The others are in the tables, not on the chart.
- **AI pages read the database on every request,** like the economy pages. That's fine at this scale.

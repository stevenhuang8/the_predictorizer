# Task 24: Next.js frontend foundation

Done on 2026-10-08.

## Quick reference

```sh
cd web
cp .env.local.example .env.local      # DATABASE_URL (server-side only)
npm install
npm run dev                           # http://localhost:3000
npm run build                         # production build, includes the type check
npm run lint && npm run format:check && npm run typecheck
npm run format                        # apply Prettier
```

---

## What I did

### `web/`, scaffolded with `create-next-app@16.4.0`
- App Router (`app/`, no `src/`), TypeScript (strict), Tailwind CSS 4, ESLint 9 (`eslint-config-next`), import alias `@/*`.
- **Prettier 3.9.9**, with `.prettierrc.json` (88 columns, matching ruff on the Python side) and `.prettierignore`. `eslint-config-prettier` is last in `eslint.config.mjs`, so ESLint leaves formatting to Prettier.
- **Scripts:** `format`, `format:check` and `typecheck` (`next typegen && tsc --noEmit`) alongside the scaffold's `dev`, `build`, `start` and `lint`.
- **Pages:**
  - **`app/layout.tsx`:** header navigation (Overview, Forecasts, Track record), footer, and a metadata title template (`Forecasts | Eco Prediction`).
  - **`app/page.tsx`:** the landing page. It describes the three targets and the project's rules: point-in-time data, forecasts stored before the outcome is known, scoring against first releases, baselines scored too.
  - **`app/forecasts/page.tsx` and `app/track-record/page.tsx`:** "Coming soon" pages built on a shared `app/_components/coming-soon.tsx`. Each lists what the page will show and **which tables feed it**, so Phase 4 starts from a spec:
    - forecasts ← `questions`, `forecasts`, `model_versions`, `feature_snapshots`
    - track record ← `resolutions`, forecast scores, `postmortems`
- **Styling:** colour tokens on `:root` (surface, ink, muted, line, panel) with dark-mode values, exposed to Tailwind through `@theme`. It uses the same neutrals as Task 19's report.
- **`web/.env.local.example`:** `DATABASE_URL`, deliberately without the `NEXT_PUBLIC_` prefix so it stays on the server.
- **Ignores:**
  - `web/.gitignore` (from the scaffold) ignores `.env*`, which would have hidden the example file, so I added `!.env.local.example`.
  - The root `.gitignore` gets `web/.next/` and `web/node_modules/`, as the task asked. They're redundant with `web/.gitignore`, but harmless.
- Removed the scaffold's demo SVGs from `public/`, and replaced its README with a project-specific one.

### Where I changed the task's plan
- **Next.js 16.4, not 14.** The task's title says 14, but its command is `create-next-app@latest`, and 16.4.0 is current. Next 14 is past its support window. I pinned the scaffolder to 16.4.0 rather than `@latest`, so a rerun gives the same project.
- **The scaffold's `AGENTS.md` stays.** It warns that this Next.js version differs from older documentation, and `next dev` re-creates it anyway. I read the bundled docs (`node_modules/next/dist/docs/`) on layouts, linking and metadata before writing the pages. `LayoutProps<"/">` is the new globally available, generated layout type.
- **The scaffold enables `cacheComponents` and `partialPrefetching`** in `next.config.ts`. I left them on. The pages are static, and they matter once Phase 4 adds database reads.

---

## Checks
- **`npm run build`:** compiled, type-checked, and prerendered `/`, `/forecasts`, `/track-record` and `/_not-found` as static pages.
- **`npm run dev`:** `/` (title "Eco Prediction"), `/forecasts` ("Forecasts | Eco Prediction") and `/track-record` ("Track record | Eco Prediction") return 200; an unknown path returns 404.
- **`npm run lint`, `npm run typecheck`, `prettier --check`:** no problems.
- **Not checked:** I didn't view the pages in a browser, only their HTML and titles. Worth a look with `npm run dev` before Phase 4 builds on the layout.

## Notes
- **`npm audit`** reports 5 "high" findings. They're one advisory (`braces`, a denial of service from deeply nested glob patterns), pulled in by ESLint's dev-only tooling through `eslint-config-next`. Nothing runs in the browser or server, and nothing processes untrusted globs. The suggested `npm audit fix --force` would *downgrade* `eslint-config-next` to 14.x, so I didn't apply it. It should clear when `eslint-config-next` updates.
- **`next dev` warns about a `package-lock.json` in your home folder** (`/Users/stevenhuang/package-lock.json`). It's outside the repository and Next ignores it, but it's probably a leftover worth deleting.
- **The npm "Unknown user config" warnings** (`config`, `python`, `python3`) come from your `~/.npmrc`, not from this project.

---

## Follow-up: the live forecasts dashboard (commit `32e2292`, 2026-10-08)

Not a separate Taskmaster task. It replaces the `/forecasts` placeholder above with a real page that reads the database. `/track-record` is still a placeholder. (These notes were written afterwards, on 2026-10-09, from the code.)

### Data: `web/lib/db.ts` and `web/lib/dashboard-data.ts`
- **`db()`** keeps one `pg` connection pool per server process (at most 4 connections), stored on `globalThis` so hot reloads in development don't open new pools. `DATE` columns come back as `"YYYY-MM-DD"` strings. A JavaScript `Date` would shift them by the server's time zone.
- Both files import `server-only`, so the database code and `DATABASE_URL` can't end up in the browser bundle.
- **`loadDashboard(today)`** runs its queries in parallel and returns:
  - **forecasts:** for every *open* question (no resolution yet), the live forecasts from that question's latest forecast date. Backtest rows are excluded. They're split into numeric forecasts (point and 80% range) and FOMC forecasts (cut/hold/hike probabilities);
  - **history:** the last 36 months of CPI year-over-year and unemployment, using the latest vintage of each month. CPI year-over-year is computed from the index levels, so 12 extra months are loaded;
  - **the policy rate:** the midpoint of the Fed's target range (`DFEDTARU` − 0.125), reduced to the days it changed;
  - **"data through"** (the newest `as_of` in `observations`) and **"last run"** (the newest live forecast date).

### Page: `web/app/forecasts/`
- **`page.tsx`** wraps the dashboard in `<Suspense>` with a loading message.
- **`dashboard.tsx`** is a server component. `await connection()` makes it read the database on every request, not once at build time. If the database can't be reached, it shows a "Can't load forecasts right now" panel with the error instead of crashing.
  - **A status line:** when the latest forecasts were made, and the date the data runs through.
  - **Summary tiles:**
    - inflation and unemployment: the latest value, and the range of the models' forecasts for the middle horizon (3 months), e.g. "expected 3.4% to 3.8% by Jan 2027";
    - the next FOMC meeting: the FOMC model's most likely outcome and its probability.
  - **Inflation and unemployment sections:** a sentence summarizing where the models expect the value to go ("rise to", "fall to" or "stay near" a value, using a 0.1-point threshold), a chart, and a table of each model's forecast and range for each target month.
  - **Fed section:** the current target range and one card per open meeting, with a probability bar per model and the FOMC model's most likely outcome.
  - **"How to read this":** what the ranges and probabilities mean, and a plain-language description of each model.
- **`models.ts`:** a display name, a one-line description and a color for each model id. The random walk is shown as **"No change"** and persistence as **"Repeat last decision"**, since the technical names mean nothing to a general reader.

### Components: `web/app/_components/`
- **`time-chart.tsx`** (client component): an SVG chart of the actual history plus each model's forecasts, drawn from the latest actual value. Each forecast point has its range as a shaded band. Hovering shows every series at the nearest month. Labels sit at the right-hand end of each line instead of in a separate legend, spread apart with leader lines where they would overlap. Axis ticks use rounded steps (1, 2, 2.5, 5).
- **`probability-bar.tsx`:** cut / hold / hike as one stacked bar (blue, neutral, red), with percentages inside each segment when they fit and always in the row below. It has an `aria-label` giving all three numbers.
- **`tones.ts`:** chart colors as complete Tailwind class names. They have to be written out in full, because Tailwind only generates classes it finds as literal strings in the code. `stroke-${name}` would silently produce no color.
- **`format.ts`:** date and percent formatting, always in UTC so dates don't shift by a day.
- **`globals.css`:** new color tokens for the three models and the three decisions, with dark-mode values. Each model keeps the same color as in the Task 19 report.

### Tradeoffs
- **A hand-drawn SVG chart instead of a chart library.** No new dependency, and full control over the forecast bands and labels. The cost is about 340 lines of chart code to maintain.
- **Raw SQL through `pg` instead of an ORM**, the same as the Python side. The queries are typed by hand, so a schema change won't be caught by the TypeScript compiler.
- **Only open questions are shown.** Once a question resolves, its forecasts leave this page. They belong on `/track-record`, which hasn't been built yet.
- **The page reads the database on every request.** That's fine for one user. With more traffic it should be cached until the next daily run.

### Checks (2026-10-09)
- `npm run lint`, `npm run typecheck` and `prettier --check`: no problems.
- **Not checked here:** `npm run build`, and viewing the page in a browser.

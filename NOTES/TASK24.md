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

# Eco Prediction web

Next.js frontend for the forecasts and their track record. `/forecasts` shows the
live forecasts from the database; `/track-record` is still a placeholder.

```sh
cp .env.local.example .env.local   # DATABASE_URL, server-side only
npm install
npm run dev                        # http://localhost:3000
npm run build                      # production build (also type-checks)
npm run lint && npm run format:check && npm run typecheck
```

- `app/page.tsx`: overview
- `app/forecasts/`: the forecasts dashboard. `dashboard.tsx` is a server component
  that reads the database on each request; `models.ts` has each model's display
  name, description and color
- `app/track-record/page.tsx`: coming-soon placeholder that lists what it will show
  and which tables feed it
- `app/_components/`: the SVG time chart, the cut/hold/hike probability bar, chart
  colors (`tones.ts`) and date/percent formatting
- `lib/db.ts`, `lib/dashboard-data.ts`: the Postgres pool and the dashboard's queries
  (server-only, so `DATABASE_URL` never reaches the browser)
- Next.js 16 App Router, TypeScript, Tailwind CSS 4, ESLint + Prettier.
  This Next.js version differs from older docs; see `AGENTS.md`.

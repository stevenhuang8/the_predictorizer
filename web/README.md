# Eco Prediction web

Next.js frontend for the forecasts and their track record (Task 24: foundation and
placeholder pages; the data views come in Phase 4).

```sh
cp .env.local.example .env.local   # DATABASE_URL, server-side only
npm install
npm run dev                        # http://localhost:3000
npm run build                      # production build (also type-checks)
npm run lint && npm run format:check && npm run typecheck
```

- `app/page.tsx`: overview
- `app/forecasts/page.tsx`, `app/track-record/page.tsx`: coming-soon placeholders
  that list what each will show and which tables feed it
- Next.js 16 App Router, TypeScript, Tailwind CSS 4, ESLint + Prettier.
  This Next.js version differs from older docs; see `AGENTS.md`.

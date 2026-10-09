import Link from "next/link";

const TARGETS = [
  {
    name: "CPI inflation",
    detail: "Year-over-year % change, 1, 3 and 6 months ahead, with 80% intervals.",
  },
  {
    name: "Unemployment",
    detail: "The headline rate, 1, 3 and 6 months ahead, with 80% intervals.",
  },
  {
    name: "FOMC decisions",
    detail:
      "Cut / hold / hike probabilities for each scheduled meeting, monthly and " +
      "7 days before the decision.",
  },
];

const PRINCIPLES = [
  "Every forecast uses only data published before it was made (ALFRED vintages).",
  "Forecasts are stored before the outcome is known and never edited afterwards.",
  "Each one is scored against the first release, with the misses classified.",
  "Simple baselines are scored alongside the models, so skill is measured, not assumed.",
];

export default function Home() {
  return (
    <div className="space-y-12">
      <section className="max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Economic forecasts with an honest track record
        </h1>
        <p className="mt-4 text-lg text-muted">
          Monthly forecasts of US inflation, unemployment and Federal Reserve rate
          decisions from statistical and machine-learning models, each scored once the
          data is in.
        </p>
      </section>

      <section>
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          What is forecast
        </h2>
        <ul className="mt-4 grid gap-4 sm:grid-cols-3">
          {TARGETS.map(({ name, detail }) => (
            <li key={name} className="rounded-lg border border-line bg-panel p-4">
              <h3 className="font-semibold">{name}</h3>
              <p className="mt-2 text-sm text-muted">{detail}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="max-w-2xl">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          How it stays honest
        </h2>
        <ul className="mt-4 list-disc space-y-2 pl-5 text-muted">
          {PRINCIPLES.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      </section>

      <section className="flex flex-wrap gap-3">
        <Link
          href="/forecasts"
          className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background"
        >
          Current forecasts
        </Link>
        <Link
          href="/track-record"
          className="rounded-md border border-line px-4 py-2 text-sm font-medium"
        >
          Track record
        </Link>
      </section>
    </div>
  );
}

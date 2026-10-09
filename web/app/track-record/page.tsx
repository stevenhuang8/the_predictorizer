import type { Metadata } from "next";
import { Suspense } from "react";

import { Backtest } from "./backtest";
import { LiveRecord } from "./live-record";

export const metadata: Metadata = { title: "Track record" };

function HowToRead() {
  return (
    <section className="rounded-lg border border-line bg-panel p-5 text-sm">
      <h2 className="font-semibold">How to read this</h2>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-muted">
        <li>
          <strong className="text-foreground">RMSE</strong> is the typical size of a
          miss, in percentage points; lower is better. The best model at each horizon is
          in bold.
        </li>
        <li>
          <strong className="text-foreground">Scores use the first release</strong>, the
          number as first published. That is what was known when each question resolved,
          so a score never changes afterwards. The{" "}
          <strong className="text-foreground">vs revised</strong> column scores the same
          forecasts against the data as revised since, to show how much revisions
          matter.
        </li>
        <li>
          <strong className="text-foreground">80% ranges held</strong>: how often the
          actual landed inside a model&apos;s 80% range. The tick marks 80%; a fill
          short of it means the ranges were too narrow.
        </li>
        <li>
          <strong className="text-foreground">Brier score</strong> rates cut / hold /
          hike probabilities, from 0 (certain and right) to 2 (certain and wrong).
          &ldquo;Right call&rdquo; is how often the most likely outcome happened.
        </li>
      </ul>
    </section>
  );
}

export default function TrackRecordPage() {
  return (
    <div className="space-y-12">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Track record</h1>
        <p className="mt-2 max-w-2xl text-muted">
          How the forecasts turned out once the official numbers came in, and why the
          misses missed.
        </p>
      </div>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Live forecasts</h2>
        <Suspense fallback={<p className="text-muted">Loading the track record…</p>}>
          <LiveRecord />
        </Suspense>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Backtest</h2>
        <Backtest />
      </section>

      <HowToRead />
    </div>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";

import { Overview } from "./overview";

export const metadata: Metadata = { title: "AI progress" };

export default function AIPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">AI progress</h1>
        <p className="mt-2 max-w-2xl text-muted">
          Forecasts of how fast AI is moving: research output, open models, benchmark
          records and training compute, plus yes/no questions mirrored from Polymarket.
          Built the same way as the economic forecasts: only data published before each
          forecast, stored before the outcome, scored after.
        </p>
        <nav className="mt-4 flex flex-wrap gap-3 text-sm">
          <Link
            href="/ai/questions"
            className="rounded-md border border-line px-3 py-1.5"
          >
            Open questions
          </Link>
          <Link
            href="/ai/track-record"
            className="rounded-md border border-line px-3 py-1.5"
          >
            Track record
          </Link>
        </nav>
      </div>
      <Suspense fallback={<p className="text-muted">Loading the AI forecasts…</p>}>
        <Overview />
      </Suspense>
    </div>
  );
}

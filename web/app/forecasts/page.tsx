import type { Metadata } from "next";
import { Suspense } from "react";

import { Dashboard } from "./dashboard";

export const metadata: Metadata = { title: "Forecasts" };

export default function ForecastsPage() {
  return (
    <div className="space-y-2">
      <h1 className="text-3xl font-semibold tracking-tight">Forecasts</h1>
      <p className="max-w-2xl text-muted">
        Where inflation, unemployment and Federal Reserve rates are heading, according
        to each model, with how sure each one is.
      </p>
      <div className="pt-6">
        <Suspense fallback={<p className="text-muted">Loading forecasts…</p>}>
          <Dashboard />
        </Suspense>
      </div>
    </div>
  );
}

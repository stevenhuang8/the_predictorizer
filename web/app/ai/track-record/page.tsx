import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";

import { ValueBar } from "../../track-record/bars";
import { loadAITrackRecord } from "@/lib/ai-dashboard-data";
import { KIND_LABELS, aiModelName, aiModelTone, usd } from "../models";
import { CalibrationChart } from "./calibration-chart";

export const metadata: Metadata = { title: "AI track record" };

const CATEGORY_LABELS: Record<string, string> = {
  expected_variance: "Within its own uncertainty",
  bad_model: "Model missed an available signal",
  regime_change: "A surprise nobody priced",
  bad_data: "The data was revised",
};

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-lg border border-dashed border-line p-4 text-sm text-muted">
      {children}
    </p>
  );
}

async function Record({ includeBacktest }: { includeBacktest: boolean }) {
  await connection();
  const t = await loadAITrackRecord(includeBacktest);
  const maxRel = Math.max(0, ...t.numeric.map((r) => r.relativeRmse));
  const maxBrier = Math.max(
    0,
    ...t.binary.map((r) => r.brier),
    ...t.vsMarket.map((r) => r.marketBrier),
  );
  const horizons = [
    ...new Set(t.numeric.map((r) => `${r.seriesId}|${r.horizonMonths}`)),
  ];
  const bestRel = (key: string) =>
    Math.min(
      ...t.numeric
        .filter((r) => `${r.seriesId}|${r.horizonMonths}` === key)
        .map((r) => r.relativeRmse),
    );

  return (
    <div className="space-y-12">
      <section className="space-y-3">
        <h2 className="text-xl font-semibold tracking-tight">Monthly numbers</h2>
        {t.numeric.length === 0 ? (
          <Empty>
            No numeric forecast has resolved yet. The first ones (1 month ahead) resolve
            in early December, when November&apos;s numbers come in.
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <thead>
                <tr className="border-b border-line text-left text-muted">
                  <th className="py-2 pr-4 font-medium">Series</th>
                  <th className="py-2 pr-4 font-medium">Ahead</th>
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="py-2 pr-4 font-medium">n</th>
                  <th className="py-2 pr-4 font-medium">Typical miss</th>
                  <th className="py-2 pr-4 font-medium">80% range held</th>
                </tr>
              </thead>
              <tbody>
                {horizons.flatMap((key) =>
                  t.numeric
                    .filter((r) => `${r.seriesId}|${r.horizonMonths}` === key)
                    .map((r, i) => (
                      <tr key={`${key}|${r.model}`} className="border-b border-line">
                        <td className="py-2 pr-4">
                          {i === 0 ? r.name.split(" (")[0].split(",")[0] : ""}
                        </td>
                        <td className="py-2 pr-4 tabular-nums">
                          {i === 0 ? `${r.horizonMonths} mo` : ""}
                        </td>
                        <td className="py-2 pr-4">{aiModelName(r.model)}</td>
                        <td className="py-2 pr-4 tabular-nums">{r.n}</td>
                        <td className="py-2 pr-4">
                          <ValueBar
                            value={r.relativeRmse * 100}
                            max={maxRel * 100}
                            tone={aiModelTone(r.model)}
                            best={r.relativeRmse === bestRel(key)}
                            digits={1}
                            label={`${aiModelName(r.model)}, typical miss in %`}
                          />
                        </td>
                        <td className="py-2 pr-4 tabular-nums">
                          {r.coverage == null
                            ? "–"
                            : `${(r.coverage * 100).toFixed(0)}%`}
                        </td>
                      </tr>
                    )),
                )}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted">
              Typical miss: root mean square of the error as a % of the actual value, so
              series of different sizes compare. Lower is better; the best per row group
              is bold.
            </p>
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold tracking-tight">Yes/no questions</h2>
        {t.binary.length === 0 ? (
          <Empty>
            No yes/no forecast has resolved yet
            {includeBacktest ? "" : " among the live ones"}.
            {!includeBacktest && (
              <>
                {" "}
                <Link href="/ai/track-record?backtest=1" className="underline">
                  Include backtests
                </Link>
                .
              </>
            )}
          </Empty>
        ) : (
          <div className="grid gap-8 lg:grid-cols-[1fr_auto]">
            <div className="space-y-6">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-muted">
                    <th className="py-2 pr-4 font-medium">Kind</th>
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 font-medium">n</th>
                    <th className="py-2 pr-4 font-medium">Brier</th>
                  </tr>
                </thead>
                <tbody>
                  {t.binary.map((r) => (
                    <tr key={`${r.kind}|${r.model}`} className="border-b border-line">
                      <td className="py-2 pr-4">{KIND_LABELS[r.kind].title}</td>
                      <td className="py-2 pr-4">{aiModelName(r.model)}</td>
                      <td className="py-2 pr-4 tabular-nums">{r.n}</td>
                      <td className="py-2 pr-4">
                        <ValueBar
                          value={r.brier}
                          max={maxBrier}
                          tone={aiModelTone(r.model)}
                          digits={3}
                          label={`${aiModelName(r.model)} Brier`}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {t.vsMarket.length > 0 && (
                <div>
                  <h3 className="font-semibold">Against the market</h3>
                  <p className="text-sm text-muted">
                    On the Polymarket questions both answered, from the same day.
                  </p>
                  <table className="mt-2 w-full text-sm">
                    <thead>
                      <tr className="border-b border-line text-left text-muted">
                        <th className="py-2 pr-4 font-medium">Model</th>
                        <th className="py-2 pr-4 font-medium">n</th>
                        <th className="py-2 pr-4 font-medium">Its Brier</th>
                        <th className="py-2 pr-4 font-medium">Market&apos;s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {t.vsMarket.map((r) => (
                        <tr key={r.model} className="border-b border-line">
                          <td className="py-2 pr-4">{aiModelName(r.model)}</td>
                          <td className="py-2 pr-4 tabular-nums">{r.n}</td>
                          <td className="py-2 pr-4">
                            <ValueBar
                              value={r.brier}
                              max={maxBrier}
                              tone={aiModelTone(r.model)}
                              best={r.brier < r.marketBrier}
                              digits={3}
                              label="Brier"
                            />
                          </td>
                          <td className="py-2 pr-4">
                            <ValueBar
                              value={r.marketBrier}
                              max={maxBrier}
                              tone="yellow"
                              best={r.marketBrier <= r.brier}
                              digits={3}
                              label="Market Brier"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            {t.calibration.length > 0 && (
              <div>
                <h3 className="mb-2 font-semibold">Calibration</h3>
                <CalibrationChart
                  lines={t.calibration.map((c) => ({
                    model: c.model,
                    label: aiModelName(c.model),
                    tone: aiModelTone(c.model),
                    bins: c.bins,
                  }))}
                />
              </div>
            )}
          </div>
        )}
      </section>

      <section className="grid gap-8 sm:grid-cols-2">
        <div className="space-y-2">
          <h2 className="text-xl font-semibold tracking-tight">Why forecasts missed</h2>
          {Object.keys(t.categories).length === 0 ? (
            <Empty>No post-mortems yet.</Empty>
          ) : (
            <ul className="space-y-1 text-sm">
              {Object.entries(t.categories)
                .sort((a, b) => b[1] - a[1])
                .map(([c, n]) => (
                  <li
                    key={c}
                    className="flex justify-between gap-4 border-b border-line py-1.5"
                  >
                    <span>{CATEGORY_LABELS[c] ?? c}</span>
                    <span className="tabular-nums">{n}</span>
                  </li>
                ))}
            </ul>
          )}
        </div>
        <div className="space-y-2">
          <h2 className="text-xl font-semibold tracking-tight">LLM cost</h2>
          {t.llm.calls === 0 ? (
            <Empty>The LLM hasn&apos;t been run yet, so it has cost nothing.</Empty>
          ) : (
            <ul className="space-y-1 text-sm">
              <li className="flex justify-between border-b border-line py-1.5">
                <span>Total</span>
                <span className="tabular-nums">
                  {usd(t.llm.spentUsd)} over {t.llm.calls} calls
                </span>
              </li>
              <li className="flex justify-between border-b border-line py-1.5">
                <span>Per stored forecast</span>
                <span className="tabular-nums">
                  {t.llm.forecasts ? usd(t.llm.spentUsd / t.llm.forecasts) : "–"}
                </span>
              </li>
              <li className="flex justify-between border-b border-line py-1.5">
                <span>Failed calls</span>
                <span className="tabular-nums">{t.llm.failed}</span>
              </li>
              {t.llm.byModel.map((m) => (
                <li
                  key={m.model}
                  className="flex justify-between border-b border-line py-1.5 text-muted"
                >
                  <span className="font-mono text-xs">{m.model}</span>
                  <span className="tabular-nums">
                    {usd(m.spentUsd)} · {m.calls} calls
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}

async function ScopeAndRecord({
  searchParams,
}: {
  searchParams: Promise<{ backtest?: string }>;
}) {
  const includeBacktest = (await searchParams).backtest === "1";
  return (
    <div className="space-y-8">
      <p className="max-w-2xl text-sm text-muted">
        {includeBacktest ? "Including" : "Live forecasts only, not"} backtests
        (forecasts made after the fact with data cut off at the time).{" "}
        <Link
          href={includeBacktest ? "/ai/track-record" : "/ai/track-record?backtest=1"}
          className="underline-offset-2 hover:underline"
        >
          {includeBacktest ? "Live only" : "Include backtests"}
        </Link>
      </p>
      <Record includeBacktest={includeBacktest} />
    </div>
  );
}

/** `searchParams` is read inside the Suspense boundary (Cache Components). */
export default function AITrackRecordPage({
  searchParams,
}: {
  searchParams: Promise<{ backtest?: string }>;
}) {
  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm text-muted">
          <Link href="/ai" className="hover:text-foreground">
            AI progress
          </Link>{" "}
          / Track record
        </p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">AI track record</h1>
        <p className="mt-2 max-w-2xl text-muted">
          How the AI forecasts turned out, against simple baselines and the prediction
          market.
        </p>
      </div>
      <Suspense fallback={<p className="text-muted">Loading the track record…</p>}>
        <ScopeAndRecord searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

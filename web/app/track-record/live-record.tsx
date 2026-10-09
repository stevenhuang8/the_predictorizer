import { connection } from "next/server";

import { longDate, monthYear, pct } from "../_components/format";
import { HorizonChart } from "../_components/horizon-chart";
import { FOMC_ORDER, NUMERIC_ORDER, modelName, modelTone } from "../forecasts/models";
import type { NumericTarget } from "@/lib/dashboard-data";
import {
  type FomcScore,
  type MissCategory,
  type NumericScore,
  type TrackRecord,
  loadTrackRecord,
} from "@/lib/track-record-data";
import { CoverageMeter, ModelLabel, ValueBar } from "./bars";

const TITLES: Record<NumericTarget, string> = {
  cpi_yoy: "Inflation (CPI, year over year)",
  unemployment: "Unemployment rate",
};

const TARGET_NAMES: Record<string, string> = {
  cpi_yoy: "CPI",
  unemployment: "Unemployment",
  fomc_decision: "FOMC",
};

/** The post-mortem categories, in the order a reader should care about them. */
const CATEGORIES: Record<MissCategory, { label: string; detail: string }> = {
  expected_variance: {
    label: "Within its stated uncertainty",
    detail: "Close enough given the forecast's own range: not a failure.",
  },
  bad_data: {
    label: "First number later revised",
    detail:
      "Scored against a first release that was later revised toward the forecast.",
  },
  regime_change: {
    label: "Unprecedented move",
    detail: "A move bigger than anything in the history the model could see.",
  },
  bad_model: {
    label: "Model error",
    detail:
      "A normal-sized move, or a decision markets had priced, that the model missed.",
  },
};

function rank(model: string, order: string[]): number {
  return order.includes(model) ? order.indexOf(model) : order.length;
}

function ModelCell({ model }: { model: string }) {
  return (
    <td className="py-2 pr-4">
      <ModelLabel name={modelName(model)} tone={modelTone(model)} />
    </td>
  );
}

/** RMSE by horizon per model, once some target has more than one horizon scored. */
function HorizonCharts({
  targets,
  scores,
}: {
  targets: NumericTarget[];
  scores: NumericScore[];
}) {
  const shown = targets.filter(
    (t) =>
      new Set(scores.filter((s) => s.target === t).map((s) => s.horizonMonths)).size >
      1,
  );
  if (!shown.length) return null;
  return (
    <div>
      <h3 className="font-semibold">Misses by forecast horizon</h3>
      <p className="mt-1 max-w-2xl text-sm text-muted">
        Typical miss (RMSE, percentage points) at each horizon, against the first
        release.
      </p>
      <div className="mt-4 grid gap-8 md:grid-cols-2">
        {shown.map((t) => {
          const forTarget = scores.filter((s) => s.target === t);
          const models = [...new Set(forTarget.map((s) => s.model))].sort(
            (a, b) => rank(a, NUMERIC_ORDER) - rank(b, NUMERIC_ORDER),
          );
          return (
            <HorizonChart
              key={t}
              title={TITLES[t]}
              series={models.map((model) => ({
                id: model,
                label: modelName(model),
                tone: modelTone(model),
                points: forTarget
                  .filter((s) => s.model === model)
                  .sort((a, b) => a.horizonMonths - b.horizonMonths)
                  .map((s) => ({ horizon: s.horizonMonths, value: s.rmse })),
              }))}
            />
          );
        })}
      </div>
    </div>
  );
}

function NumericScores({
  target,
  scores,
}: {
  target: NumericTarget;
  scores: NumericScore[];
}) {
  const rows = [...scores].sort(
    (a, b) =>
      a.horizonMonths - b.horizonMonths ||
      rank(a.model, NUMERIC_ORDER) - rank(b.model, NUMERIC_ORDER),
  );
  const max = Math.max(...rows.map((r) => r.rmse));
  const best = new Map<number, number>();
  for (const r of rows) {
    best.set(r.horizonMonths, Math.min(best.get(r.horizonMonths) ?? Infinity, r.rmse));
  }
  return (
    <div>
      <h3 className="font-semibold">{TITLES[target]}</h3>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[44rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Horizon</th>
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Scored</th>
              <th className="py-2 pr-4 font-medium">RMSE</th>
              <th className="py-2 pr-4 font-medium">vs revised</th>
              <th className="py-2 pr-4 font-medium">Bias</th>
              <th className="py-2 font-medium">80% ranges held</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((r) => (
              <tr
                key={`${r.horizonMonths}-${r.model}`}
                className="border-b border-line"
              >
                <td className="py-2 pr-4 whitespace-nowrap text-muted">
                  {r.horizonMonths} mo
                </td>
                <ModelCell model={r.model} />
                <td className="py-2 pr-4">{r.n}</td>
                <td className="py-2 pr-4">
                  <ValueBar
                    value={r.rmse}
                    max={max}
                    tone={modelTone(r.model)}
                    best={r.rmse === best.get(r.horizonMonths)}
                    label={`${modelName(r.model)}, ${r.horizonMonths}-month RMSE`}
                  />
                </td>
                <td className="py-2 pr-4 text-muted">
                  {r.rmseLatest != null ? r.rmseLatest.toFixed(2) : "–"}
                </td>
                <td className="py-2 pr-4">
                  {r.bias >= 0 ? "+" : ""}
                  {r.bias.toFixed(2)}
                </td>
                <td className="py-2">
                  {r.coverage != null ? (
                    <CoverageMeter share={r.coverage} tone={modelTone(r.model)} />
                  ) : (
                    "–"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function horizonLabel(s: FomcScore): string {
  return s.leadDays != null
    ? `${s.leadDays} days before`
    : `${s.horizonMonths} mo ahead`;
}

function FomcScores({ scores }: { scores: FomcScore[] }) {
  const rows = [...scores].sort(
    (a, b) =>
      (a.leadDays ?? Infinity) - (b.leadDays ?? Infinity) ||
      (a.horizonMonths ?? 0) - (b.horizonMonths ?? 0) ||
      rank(a.model, FOMC_ORDER) - rank(b.model, FOMC_ORDER),
  );
  const max = Math.max(...rows.map((r) => r.brier));
  const best = new Map<string, number>();
  for (const r of rows) {
    const h = horizonLabel(r);
    best.set(h, Math.min(best.get(h) ?? Infinity, r.brier));
  }
  return (
    <div>
      <h3 className="font-semibold">Federal Reserve rate decisions</h3>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[34rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Forecast</th>
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Meetings</th>
              <th className="py-2 pr-4 font-medium">Brier score</th>
              <th className="py-2 font-medium">Right call</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((r) => (
              <tr
                key={`${horizonLabel(r)}-${r.model}`}
                className="border-b border-line"
              >
                <td className="py-2 pr-4 whitespace-nowrap text-muted">
                  {horizonLabel(r)}
                </td>
                <ModelCell model={r.model} />
                <td className="py-2 pr-4">{r.n}</td>
                <td className="py-2 pr-4">
                  <ValueBar
                    value={r.brier}
                    max={max}
                    tone={modelTone(r.model)}
                    best={r.brier === best.get(horizonLabel(r))}
                    label={`${modelName(r.model)}, Brier score`}
                  />
                </td>
                <td className="py-2">{Math.round(r.hitRate * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PostMortems({ record }: { record: TrackRecord }) {
  const total = Object.values(record.categories).reduce((a, b) => a + (b ?? 0), 0);
  if (total === 0) return null;
  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-semibold">Why forecasts missed</h3>
        <p className="mt-1 text-sm text-muted">
          Every scored forecast is classified automatically; a person can override the
          call.
        </p>
      </div>
      <ul className="grid gap-3 sm:grid-cols-2">
        {(Object.keys(CATEGORIES) as MissCategory[]).map((c) => (
          <li key={c} className="rounded-lg border border-line bg-panel p-4">
            <p className="text-2xl font-semibold tracking-tight">
              {record.categories[c] ?? 0}
              <span className="ml-2 text-sm font-normal text-muted">
                of {total} ({Math.round(((record.categories[c] ?? 0) / total) * 100)}%)
              </span>
            </p>
            <p className="mt-1 text-sm font-medium">{CATEGORIES[c].label}</p>
            <p className="mt-0.5 text-sm text-muted">{CATEGORIES[c].detail}</p>
          </li>
        ))}
      </ul>
      {record.misses.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold">Latest misses</h4>
          <ul className="mt-2 divide-y divide-line text-sm">
            {record.misses.map((m) => (
              <li key={m.forecastId} className="py-2">
                <p>
                  <span className="font-medium">
                    {TARGET_NAMES[m.target] ?? m.target}{" "}
                    {m.target === "fomc_decision"
                      ? longDate(m.targetDate)
                      : monthYear(m.targetDate)}
                  </span>
                  <span className="text-muted">
                    {" "}
                    · {modelName(m.model)} · {CATEGORIES[m.category].label}
                    {m.manual ? " (reviewed by hand)" : ""}
                  </span>
                </p>
                {m.notes && <p className="mt-0.5 text-muted">{m.notes}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function RecentResolutions({ record }: { record: TrackRecord }) {
  if (record.recent.length === 0) return null;
  return (
    <div>
      <h3 className="font-semibold">Recently scored</h3>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[44rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Month</th>
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Forecast</th>
              <th className="py-2 pr-4 font-medium">First release</th>
              <th className="py-2 pr-4 font-medium">Revised</th>
              <th className="py-2 font-medium">Verdict</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {record.recent.map((r) => (
              <tr key={r.forecastId} className="border-b border-line">
                <td className="py-2 pr-4 whitespace-nowrap">
                  {TARGET_NAMES[r.target]} {monthYear(r.targetDate)}
                  <span className="text-muted"> · {r.horizonMonths} mo</span>
                </td>
                <ModelCell model={r.model} />
                <td className="py-2 pr-4 whitespace-nowrap">
                  {pct(r.prediction, 2)}
                  {r.lower != null && r.upper != null && (
                    <span className="text-muted">
                      {" "}
                      ({r.lower.toFixed(1)}–{r.upper.toFixed(1)})
                    </span>
                  )}
                </td>
                <td className="py-2 pr-4 font-semibold">{pct(r.actual, 2)}</td>
                <td className="py-2 pr-4 text-muted">
                  {r.latest != null && Math.abs(r.latest - r.actual) >= 0.005
                    ? pct(r.latest, 2)
                    : "–"}
                </td>
                <td className="py-2 whitespace-nowrap">
                  {r.inInterval === true
                    ? "In range"
                    : r.inInterval === false
                      ? "Outside range"
                      : "–"}
                  {r.category && r.category !== "expected_variance" && (
                    <span className="text-muted">
                      {" "}
                      · {CATEGORIES[r.category].label}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export async function LiveRecord() {
  await connection(); // read the database per request, not at build time
  let record: TrackRecord;
  try {
    record = await loadTrackRecord();
  } catch (error) {
    return (
      <div className="rounded-lg border border-line bg-panel p-5">
        <p className="font-semibold">Can&apos;t load the track record right now.</p>
        <p className="mt-1 text-sm text-muted">
          The database didn&apos;t answer. Is Docker running, and is DATABASE_URL set in
          web/.env.local?
        </p>
        <p className="mt-2 font-mono text-xs text-muted">{String(error)}</p>
      </div>
    );
  }

  if (record.scored === 0) {
    return (
      <div className="rounded-lg border border-line bg-panel p-5">
        <p className="font-semibold">No live forecast has been scored yet.</p>
        <p className="mt-1 text-sm text-muted">
          {record.open > 0 ? (
            <>
              {record.open} question{record.open === 1 ? " is" : "s are"} waiting on the
              data. The earliest, about{" "}
              {record.firstOpen ? monthYear(record.firstOpen) : "the coming month"},
              resolves once the official number is first published, usually a week or
              two into the following month.
            </>
          ) : (
            "No live forecasts have been made yet."
          )}{" "}
          Until then, the backtest below shows how the models would have done.
        </p>
      </div>
    );
  }

  const targets = (["cpi_yoy", "unemployment"] as const).filter((t) =>
    record.numeric.some((s) => s.target === t),
  );
  return (
    <div className="space-y-10">
      <p className="text-sm text-muted">
        {record.scored} forecast{record.scored === 1 ? "" : "s"} scored · {record.open}{" "}
        question{record.open === 1 ? "" : "s"} still open
      </p>
      <HorizonCharts targets={[...targets]} scores={record.numeric} />
      {targets.map((t) => (
        <NumericScores
          key={t}
          target={t}
          scores={record.numeric.filter((s) => s.target === t)}
        />
      ))}
      {record.fomc.length > 0 && <FomcScores scores={record.fomc} />}
      <PostMortems record={record} />
      <RecentResolutions record={record} />
    </div>
  );
}

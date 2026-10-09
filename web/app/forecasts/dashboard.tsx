import { connection } from "next/server";
import type { ReactNode } from "react";

import { longDate, monthYear, pct } from "../_components/format";
import { ProbabilityBar } from "../_components/probability-bar";
import { type ChartSeries, TimeChart } from "../_components/time-chart";
import { TONES } from "../_components/tones";
import {
  type DashboardData,
  type FomcForecast,
  type NumericForecast,
  type NumericTarget,
  type Point,
  loadDashboard,
} from "@/lib/dashboard-data";
import { FOMC_ORDER, MODELS, NUMERIC_ORDER, modelName } from "./models";

const TARGETS: Record<NumericTarget, { title: string; noun: string }> = {
  cpi_yoy: { title: "Inflation (CPI, year over year)", noun: "inflation" },
  unemployment: { title: "Unemployment rate", noun: "unemployment" },
};

function byOrder<T extends { model: string }>(items: T[], order: string[]): T[] {
  return [...items].sort((a, b) => order.indexOf(a.model) - order.indexOf(b.model));
}

/** "rise to", "fall to" or "stay near", against the latest actual value. */
function direction(from: number, to: number): string {
  if (to - from > 0.1) return "rise to";
  if (from - to > 0.1) return "fall to";
  return "stay near";
}

function range(values: number[]): string {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return Math.abs(hi - lo) < 0.05 ? pct(lo) : `${pct(lo)} to ${pct(hi)}`;
}

function Tile({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight tabular-nums">{value}</p>
      <p className="mt-1 text-sm text-muted">{detail}</p>
    </div>
  );
}

function NumericSection({
  target,
  history,
  forecasts,
}: {
  target: NumericTarget;
  history: Point[];
  forecasts: NumericForecast[];
}) {
  const { title, noun } = TARGETS[target];
  const latest = history[history.length - 1];
  const horizons = [...new Set(forecasts.map((f) => f.targetDate))].sort();
  const farthest = horizons[horizons.length - 1];
  const models = NUMERIC_ORDER.filter((m) => forecasts.some((f) => f.model === m));

  const series: ChartSeries[] = [
    {
      id: "actual",
      label: "Actual",
      tone: "ink",
      kind: "actual",
      points: history.map((p) => ({ date: p.date, value: p.value })),
    },
    ...models.map((m) => ({
      id: m,
      label: modelName(m),
      tone: MODELS[m].tone,
      kind: "forecast" as const,
      points: [
        { date: latest.date, value: latest.value },
        ...byOrder(forecasts, NUMERIC_ORDER)
          .filter((f) => f.model === m)
          .sort((a, b) => a.targetDate.localeCompare(b.targetDate))
          .map((f) => ({
            date: f.targetDate,
            value: f.point,
            lower: f.lower,
            upper: f.upper,
          })),
      ],
    })),
  ];

  const far = forecasts.filter((f) => f.targetDate === farthest).map((f) => f.point);
  const meanFar = far.reduce((a, b) => a + b, 0) / (far.length || 1);

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {latest && farthest && (
          <p className="mt-1 text-muted">
            {noun[0].toUpperCase() + noun.slice(1)} was{" "}
            <strong className="text-foreground">{pct(latest.value, 2)}</strong> in{" "}
            {monthYear(latest.date)}. The models expect it to{" "}
            {direction(latest.value, meanFar)}{" "}
            <strong className="text-foreground">{range(far)}</strong> by{" "}
            {monthYear(farthest)}.
          </p>
        )}
      </div>
      <TimeChart
        series={series}
        splitDate={latest?.date}
        ariaLabel={`${title}: the last three years and each model's forecast`}
      />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[32rem] text-sm tabular-nums">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Model</th>
              {horizons.map((d) => (
                <th key={d} className="py-2 pr-4 font-medium">
                  {monthYear(d)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m} className="border-b border-line">
                <td className="py-2 pr-4">
                  <span className="flex items-center gap-2">
                    <span
                      className={`inline-block h-0.5 w-3 rounded ${TONES[MODELS[m].tone].bg}`}
                      aria-hidden
                    />
                    {modelName(m)}
                  </span>
                </td>
                {horizons.map((d) => {
                  const f = forecasts.find((x) => x.model === m && x.targetDate === d);
                  return (
                    <td key={d} className="py-2 pr-4">
                      {f ? (
                        <>
                          <strong className="font-semibold">{pct(f.point, 2)}</strong>
                          {f.lower != null && f.upper != null && (
                            <span className="text-muted">
                              {" "}
                              ({f.lower.toFixed(1)}–{f.upper.toFixed(1)})
                            </span>
                          )}
                        </>
                      ) : (
                        "–"
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FomcSection({ data }: { data: DashboardData }) {
  const meetings = [...new Set(data.fomc.map((f) => f.meeting))].sort();
  const rate = data.policyRate[data.policyRate.length - 1];
  const headline = (meeting: string): FomcForecast | undefined =>
    data.fomc.find((f) => f.meeting === meeting && f.model === "fomc_lightgbm");

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">
          Federal Reserve rate decisions
        </h2>
        <p className="mt-1 text-muted">
          {rate && (
            <>
              The fed funds target is{" "}
              <strong className="text-foreground">
                {(rate.value - 0.125).toFixed(2)}–{(rate.value + 0.125).toFixed(2)}%
              </strong>
              .{" "}
            </>
          )}
          Each bar shows how likely each model thinks a cut, hold or hike is at that
          meeting.
        </p>
      </div>
      {meetings.length === 0 && <p className="text-muted">No open FOMC questions.</p>}
      {meetings.map((meeting) => {
        const rows = byOrder(
          data.fomc.filter((f) => f.meeting === meeting),
          FOMC_ORDER,
        );
        const lead = rows[0];
        const top = headline(meeting);
        const likeliest = top
          ? (Object.entries(top.probabilities).sort((a, b) => b[1] - a[1])[0] as [
              string,
              number,
            ])
          : null;
        return (
          <div key={meeting} className="rounded-lg border border-line bg-panel p-4">
            <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="font-semibold">Meeting of {longDate(meeting)}</h3>
              <p className="text-xs text-muted">
                {lead.leadDays != null
                  ? `forecast ${lead.leadDays} days before`
                  : `forecast ${lead.horizonMonths} month(s) ahead`}{" "}
                on {longDate(lead.forecastDate)}
              </p>
            </div>
            {likeliest && (
              <p className="mb-3 text-sm">
                Most likely: <strong>{likeliest[0]}</strong> (
                {(likeliest[1] * 100).toFixed(0)}% per the FOMC model)
              </p>
            )}
            <div className="space-y-3">
              {rows.map((f) => (
                <ProbabilityBar
                  key={f.model}
                  label={modelName(f.model)}
                  probabilities={f.probabilities}
                />
              ))}
            </div>
          </div>
        );
      })}
      {data.policyRate.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold">
            Fed funds target, last three years
          </h3>
          <TimeChart
            series={[
              {
                id: "rate",
                label: "Target (mid)",
                tone: "ink",
                kind: "actual",
                points: data.policyRate,
              },
            ]}
            step
            ariaLabel="Fed funds target rate over the last three years"
          />
        </div>
      )}
    </section>
  );
}

function HowToRead() {
  const used = [...NUMERIC_ORDER, ...FOMC_ORDER];
  return (
    <section className="rounded-lg border border-line bg-panel p-5 text-sm">
      <h2 className="font-semibold">How to read this</h2>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-muted">
        <li>
          <strong className="text-foreground">Ranges</strong> (the shaded bars on the
          charts and the numbers in brackets) are 80% intervals: the model expects the
          actual value to land inside 8 times out of 10.{" "}
          <strong className="text-foreground">Treat them as optimistic.</strong> In our
          2020–2023 backtest they held the actual value only 35–75% of the time.
        </li>
        <li>
          Forecasts use only data published before the day they were made, and they are
          never edited. They are scored once the official numbers come out (first
          release), on the Track record page.
        </li>
        <li>
          Forecasts are made on the first of each month for 1, 3 and 6 months ahead, and
          for each FOMC meeting a week before it.
        </li>
      </ul>
      <h3 className="mt-5 font-semibold">The models</h3>
      <dl className="mt-2 grid gap-3 sm:grid-cols-2">
        {used.map((m) => (
          <div key={m}>
            <dt className="flex items-center gap-2 font-medium">
              <span
                className={`inline-block h-0.5 w-3 rounded ${TONES[MODELS[m].tone].bg}`}
                aria-hidden
              />
              {MODELS[m].name}
            </dt>
            <dd className="mt-0.5 text-muted">{MODELS[m].blurb}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export async function Dashboard() {
  await connection(); // read the database per request, not at build time
  let data: DashboardData;
  try {
    data = await loadDashboard(new Date());
  } catch (error) {
    return (
      <div className="rounded-lg border border-line bg-panel p-5">
        <p className="font-semibold">Can&apos;t load forecasts right now.</p>
        <p className="mt-1 text-sm text-muted">
          The database didn&apos;t answer. Is Docker running, and is DATABASE_URL set in
          web/.env.local?
        </p>
        <p className="mt-2 font-mono text-xs text-muted">{String(error)}</p>
      </div>
    );
  }

  const numericFor = (t: NumericTarget) => data.numeric.filter((f) => f.target === t);
  const tileFor = (t: NumericTarget) => {
    const history = data.history[t];
    const latest = history[history.length - 1];
    const forecasts = numericFor(t);
    const dates = [...new Set(forecasts.map((f) => f.targetDate))].sort();
    const mid = dates[Math.floor(dates.length / 2)];
    const values = forecasts.filter((f) => f.targetDate === mid).map((f) => f.point);
    return { latest, mid, values };
  };
  const nextMeeting = [...new Set(data.fomc.map((f) => f.meeting))].sort()[0];
  const nextTop = data.fomc.find(
    (f) => f.meeting === nextMeeting && f.model === "fomc_lightgbm",
  );
  const nextLikeliest = nextTop
    ? Object.entries(nextTop.probabilities).sort((a, b) => b[1] - a[1])[0]
    : null;

  return (
    <div className="space-y-12">
      <p className="text-sm text-muted">
        {data.lastRun
          ? `Latest forecasts made ${longDate(data.lastRun)}`
          : "No forecasts yet"}
        {data.dataThrough ? ` · data through ${longDate(data.dataThrough)}` : ""}
      </p>

      <div className="grid gap-4 sm:grid-cols-3">
        {(["cpi_yoy", "unemployment"] as const).map((t) => {
          const { latest, mid, values } = tileFor(t);
          return (
            <Tile
              key={t}
              label={t === "cpi_yoy" ? "Inflation" : "Unemployment"}
              value={latest ? pct(latest.value) : "–"}
              detail={
                latest && mid && values.length ? (
                  <>
                    in {monthYear(latest.date)} · expected {range(values)} by{" "}
                    {monthYear(mid)}
                  </>
                ) : (
                  "no open forecasts"
                )
              }
            />
          );
        })}
        <Tile
          label="Next forecast FOMC meeting"
          value={
            nextLikeliest
              ? `${nextLikeliest[0][0].toUpperCase()}${nextLikeliest[0].slice(1)} ${(
                  Number(nextLikeliest[1]) * 100
                ).toFixed(0)}%`
              : "–"
          }
          detail={
            nextMeeting ? `${longDate(nextMeeting)} · FOMC model` : "no open questions"
          }
        />
      </div>

      {(["cpi_yoy", "unemployment"] as const).map((t) =>
        data.history[t].length && numericFor(t).length ? (
          <NumericSection
            key={t}
            target={t}
            history={data.history[t]}
            forecasts={numericFor(t)}
          />
        ) : null,
      )}
      <FomcSection data={data} />
      <HowToRead />
    </div>
  );
}

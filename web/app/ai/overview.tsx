import Link from "next/link";
import { connection } from "next/server";
import type { ReactNode } from "react";

import { longDate, monthYear } from "../_components/format";
import { type ChartSeries, TimeChart } from "../_components/time-chart";
import { TONES } from "../_components/tones";
import { type AISeries, loadAIOverview } from "@/lib/ai-dashboard-data";
import {
  AI_MODELS,
  KIND_LABELS,
  NUMERIC_MODEL_ORDER,
  aiModelName,
  aiModelTone,
  formatValue,
  SERIES_SHORT,
  unitNote,
  usd,
} from "./models";

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

function SeriesSection({ s }: { s: AISeries }) {
  const latest = s.history[s.history.length - 1];
  const targets = [...new Set(s.forecasts.map((f) => f.targetDate))].sort();
  const models = NUMERIC_MODEL_ORDER.filter((m) =>
    s.forecasts.some((f) => f.model === m),
  );
  // One shaded 80% range (the first model in order, normally LightGBM): five
  // overlapping bands smear together. The table lists every model's range.
  const banded = models.find((m) => m !== "claude_llm") ?? models[0];
  const series: ChartSeries[] = [
    {
      id: "actual",
      label: "Actual",
      tone: "ink",
      kind: "actual",
      points: s.history.map((p) => ({ date: p.date, value: p.value })),
    },
    ...models.map((m) => ({
      id: m,
      label: aiModelName(m),
      tone: aiModelTone(m),
      kind: "forecast" as const,
      points: [
        ...(latest ? [{ date: latest.date, value: latest.value }] : []),
        ...s.forecasts
          .filter((f) => f.model === m && f.point !== null)
          .sort((a, b) => a.targetDate.localeCompare(b.targetDate))
          .map((f) => ({
            date: f.targetDate,
            value: f.point as number,
            lower: m === banded ? f.lower : null,
            upper: m === banded ? f.upper : null,
          })),
      ],
    })),
  ];
  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-lg font-semibold tracking-tight">
          {SERIES_SHORT[s.seriesId] ?? s.name}
        </h3>
        <p className="text-xs text-muted">{s.name}</p>
        <p className="mt-1 text-sm text-muted">
          {latest ? (
            <>
              <strong className="text-foreground">
                {formatValue(latest.value, s.kind)}
              </strong>{" "}
              in {monthYear(latest.date)} ({unitNote(s.kind)}).
            </>
          ) : (
            "No data yet."
          )}
        </p>
      </div>
      {latest && models.length > 0 ? (
        <div>
          <TimeChart
            series={series}
            splitDate={latest.date}
            format={s.kind}
            step={s.kind !== "count"}
            ariaLabel={`${s.name}: the last three years and each model's forecast`}
          />
          <p className="text-xs text-muted">
            Shaded: {aiModelName(banded)}&apos;s 80% range (every model&apos;s is in the
            table).
          </p>
        </div>
      ) : (
        <p className="text-sm text-muted">No open forecasts for this series.</p>
      )}
      {models.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[32rem] text-sm tabular-nums">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="py-2 pr-4 font-medium">Model</th>
                {targets.map((d) => (
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
                    <span
                      className="flex items-center gap-2"
                      title={AI_MODELS[m]?.blurb}
                    >
                      <span
                        className={`inline-block h-0.5 w-3 rounded ${TONES[aiModelTone(m)].bg}`}
                        aria-hidden
                      />
                      {aiModelName(m)}
                    </span>
                  </td>
                  {targets.map((d) => {
                    const f = s.forecasts.find(
                      (x) => x.model === m && x.targetDate === d,
                    );
                    return (
                      <td key={d} className="py-2 pr-4">
                        {f?.point != null ? (
                          <>
                            <strong className="font-semibold">
                              {formatValue(f.point, s.kind)}
                            </strong>
                            {f.lower != null && f.upper != null && (
                              <span className="text-muted">
                                {" "}
                                ({formatValue(f.lower, s.kind)}–
                                {formatValue(f.upper, s.kind)})
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
      )}
    </section>
  );
}

export async function Overview() {
  await connection(); // read the database per request, not at build time
  const data = await loadAIOverview(new Date());
  const open = data.open.numeric + data.open.threshold + data.open.curated;
  return (
    <div className="space-y-12">
      <section className="grid gap-4 sm:grid-cols-4">
        <Tile
          label="Open questions"
          value={open.toLocaleString("en-US")}
          detail={
            <Link href="/ai/questions" className="underline-offset-2 hover:underline">
              {data.open.numeric} numeric · {data.open.threshold} thresholds ·{" "}
              {data.open.curated} Polymarket
            </Link>
          }
        />
        <Tile
          label="Live forecasts"
          value={data.liveForecasts.toLocaleString("en-US")}
          detail={data.lastRun ? `Last run ${longDate(data.lastRun)}` : "No run yet"}
        />
        <Tile
          label="Scored"
          value={data.scored.toLocaleString("en-US")}
          detail={
            <Link
              href="/ai/track-record"
              className="underline-offset-2 hover:underline"
            >
              See the track record
            </Link>
          }
        />
        <Tile
          label="LLM spend"
          value={usd(data.llm.spentUsd)}
          detail={
            data.llm.calls
              ? `${data.llm.calls} calls, ${usd(data.llm.spentUsd / data.llm.calls)} each`
              : "The LLM hasn't been run yet"
          }
        />
      </section>

      <section className="space-y-10">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">
            {KIND_LABELS.numeric.title}
          </h2>
          <p className="mt-1 max-w-2xl text-muted">
            Each series&apos; last three years and every model&apos;s forecast 1, 3 and
            6 months ahead, with 80% ranges.
            {data.dataThrough && <> Data through {longDate(data.dataThrough)}.</>}
          </p>
        </div>
        {data.series.map((s) => (
          <SeriesSection key={s.seriesId} s={s} />
        ))}
      </section>
    </div>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { connection } from "next/server";
import { Suspense } from "react";

import { longDate } from "../../../_components/format";
import { TimeChart } from "../../../_components/time-chart";
import { TONES } from "../../../_components/tones";
import {
  type AIForecastDetail,
  type AIQuestionDetail,
  loadAIQuestion,
  seriesKind,
} from "@/lib/ai-dashboard-data";
import {
  AI_MODELS,
  KIND_LABELS,
  aiModelName,
  aiModelTone,
  formatValue,
  probability,
  usd,
} from "../../models";

export const metadata: Metadata = { title: "AI forecast" };

const CATEGORY_LABELS: Record<string, string> = {
  expected_variance: "Within its own uncertainty",
  bad_model: "Model missed an available signal",
  regime_change: "A surprise nobody priced",
  bad_data: "The data was revised",
};

function value(q: AIQuestionDetail, f: AIForecastDetail): string {
  if (f.pYes != null) return `P(yes) ${probability(f.pYes)}`;
  if (f.point == null) return "–";
  const kind = seriesKind(q.units ?? "");
  const range =
    f.lower != null && f.upper != null
      ? ` (80%: ${formatValue(f.lower, kind)}–${formatValue(f.upper, kind)})`
      : "";
  return `${formatValue(f.point, kind)}${range}`;
}

function ForecastCard({ q, f }: { q: AIQuestionDetail; f: AIForecastDetail }) {
  const isLLM = f.model === "claude_llm";
  return (
    <li className="rounded-lg border border-line bg-panel p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="flex items-center gap-2 font-semibold">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${TONES[aiModelTone(f.model)].bg}`}
            aria-hidden
          />
          {aiModelName(f.model)}
        </p>
        <p className="text-xs text-muted">
          {f.isBacktest ? "Backtest" : "Live"}, made {longDate(f.forecastDate)}
          {f.llmCostUsd != null && <> · {usd(f.llmCostUsd)}</>}
        </p>
      </div>
      <p className="mt-2 text-lg tabular-nums">{value(q, f)}</p>
      {AI_MODELS[f.model] && (
        <p className="mt-1 text-sm text-muted">{AI_MODELS[f.model].blurb}</p>
      )}
      {(f.brier != null || f.error != null) && (
        <p className="mt-2 text-sm">
          Score:{" "}
          {f.brier != null ? (
            <strong>Brier {f.brier.toFixed(3)}</strong>
          ) : (
            <strong>
              error {f.error! > 0 ? "+" : ""}
              {formatValue(f.error!, seriesKind(q.units ?? ""))}
              {f.inInterval === false ? " (outside its range)" : ""}
            </strong>
          )}
        </p>
      )}
      {f.rationale && (
        <details className="mt-3" open={isLLM}>
          <summary className="cursor-pointer text-sm font-medium">Rationale</summary>
          <p className="mt-2 whitespace-pre-line text-sm leading-relaxed">
            {f.rationale}
          </p>
        </details>
      )}
      {f.drivers.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">
            What drove it
          </summary>
          <ul className="mt-2 space-y-1 text-sm tabular-nums">
            {f.drivers.map(([name, v]) => (
              <li key={name} className="flex justify-between gap-4">
                <span className="font-mono text-xs">{name}</span>
                <span className="text-muted">
                  {v > 0 ? "+" : ""}
                  {v.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-muted">
            SHAP contributions, on the model&apos;s scale.
          </p>
        </details>
      )}
      {f.postmortem && f.postmortem.category !== "expected_variance" && (
        <div className="mt-3 rounded-md border border-line p-3 text-sm">
          <p className="font-medium">
            Why it missed:{" "}
            {CATEGORY_LABELS[f.postmortem.category] ?? f.postmortem.category}
          </p>
          {f.postmortem.notes && (
            <p className="mt-1 text-muted">{f.postmortem.notes}</p>
          )}
          {f.postmortem.draft && (
            <p className="mt-2">
              <span className="text-xs font-medium uppercase tracking-wide text-muted">
                {f.postmortem.confirmed
                  ? "Reviewed note"
                  : "LLM draft, not yet reviewed"}
              </span>
              <br />
              {f.postmortem.draft}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

async function Detail({ id }: { id: number }) {
  await connection();
  const q = await loadAIQuestion(id);
  if (!q) notFound();
  const kind = q.kind === "curated" ? "percent" : seriesKind(q.units ?? "");
  const live = q.forecasts.filter((f) => !f.isBacktest);
  const backtest = q.forecasts.filter((f) => f.isBacktest);
  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm text-muted">
          <Link href="/ai" className="hover:text-foreground">
            AI progress
          </Link>{" "}
          /{" "}
          <Link href={`/ai/questions?kind=${q.kind}`} className="hover:text-foreground">
            {KIND_LABELS[q.kind].title}
          </Link>
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{q.text}</h1>
        <p className="mt-2 text-sm text-muted">
          Resolves {longDate(q.targetDate)}
          {q.horizonMonths != null && <> · asked {q.horizonMonths} months ahead</>}
          {q.leadDays != null && (
            <> · asked {q.leadDays} days before the market closes</>
          )}
          {q.url && (
            <>
              {" · "}
              <a
                href={q.url}
                target="_blank"
                rel="noreferrer"
                className="underline-offset-2 hover:underline"
              >
                Polymarket ↗
              </a>
            </>
          )}
        </p>
        <p className="mt-3 rounded-md border border-line bg-panel p-3 text-sm">
          {q.resolution ? (
            <>
              <strong>Resolved</strong>{" "}
              {q.resolution.outcome ?? formatValue(q.resolution.value ?? 0, kind)}{" "}
              (known {longDate(q.resolution.asOf)}).
            </>
          ) : (
            <>
              <strong>Open.</strong> No answer yet.
            </>
          )}
        </p>
        {q.rule && (
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer text-muted">How it resolves</summary>
            <p className="mt-2 whitespace-pre-line text-muted">{q.rule}</p>
          </details>
        )}
      </div>

      {q.history.length > 1 && (
        <section className="space-y-2">
          <h2 className="text-lg font-semibold tracking-tight">
            {q.kind === "curated" ? "The market's price" : "The series"}
          </h2>
          <TimeChart
            series={[
              {
                id: "actual",
                label: q.kind === "curated" ? "Polymarket" : "Actual",
                tone: "ink",
                kind: "actual",
                points: q.history,
              },
            ]}
            format={kind}
            step={kind !== "count"}
            ariaLabel={
              q.kind === "curated"
                ? "Polymarket's daily price"
                : "The series' recent values"
            }
          />
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-lg font-semibold tracking-tight">Forecasts</h2>
        {live.length === 0 && backtest.length === 0 && (
          <p className="text-muted">No forecasts yet.</p>
        )}
        <ul className="grid gap-4 sm:grid-cols-2">
          {live.map((f) => (
            <ForecastCard key={f.id} q={q} f={f} />
          ))}
        </ul>
        {backtest.length > 0 && (
          // Open when they're all there is (a question first forecast late).
          <details open={live.length === 0}>
            <summary className="cursor-pointer text-sm text-muted">
              {backtest.length} backtest forecast{backtest.length > 1 ? "s" : ""} (made
              after the fact, with data cut off at the time)
            </summary>
            <ul className="mt-3 grid gap-4 sm:grid-cols-2">
              {backtest.map((f) => (
                <ForecastCard key={f.id} q={q} f={f} />
              ))}
            </ul>
          </details>
        )}
      </section>
    </div>
  );
}

async function DetailFromParams({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const questionId = Number(id);
  if (!Number.isInteger(questionId) || questionId <= 0) notFound();
  return <Detail id={questionId} />;
}

/** `params` is read inside the Suspense boundary (Cache Components). */
export default function AIForecastPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  return (
    <Suspense fallback={<p className="text-muted">Loading the question…</p>}>
      <DetailFromParams params={params} />
    </Suspense>
  );
}

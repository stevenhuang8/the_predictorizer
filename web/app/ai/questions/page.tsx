import type { Metadata } from "next";
import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";

import { longDate, monthYear } from "../../_components/format";
import { TONES } from "../../_components/tones";
import {
  type AIQuestionRow,
  type QuestionKind,
  loadAIQuestions,
} from "@/lib/ai-dashboard-data";
import {
  BINARY_MODEL_ORDER,
  KIND_LABELS,
  NUMERIC_MODEL_ORDER,
  aiModelName,
  aiModelTone,
  formatValue,
  probability,
  questionLabel,
  seriesKindOfKey,
} from "../models";

export const metadata: Metadata = { title: "AI questions" };

const KINDS: QuestionKind[] = ["numeric", "threshold", "curated"];
const LIMIT = 200;

function ModelCell({ q, model }: { q: AIQuestionRow; model: string }) {
  const f = q.forecasts.find((x) => x.model === model);
  if (!f) return <span className="text-muted">–</span>;
  if (f.pYes != null) {
    return (
      <span
        className="flex items-center gap-2"
        title={`${aiModelName(model)}: ${probability(f.pYes)}`}
      >
        <span className="w-9 text-right tabular-nums">{probability(f.pYes)}</span>
        <span className="h-2 w-12 shrink-0 rounded-sm bg-line" aria-hidden>
          <span
            className={`block h-full rounded-sm ${TONES[aiModelTone(model)].bg}`}
            style={{ width: `${Math.max(2, f.pYes * 100)}%` }}
          />
        </span>
      </span>
    );
  }
  return f.point != null ? (
    <span className="tabular-nums">{formatValue(f.point, seriesKindOfKey(q.key))}</span>
  ) : (
    <span className="text-muted">–</span>
  );
}

function when(q: AIQuestionRow): string {
  if (q.horizonMonths != null) return `${q.horizonMonths} mo ahead`;
  if (q.leadDays != null) return `asked ${q.leadDays} d before`;
  return "";
}

async function QuestionList({ kind }: { kind?: QuestionKind }) {
  await connection();
  const { rows, total } = await loadAIQuestions(new Date(), kind, LIMIT);
  if (rows.length === 0) return <p className="text-muted">No open questions.</p>;
  const groups = KINDS.filter((k) => !kind || k === kind)
    .map((k) => ({ kind: k, rows: rows.filter((r) => r.kind === k) }))
    .filter((g) => g.rows.length > 0);
  return (
    <div className="space-y-10">
      {total > rows.length && (
        <p className="text-sm text-muted">
          Showing the {rows.length} resolving soonest of {total}. Filter by kind to see
          more.
        </p>
      )}
      {groups.map((g) => {
        const order = g.kind === "numeric" ? NUMERIC_MODEL_ORDER : BINARY_MODEL_ORDER;
        const models = order.filter((m) =>
          g.rows.some((r) => r.forecasts.some((f) => f.model === m)),
        );
        return (
          <section key={g.kind} className="space-y-3">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">
                {KIND_LABELS[g.kind].title}
              </h2>
              <p className="text-sm text-muted">{KIND_LABELS[g.kind].blurb}</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[40rem] text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-muted">
                    <th className="py-2 pr-4 font-medium">Question</th>
                    <th className="py-2 pr-4 font-medium">Resolves</th>
                    {models.map((m) => (
                      <th key={m} className="py-2 pr-4 font-medium whitespace-nowrap">
                        {aiModelName(m)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {g.rows.map((q) => (
                    <tr key={q.id} className="border-b border-line align-top">
                      <td className="max-w-md py-2 pr-4">
                        <Link
                          href={`/ai/forecasts/${q.id}`}
                          className="hover:underline"
                        >
                          {questionLabel(q.key, q.text, monthYear(q.targetDate))}
                        </Link>
                        {q.url && (
                          <a
                            href={q.url}
                            target="_blank"
                            rel="noreferrer"
                            className="ml-2 text-xs text-muted hover:text-foreground"
                          >
                            Polymarket ↗
                          </a>
                        )}
                      </td>
                      <td className="py-2 pr-4 whitespace-nowrap text-muted tabular-nums">
                        {longDate(q.targetDate)}
                        <span className="block text-xs">{when(q)}</span>
                      </td>
                      {models.map((m) => (
                        <td key={m} className="py-2 pr-4">
                          <ModelCell q={q} model={m} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}

async function KindsAndList({
  searchParams,
}: {
  searchParams: Promise<{ kind?: string }>;
}) {
  const { kind } = await searchParams;
  const chosen = KINDS.find((k) => k === kind);
  return (
    <div className="space-y-8">
      <nav className="flex flex-wrap gap-2 text-sm" aria-label="Question kind">
        {[undefined, ...KINDS].map((k) => (
          <Link
            key={k ?? "all"}
            href={k ? `/ai/questions?kind=${k}` : "/ai/questions"}
            aria-current={chosen === k ? "page" : undefined}
            className={`rounded-md border px-3 py-1.5 ${
              chosen === k ? "border-foreground font-medium" : "border-line text-muted"
            }`}
          >
            {k ? KIND_LABELS[k].title : "All"}
          </Link>
        ))}
      </nav>
      <QuestionList kind={chosen} />
    </div>
  );
}

/** `searchParams` is read inside the Suspense boundary (Cache Components). */
export default function AIQuestionsPage({
  searchParams,
}: {
  searchParams: Promise<{ kind?: string }>;
}) {
  return (
    <div className="space-y-8">
      <div>
        <p className="text-sm text-muted">
          <Link href="/ai" className="hover:text-foreground">
            AI progress
          </Link>{" "}
          / Questions
        </p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">
          Open AI questions
        </h1>
        <p className="mt-2 max-w-2xl text-muted">
          Every question still waiting for its answer, with each model&apos;s latest
          live forecast. Open one for the reasoning behind it.
        </p>
      </div>
      <Suspense fallback={<p className="text-muted">Loading questions…</p>}>
        <KindsAndList searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

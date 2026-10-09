import Link from "next/link";

/** Placeholder for a page whose data views come in Phase 4. */
export function ComingSoon({
  title,
  summary,
  planned,
  source,
}: {
  title: string;
  summary: string;
  planned: string[];
  source: string;
}) {
  return (
    <article className="max-w-2xl">
      <p className="text-sm font-medium text-muted">Coming soon</p>
      <h1 className="mt-1 text-3xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-4 text-muted">{summary}</p>
      <section className="mt-8 rounded-lg border border-line bg-panel p-5">
        <h2 className="text-sm font-semibold">Planned</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-muted">
          {planned.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-muted">
          Data: <code className="font-mono">{source}</code>
        </p>
      </section>
      <Link href="/" className="mt-8 inline-block text-sm underline underline-offset-4">
        Back to overview
      </Link>
    </article>
  );
}

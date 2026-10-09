import { DECISION_BG } from "./tones";

const ORDER = ["cut", "hold", "hike"] as const;
type Outcome = (typeof ORDER)[number];

const LABEL: Record<Outcome, string> = { cut: "Cut", hold: "Hold", hike: "Hike" };

/**
 * Cut / hold / hike as one diverging stacked bar (blue, neutral, red), with
 * each share labeled in the segment when it fits and always in the legend row.
 */
export function ProbabilityBar({
  probabilities,
  label,
}: {
  probabilities: Record<Outcome, number>;
  label: string;
}) {
  const percent = (o: Outcome) => `${((probabilities[o] ?? 0) * 100).toFixed(0)}%`;
  return (
    <div className="grid grid-cols-[minmax(7rem,9rem)_1fr] items-center gap-3">
      <span className="text-sm text-muted">{label}</span>
      <div>
        <div
          className="flex h-7 gap-0.5 overflow-hidden rounded"
          role="img"
          aria-label={`${label}: ${ORDER.map((o) => `${LABEL[o]} ${percent(o)}`).join(", ")}`}
        >
          {ORDER.map((o) => {
            const share = probabilities[o] ?? 0;
            return share > 0.005 ? (
              <div
                key={o}
                title={`${LABEL[o]}: ${percent(o)}`}
                className={`flex items-center justify-center text-xs font-medium ${DECISION_BG[o]} ${
                  o === "hold" ? "text-foreground" : "text-white"
                }`}
                style={{ width: `${share * 100}%` }}
              >
                {share >= 0.12 ? `${LABEL[o]} ${percent(o)}` : ""}
              </div>
            ) : null;
          })}
        </div>
        <p className="mt-1 text-xs text-muted tabular-nums">
          {ORDER.map((o) => `${LABEL[o]} ${percent(o)}`).join(" · ")}
        </p>
      </div>
    </div>
  );
}

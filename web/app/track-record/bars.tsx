/**
 * Inline bars for score tables: the number is always printed, so the bar only
 * adds a sense of size. One neutral hue; the best value in its group is ink.
 */
export function ValueBar({
  value,
  max,
  best = false,
  digits = 2,
  label,
}: {
  value: number;
  max: number;
  best?: boolean;
  digits?: number;
  label: string;
}) {
  const width = max > 0 ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return (
    <span
      className="flex items-center gap-2"
      title={`${label}: ${value.toFixed(digits)}`}
    >
      <span className={`w-10 text-right tabular-nums ${best ? "font-semibold" : ""}`}>
        {value.toFixed(digits)}
      </span>
      <span className="h-2 w-20 shrink-0" aria-hidden>
        <span
          className={`block h-full rounded-r-sm ${best ? "bg-foreground" : "bg-muted/45"}`}
          style={{ width: `${width}%` }}
        />
      </span>
    </span>
  );
}

/**
 * How often actuals landed inside a stated interval: the fill is the share
 * observed, the tick the share promised (80% by default).
 */
export function CoverageMeter({
  share,
  nominal = 0.8,
}: {
  share: number;
  nominal?: number;
}) {
  const percent = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <span
      className="flex items-center gap-2"
      title={`Held the actual ${percent(share)} of the time; promised ${percent(nominal)}`}
    >
      <span className="w-10 text-right tabular-nums">{percent(share)}</span>
      <span className="relative h-2 w-20 shrink-0 rounded-sm bg-line" aria-hidden>
        <span
          className="block h-full rounded-sm bg-muted/70"
          style={{ width: `${share * 100}%` }}
        />
        <span
          className="absolute -top-1 h-4 w-0.5 -translate-x-1/2 rounded-full bg-foreground"
          style={{ left: `${nominal * 100}%` }}
        />
      </span>
    </span>
  );
}

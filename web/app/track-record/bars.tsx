import { type Tone, TONES } from "../_components/tones";

/**
 * Inline bars for score tables, in the model's color: the number is always
 * printed, so the bar only adds a sense of size. The best value in its group
 * has a bold number.
 */
export function ValueBar({
  value,
  max,
  tone,
  best = false,
  digits = 2,
  label,
}: {
  value: number;
  max: number;
  tone: Tone;
  best?: boolean;
  digits?: number;
  label: string;
}) {
  const width = max > 0 ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return (
    <span
      className="flex items-center gap-2"
      title={`${label}: ${value.toFixed(digits)}${best ? " (best)" : ""}`}
    >
      <span className={`w-10 text-right tabular-nums ${best ? "font-semibold" : ""}`}>
        {value.toFixed(digits)}
      </span>
      <span className="h-2 w-20 shrink-0" aria-hidden>
        <span
          className={`block h-full rounded-r-sm ${TONES[tone].bg}`}
          style={{ width: `${width}%` }}
        />
      </span>
    </span>
  );
}

/**
 * How often actuals landed inside a stated interval: the fill (model color)
 * is the share observed, the tick the share promised (80% by default).
 */
export function CoverageMeter({
  share,
  tone,
  nominal = 0.8,
}: {
  share: number;
  tone: Tone;
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
          className={`block h-full rounded-sm ${TONES[tone].bg}`}
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

/** A model's name with its line key, as in the forecasts page tables. */
export function ModelLabel({ name, tone }: { name: string; tone: Tone }) {
  return (
    <span className="flex items-center gap-2 whitespace-nowrap">
      <span
        className={`inline-block h-0.5 w-3 rounded ${TONES[tone].bg}`}
        aria-hidden
      />
      {name}
    </span>
  );
}

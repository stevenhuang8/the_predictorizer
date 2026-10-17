"use client";

import { useState } from "react";

import { type Tone, TONES } from "../../_components/tones";

export type CalibrationLine = {
  model: string;
  label: string;
  tone: Tone;
  bins: { lower: number; predicted: number; observed: number; n: number }[];
};

const S = 300;
const M = { top: 12, right: 12, bottom: 34, left: 40 };
const PLOT = S - M.left - M.right;

/**
 * Predicted probability against how often it happened, one line per model,
 * dots sized by how many forecasts fell in the bin. A calibrated forecaster
 * sits on the dashed diagonal. Hover a dot for its numbers.
 */
export function CalibrationChart({ lines }: { lines: CalibrationLine[] }) {
  const [hover, setHover] = useState<{ line: CalibrationLine; i: number } | null>(null);
  const x = (p: number) => M.left + p * PLOT;
  const y = (p: number) => M.top + (1 - p) * PLOT;
  const maxN = Math.max(1, ...lines.flatMap((l) => l.bins.map((b) => b.n)));
  const r = (n: number) => 4 + 6 * Math.sqrt(n / maxN);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const hovered = hover ? hover.line.bins[hover.i] : null;
  return (
    <figure className="relative max-w-sm">
      <svg
        viewBox={`0 0 ${S} ${S}`}
        className="w-full"
        role="img"
        aria-label="Calibration: predicted probability against observed frequency, by model"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={x(0)}
              x2={x(1)}
              y1={y(t)}
              y2={y(t)}
              className="stroke-line"
              strokeWidth={1}
            />
            <text
              x={M.left - 6}
              y={y(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-muted text-[10px]"
            >
              {t * 100}%
            </text>
            <text
              x={x(t)}
              y={S - M.bottom + 14}
              textAnchor="middle"
              className="fill-muted text-[10px]"
            >
              {t * 100}%
            </text>
          </g>
        ))}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          className="stroke-muted"
          strokeDasharray="4 4"
          strokeWidth={1.5}
        />
        <text
          x={x(0.5)}
          y={S - 4}
          textAnchor="middle"
          className="fill-muted text-[10px]"
        >
          predicted
        </text>
        {lines.map((line) => (
          <g key={line.model}>
            <polyline
              points={line.bins
                .map((b) => `${x(b.predicted)},${y(b.observed)}`)
                .join(" ")}
              fill="none"
              strokeWidth={2}
              className={TONES[line.tone].stroke}
            />
            {line.bins.map((b, i) => (
              <circle
                key={b.lower}
                cx={x(b.predicted)}
                cy={y(b.observed)}
                r={r(b.n)}
                strokeWidth={2}
                className={`${TONES[line.tone].fill} stroke-background cursor-pointer`}
                onPointerEnter={() => setHover({ line, i })}
                onPointerLeave={() => setHover(null)}
              />
            ))}
          </g>
        ))}
      </svg>
      {hover && hovered && (
        <div className="pointer-events-none absolute top-2 right-2 rounded-md border border-line bg-panel px-3 py-2 text-xs shadow-sm">
          <p className="font-medium">{hover.line.label}</p>
          <p className="tabular-nums">
            Said {(hovered.predicted * 100).toFixed(0)}% on average; happened{" "}
            {(hovered.observed * 100).toFixed(0)}% ({hovered.n} forecast
            {hovered.n > 1 ? "s" : ""})
          </p>
        </div>
      )}
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {lines.map((l) => (
          <span key={l.model} className="flex items-center gap-1.5">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${TONES[l.tone].bg}`}
              aria-hidden
            />
            {l.label}
          </span>
        ))}
        <span>Dashed: perfectly calibrated</span>
      </figcaption>
    </figure>
  );
}

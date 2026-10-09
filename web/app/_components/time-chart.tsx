"use client";

import { useMemo, useRef, useState } from "react";
import { monthYear, utcDate } from "./format";
import { type Tone, TONES } from "./tones";

export type ChartPoint = {
  date: string;
  value: number;
  lower?: number | null;
  upper?: number | null;
};

export type ChartSeries = {
  id: string;
  label: string;
  tone: Tone;
  points: ChartPoint[];
  kind: "actual" | "forecast";
};

const W = 720;
const H = 300;
const M = { top: 16, right: 116, bottom: 30, left: 44 };
const LABEL_GAP = 15;

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) {
    ticks.push(Number(t.toFixed(6)));
  }
  return ticks;
}

/**
 * Actual history (ink) with model forecasts extending past `splitDate`, each
 * forecast point carrying its interval as a band. Hover shows every series at
 * the nearest month. `step` draws the actual line as a step function.
 *
 * Colors come from TONES (Tailwind classes), never var() in attributes: the
 * CSS pipeline drops variables that no rule uses.
 */
export function TimeChart({
  series,
  splitDate,
  unit = "%",
  step = false,
  ariaLabel,
}: {
  series: ChartSeries[];
  splitDate?: string;
  unit?: string;
  step?: boolean;
  ariaLabel: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const layout = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    const times = all.map((p) => utcDate(p.date).getTime());
    const values = all.flatMap((p) =>
      [p.value, p.lower, p.upper].filter((v): v is number => v != null),
    );
    const x0 = Math.min(...times);
    const x1 = Math.max(...times);
    const pad = (Math.max(...values) - Math.min(...values)) * 0.08 || 0.5;
    const yTicks = niceTicks(Math.min(...values) - pad, Math.max(...values) + pad);
    const y0 = Math.min(yTicks[0], Math.min(...values) - pad);
    const y1 = Math.max(yTicks[yTicks.length - 1], Math.max(...values) + pad);
    const x = (iso: string) =>
      M.left +
      ((utcDate(iso).getTime() - x0) / (x1 - x0 || 1)) * (W - M.left - M.right);
    const y = (v: number) =>
      H - M.bottom - ((v - y0) / (y1 - y0)) * (H - M.top - M.bottom);
    const xTicks: string[] = [];
    for (
      let yr = new Date(x0).getUTCFullYear();
      yr <= new Date(x1).getUTCFullYear();
      yr++
    ) {
      const iso = `${yr}-01-01`;
      const t = utcDate(iso).getTime();
      if (t >= x0 && t <= x1) xTicks.push(iso);
    }
    const dates = [...new Set(all.map((p) => p.date))].sort();
    return { x, y, yTicks, xTicks, dates };
  }, [series]);

  const { x, y, yTicks, xTicks, dates } = layout;

  // End-of-line labels, spread apart with leader lines where they'd collide.
  const labels = useMemo(() => {
    const ends = series
      .filter((s) => s.points.length)
      .map((s) => {
        const last = s.points[s.points.length - 1];
        return {
          id: s.id,
          label: s.label,
          tone: s.tone,
          ex: x(last.date),
          ey: y(last.value),
        };
      })
      .sort((a, b) => a.ey - b.ey);
    const placed = ends.map((e) => ({ ...e, ly: e.ey }));
    for (let i = 1; i < placed.length; i++) {
      placed[i].ly = Math.max(placed[i].ly, placed[i - 1].ly + LABEL_GAP);
    }
    const overflow = placed.length ? placed[placed.length - 1].ly - (H - M.bottom) : 0;
    if (overflow > 0) placed.forEach((p) => (p.ly -= overflow));
    return placed;
  }, [series, x, y]);

  function path(points: ChartPoint[], stepped: boolean): string {
    return points
      .map((p, i) => {
        const px = x(p.date).toFixed(1);
        const py = y(p.value).toFixed(1);
        if (i === 0) return `M${px},${py}`;
        return stepped ? `H${px}V${py}` : `L${px},${py}`;
      })
      .join("");
  }

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    dates.forEach((d, i) => {
      if (Math.abs(x(d) - px) < Math.abs(x(dates[best]) - px)) best = i;
    });
    setHover(best);
  }

  const hoverDate = hover === null ? null : dates[hover];
  const rows =
    hoverDate === null
      ? []
      : series.flatMap((s) => {
          // Step series hold their last value until the next change.
          const exact = s.points.find((p) => p.date === hoverDate);
          const held =
            step && s.kind === "actual"
              ? [...s.points].reverse().find((p) => p.date <= hoverDate)
              : undefined;
          const p = exact ?? held;
          return p ? [{ s, p }] : [];
        });

  return (
    <figure className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none select-none"
        role="img"
        aria-label={ariaLabel}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {splitDate && (
          <g>
            <rect
              x={x(splitDate)}
              y={M.top}
              width={W - M.right - x(splitDate)}
              height={H - M.top - M.bottom}
              rx={4}
              className="fill-forecast-zone"
            />
            <text
              x={x(splitDate) + 6}
              y={M.top + 14}
              className="fill-muted text-[11px]"
            >
              Forecast
            </text>
          </g>
        )}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={M.left}
              x2={W - M.right}
              y1={y(t)}
              y2={y(t)}
              className="stroke-line"
              strokeWidth={1}
            />
            <text
              x={M.left - 8}
              y={y(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-muted text-[11px] tabular-nums"
            >
              {t}
              {unit}
            </text>
          </g>
        ))}
        {xTicks.map((d) => (
          <text
            key={d}
            x={x(d)}
            y={H - M.bottom + 18}
            textAnchor="middle"
            className="fill-muted text-[11px]"
          >
            {d.slice(0, 4)}
          </text>
        ))}

        {series.map((s) => (
          <g key={s.id}>
            {s.kind === "forecast" &&
              s.points
                .slice(1)
                .map((p) =>
                  p.lower != null && p.upper != null ? (
                    <line
                      key={p.date}
                      x1={x(p.date)}
                      x2={x(p.date)}
                      y1={y(p.lower)}
                      y2={y(p.upper)}
                      className={TONES[s.tone].stroke}
                      strokeWidth={6}
                      strokeOpacity={0.3}
                      strokeLinecap="round"
                    />
                  ) : null,
                )}
            <path
              d={path(s.points, step && s.kind === "actual")}
              fill="none"
              className={TONES[s.tone].stroke}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.kind === "forecast" &&
              s.points
                .slice(1)
                .map((p) => (
                  <circle
                    key={p.date}
                    cx={x(p.date)}
                    cy={y(p.value)}
                    r={4.5}
                    className={`${TONES[s.tone].fill} stroke-background`}
                    strokeWidth={2}
                  />
                ))}
          </g>
        ))}

        {labels.map((l) => (
          <g key={l.id}>
            {Math.abs(l.ly - l.ey) > 1 && (
              <line
                x1={l.ex + 6}
                y1={l.ey}
                x2={l.ex + 14}
                y2={l.ly}
                className="stroke-muted"
                strokeWidth={1}
              />
            )}
            <line
              x1={l.ex + 16}
              x2={l.ex + 26}
              y1={l.ly}
              y2={l.ly}
              className={TONES[l.tone].stroke}
              strokeWidth={2}
              strokeLinecap="round"
            />
            <text
              x={l.ex + 30}
              y={l.ly}
              dominantBaseline="middle"
              className="fill-foreground text-[11px]"
            >
              {l.label}
            </text>
          </g>
        ))}

        {hoverDate && (
          <line
            x1={x(hoverDate)}
            x2={x(hoverDate)}
            y1={M.top}
            y2={H - M.bottom}
            className="stroke-muted"
            strokeWidth={1}
          />
        )}
      </svg>

      {hoverDate && rows.length > 0 && (
        <div
          className="pointer-events-none absolute top-2 rounded-md border border-line bg-panel px-3 py-2 text-xs shadow-sm"
          style={
            x(hoverDate) > W / 2
              ? { right: `${((W - x(hoverDate)) / W) * 100 + 2}%` }
              : { left: `${(x(hoverDate) / W) * 100 + 2}%` }
          }
        >
          <p className="mb-1 font-medium">{monthYear(hoverDate)}</p>
          {rows.map(({ s, p }) => (
            <p key={s.id} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className={`inline-block h-0.5 w-3 rounded ${TONES[s.tone].bg}`}
                aria-hidden
              />
              <strong className="tabular-nums">
                {p.value.toFixed(2)}
                {unit}
              </strong>
              <span className="text-muted">
                {s.label}
                {p.lower != null && p.upper != null && s.kind === "forecast"
                  ? ` (${p.lower.toFixed(1)}–${p.upper.toFixed(1)})`
                  : ""}
              </span>
            </p>
          ))}
        </div>
      )}
    </figure>
  );
}

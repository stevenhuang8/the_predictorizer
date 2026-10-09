"use client";

import { useMemo, useRef, useState } from "react";
import { type Tone, TONES } from "./tones";

export type HorizonSeries = {
  id: string;
  label: string;
  tone: Tone;
  points: { horizon: number; value: number }[];
};

const W = 420;
const H = 240;
const M = { top: 14, right: 132, bottom: 32, left: 36 };
const LABEL_GAP = 14;

function niceStep(max: number, count = 4): number {
  const raw = max / count || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
}

/**
 * How a score (RMSE) changes with the forecast horizon, one line per model in
 * its color, from a zero baseline. End labels name each line (with leader
 * lines where they'd collide); hover snaps to a horizon and lists every model
 * there, best first.
 */
export function HorizonChart({
  series,
  title,
  digits = 2,
}: {
  series: HorizonSeries[];
  title: string;
  digits?: number;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const { x, y, ticks, horizons } = useMemo(() => {
    const horizons = [
      ...new Set(series.flatMap((s) => s.points.map((p) => p.horizon))),
    ].sort((a, b) => a - b);
    const max = Math.max(...series.flatMap((s) => s.points.map((p) => p.value)), 0);
    const step = niceStep(max);
    const top = Math.ceil((max * 1.05) / step) * step || step;
    const ticks: number[] = [];
    for (let t = 0; t <= top + 1e-9; t += step) ticks.push(Number(t.toFixed(6)));
    const h0 = horizons[0] ?? 0;
    const h1 = horizons[horizons.length - 1] ?? 1;
    const x = (h: number) =>
      M.left + 12 + ((h - h0) / (h1 - h0 || 1)) * (W - M.left - M.right - 24);
    const y = (v: number) => H - M.bottom - (v / top) * (H - M.top - M.bottom);
    return { x, y, ticks, horizons };
  }, [series]);

  const labels = useMemo(() => {
    const ends = series
      .filter((s) => s.points.length)
      .map((s) => {
        const last = s.points[s.points.length - 1];
        return { ...s, ex: x(last.horizon), ey: y(last.value) };
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

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || !horizons.length) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    const nearest = horizons.reduce((best, h) =>
      Math.abs(x(h) - px) < Math.abs(x(best) - px) ? h : best,
    );
    setHover(nearest);
  }

  const rows =
    hover === null
      ? []
      : series
          .flatMap((s) => {
            const p = s.points.find((q) => q.horizon === hover);
            return p ? [{ s, value: p.value }] : [];
          })
          .sort((a, b) => a.value - b.value);

  return (
    <figure className="relative">
      <figcaption className="mb-1 text-sm font-medium">{title}</figcaption>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none select-none"
        role="img"
        aria-label={`${title}: ${series
          .map(
            (s) =>
              `${s.label} ${s.points.map((p) => `${p.horizon} mo ${p.value.toFixed(digits)}`).join(", ")}`,
          )
          .join("; ")}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
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
              x={M.left - 6}
              y={y(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-muted text-[11px] tabular-nums"
            >
              {t}
            </text>
          </g>
        ))}
        {horizons.map((h) => (
          <text
            key={h}
            x={x(h)}
            y={H - M.bottom + 18}
            textAnchor="middle"
            className="fill-muted text-[11px]"
          >
            {h} mo
          </text>
        ))}

        {hover !== null && (
          <line
            x1={x(hover)}
            x2={x(hover)}
            y1={M.top}
            y2={H - M.bottom}
            className="stroke-muted"
            strokeWidth={1}
          />
        )}

        {series.map((s) => (
          <g key={s.id}>
            <path
              d={s.points
                .map(
                  (p, i) =>
                    `${i ? "L" : "M"}${x(p.horizon).toFixed(1)},${y(p.value).toFixed(1)}`,
                )
                .join("")}
              fill="none"
              className={TONES[s.tone].stroke}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.points.map((p) => (
              <circle
                key={p.horizon}
                cx={x(p.horizon)}
                cy={y(p.value)}
                r={hover === p.horizon ? 5.5 : 4}
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
              x2={l.ex + 24}
              y1={l.ly}
              y2={l.ly}
              className={TONES[l.tone].stroke}
              strokeWidth={2}
              strokeLinecap="round"
            />
            <text
              x={l.ex + 28}
              y={l.ly}
              dominantBaseline="middle"
              className="fill-foreground text-[11px]"
            >
              {l.label}
            </text>
          </g>
        ))}
      </svg>

      {hover !== null && rows.length > 0 && (
        <div
          className="pointer-events-none absolute top-6 rounded-md border border-line bg-panel px-3 py-2 text-xs shadow-sm"
          style={
            x(hover) > (W - M.right) / 2
              ? { right: `${((W - x(hover)) / W) * 100 + 2}%` }
              : { left: `${(x(hover) / W) * 100 + 2}%` }
          }
        >
          <p className="mb-1 font-medium">{hover} months ahead</p>
          {rows.map(({ s, value }) => (
            <p key={s.id} className="flex items-center gap-2 whitespace-nowrap">
              <span
                className={`inline-block h-0.5 w-3 rounded ${TONES[s.tone].bg}`}
                aria-hidden
              />
              <strong className="tabular-nums">{value.toFixed(digits)}</strong>
              <span className="text-muted">{s.label}</span>
            </p>
          ))}
        </div>
      )}
    </figure>
  );
}

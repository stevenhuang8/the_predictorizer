/**
 * Chart colors as literal Tailwind classes. They must stay literal strings so
 * Tailwind's scanner generates them (see globals.css): never build them from
 * parts like `stroke-${name}`.
 */
export const TONES = {
  ink: { stroke: "stroke-foreground", fill: "fill-foreground", bg: "bg-foreground" },
  lightgbm: {
    stroke: "stroke-series-lightgbm",
    fill: "fill-series-lightgbm",
    bg: "bg-series-lightgbm",
  },
  random_walk: {
    stroke: "stroke-series-random-walk",
    fill: "fill-series-random-walk",
    bg: "bg-series-random-walk",
  },
  arima: {
    stroke: "stroke-series-arima",
    fill: "fill-series-arima",
    bg: "bg-series-arima",
  },
  yellow: {
    stroke: "stroke-series-yellow",
    fill: "fill-series-yellow",
    bg: "bg-series-yellow",
  },
  magenta: {
    stroke: "stroke-series-magenta",
    fill: "fill-series-magenta",
    bg: "bg-series-magenta",
  },
} as const;

export type Tone = keyof typeof TONES;

export const DECISION_BG = {
  cut: "bg-decision-cut",
  hold: "bg-decision-hold",
  hike: "bg-decision-hike",
} as const;

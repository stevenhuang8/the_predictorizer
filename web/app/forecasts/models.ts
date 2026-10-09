import type { Tone } from "../_components/tones";

/** How each model is named, explained and colored on the dashboard. */
export const MODELS: Record<string, { name: string; blurb: string; tone: Tone }> = {
  lightgbm: {
    name: "LightGBM",
    blurb:
      "Machine learning on about 40 indicators: prices, jobless claims, the yield curve and more.",
    tone: "lightgbm",
  },
  random_walk: {
    name: "No change",
    blurb:
      "Assumes the latest value simply holds. The baseline every other model has to beat.",
    tone: "random_walk",
  },
  arima: {
    name: "ARIMA",
    blurb:
      "A statistical model of the series' own trend and momentum. The most accurate on CPI in our 2020-2023 backtest.",
    tone: "arima",
  },
  fomc_lightgbm: {
    name: "FOMC model",
    blurb:
      "Machine learning on rates, markets and the economy. Most accurate in the final week before a meeting.",
    tone: "lightgbm",
  },
  fomc_persistence: {
    name: "Repeat last decision",
    blurb: "How often each decision followed the previous one in the past.",
    tone: "random_walk",
  },
  // Baselines that only appear in the backtest.
  ets: {
    name: "Exponential smoothing",
    blurb: "Tracks the level, trend and seasonality, weighting recent months most.",
    tone: "yellow",
  },
  historical_mean: {
    name: "Long-run average",
    blurb: "Predicts the average of all past values.",
    tone: "magenta",
  },
  persistence: {
    name: "Repeat last decision",
    blurb: "How often each decision followed the previous one in the past.",
    tone: "random_walk",
  },
  climatology: {
    name: "Historical frequencies",
    blurb: "How often the Fed cut, held or hiked in the past, whatever came before.",
    tone: "yellow",
  },
  always_hold: {
    name: "Always hold",
    blurb: "Puts all its weight on no change at every meeting.",
    tone: "magenta",
  },
};

/** Fixed display order (and so color order) of the models. */
export const NUMERIC_ORDER = ["lightgbm", "arima", "random_walk"];
export const FOMC_ORDER = ["fomc_lightgbm", "fomc_persistence"];

export function modelName(id: string): string {
  return MODELS[id]?.name ?? id;
}

/** A model's color; ink for any model without one. */
export function modelTone(id: string): Tone {
  return MODELS[id]?.tone ?? "ink";
}

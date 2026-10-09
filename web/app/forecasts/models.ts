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
};

/** Fixed display order (and so color order) of the models. */
export const NUMERIC_ORDER = ["lightgbm", "arima", "random_walk"];
export const FOMC_ORDER = ["fomc_lightgbm", "fomc_persistence"];

/** Models that only appear in the backtest report. */
const BACKTEST_NAMES: Record<string, string> = {
  ets: "Exponential smoothing",
  historical_mean: "Long-run average",
  persistence: "Repeat last decision",
  climatology: "Historical frequencies",
  always_hold: "Always hold",
};

export function modelName(id: string): string {
  return MODELS[id]?.name ?? BACKTEST_NAMES[id] ?? id;
}

import type { Tone } from "../_components/tones";
import type { QuestionKind, SeriesKind } from "@/lib/ai-dashboard-data";

/**
 * How each AI model is named, explained and colored. Statistical models keep
 * their economy colors; models that never share a chart share a slot (ETS and
 * the market price are yellow, the linear trend and the base rate magenta,
 * LightGBM and the threshold classifier blue). The LLM is violet (slot 7).
 */
export const AI_MODELS: Record<string, { name: string; blurb: string; tone: Tone }> = {
  claude_llm: {
    name: "Claude (LLM)",
    blurb:
      "Claude reads the recent numbers, the baselines and dated research papers, model cards and markets, and explains its forecast.",
    tone: "violet",
  },
  ai_lightgbm: {
    name: "LightGBM",
    blurb:
      "Machine learning on growth rates, benchmark records and cross-series signals.",
    tone: "lightgbm",
  },
  ai_arima: {
    name: "ARIMA",
    blurb: "A statistical model of the series' own trend, on a log or headroom scale.",
    tone: "arima",
  },
  ai_ets: {
    name: "ETS",
    blurb: "Tracks level, trend and seasonality, weighting recent months most.",
    tone: "yellow",
  },
  ai_linear_trend: {
    name: "Linear trend",
    blurb: "Extends the last year's trend (constant growth for counts).",
    tone: "magenta",
  },
  ai_random_walk: {
    name: "No change",
    blurb:
      "Assumes the latest value holds. The baseline every other model has to beat.",
    tone: "random_walk",
  },
  ai_market_price: {
    name: "Polymarket price",
    blurb: "What the prediction market said the day before the question was asked.",
    tone: "yellow",
  },
  ai_base_rate: {
    name: "Base rate",
    blurb:
      "How often this kind of event happened before: similar markets, or the series clearing the same step.",
    tone: "magenta",
  },
  ai_threshold_classifier: {
    name: "Threshold classifier",
    blurb:
      "LightGBM trained on every past month: did the series clear a step like this one?",
    tone: "lightgbm",
  },
};

export const NUMERIC_MODEL_ORDER = [
  "claude_llm",
  "ai_lightgbm",
  "ai_arima",
  "ai_ets",
  "ai_linear_trend",
  "ai_random_walk",
];
export const BINARY_MODEL_ORDER = [
  "claude_llm",
  "ai_market_price",
  "ai_threshold_classifier",
  "ai_base_rate",
];

/** Short display names; the series' full definitions stay in their subtitles. */
export const SERIES_SHORT: Record<string, string> = {
  ARXIV_AI_SUBMISSIONS: "arXiv AI papers",
  HF_NEW_MODELS: "New Hugging Face models",
  GITHUB_AI_ACTIVITY: "New GitHub AI repos",
  EPOCH_GPQA_SOTA: "GPQA Diamond record",
  EPOCH_SWEBENCH_SOTA: "SWE-bench Verified record",
  EPOCH_ARC2_SOTA: "ARC-AGI-2 record",
  EPOCH_LARGEST_FLOPS: "Largest training run",
};

export function seriesKindOfKey(key: string): SeriesKind {
  const series = key.split(">=")[0];
  if (series.endsWith("_SOTA")) return "percent";
  if (series === "EPOCH_LARGEST_FLOPS") return "log10";
  return "count";
}

/**
 * A short label for a numeric or threshold question, from its key
 * ('ARXIV_AI_SUBMISSIONS' or 'ARXIV_AI_SUBMISSIONS>=12400'): "arXiv AI papers,
 * Nov 2026" or "arXiv AI papers ≥ 12,400 in Nov 2026". Curated questions keep
 * their own text.
 */
export function questionLabel(key: string, text: string, month: string): string {
  if (key.startsWith("polymarket:")) return text;
  const [series, threshold] = key.split(">=");
  const name = SERIES_SHORT[series] ?? series;
  if (threshold === undefined) return `${name}, ${month}`;
  return `${name} ≥ ${formatValue(Number(threshold), seriesKindOfKey(key))} in ${month}`;
}

export function aiModelName(id: string): string {
  return AI_MODELS[id]?.name ?? id;
}

export function aiModelTone(id: string): Tone {
  return AI_MODELS[id]?.tone ?? "ink";
}

export const KIND_LABELS: Record<QuestionKind, { title: string; blurb: string }> = {
  numeric: {
    title: "Monthly numbers",
    blurb:
      "arXiv papers, Hugging Face models, GitHub repos, benchmark records and training compute.",
  },
  threshold: {
    title: "Will it clear the bar?",
    blurb:
      "Yes/no versions of the monthly numbers: at least 10%, 25% or 50% above the latest.",
  },
  curated: {
    title: "Polymarket questions",
    blurb: "Releases, benchmark records and model rankings, mirrored from Polymarket.",
  },
};

/** A value in its series' units: a count, a percentage, or log10 FLOP. */
export function formatValue(value: number, kind: SeriesKind): string {
  if (kind === "percent") return `${value.toFixed(1)}%`;
  if (kind === "log10") return value.toFixed(2);
  return Math.round(value).toLocaleString("en-US");
}

export function unitNote(kind: SeriesKind): string {
  if (kind === "log10") return "log10 FLOP (27 = 10²⁷ FLOP)";
  if (kind === "percent") return "best score so far, %";
  return "per month";
}

export function probability(p: number): string {
  return `${(p * 100).toFixed(0)}%`;
}

export function usd(value: number): string {
  return value < 1 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`;
}

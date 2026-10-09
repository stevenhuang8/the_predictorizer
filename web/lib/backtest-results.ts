import "server-only";

import { readFileSync } from "node:fs";
import path from "node:path";

import type { NumericTarget } from "./dashboard-data";

/**
 * The walk-forward backtest report written by `uv run eco-forecast backtest`
 * (results/model_comparison.json, at the repo root). Only the fields the
 * track-record page uses are typed.
 */
export type BacktestResults = {
  settings: { start: string; end: string; horizons: number[] };
  success_criterion: {
    criterion: string;
    lightgbm_rmse: number;
    random_walk_rmse: number;
    dm_p: number;
    passed: boolean;
  };
  findings: string[];
  metrics: {
    target: NumericTarget;
    horizon: number;
    model: string;
    n: number;
    rmse: number;
    mae: number;
    bias: number;
    coverage: number | null;
    rmse_vs_random_walk: number | null;
    dm_p: number | null;
  }[];
  calibration: {
    target: NumericTarget;
    model: string;
    nominal: number;
    empirical: number;
  }[];
  fomc: {
    model: string;
    n_resolved: number;
    brier: number;
    accuracy: number;
    n_changes: number;
    brier_on_changes: number;
  }[];
};

const RESULTS = path.join(process.cwd(), "..", "results", "model_comparison.json");

/**
 * The backtest report, or null if it hasn't been generated. A synchronous
 * read of a file that only changes when the backtest is rerun, so the section
 * is prerendered with the page.
 */
export function loadBacktest(): BacktestResults | null {
  try {
    return JSON.parse(readFileSync(RESULTS, "utf-8")) as BacktestResults;
  } catch {
    return null;
  }
}

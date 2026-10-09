import "server-only";

import { db } from "./db";
import type { NumericTarget } from "./dashboard-data";

/** Live accuracy of one model at one horizon, against the first release. */
export type NumericScore = {
  target: NumericTarget;
  horizonMonths: number;
  model: string;
  n: number;
  rmse: number;
  mae: number;
  bias: number;
  /** Share of actuals inside the 80% interval; null without intervals. */
  coverage: number | null;
  /** RMSE against the latest revised value (diagnostic only). */
  rmseLatest: number | null;
};

export type FomcScore = {
  horizonMonths: number | null;
  leadDays: number | null;
  model: string;
  n: number;
  brier: number;
  /** Share of meetings where the outcome got the highest probability. */
  hitRate: number;
};

export type MissCategory =
  "expected_variance" | "bad_data" | "regime_change" | "bad_model";

export type Miss = {
  forecastId: number;
  target: string;
  targetDate: string;
  model: string;
  category: MissCategory;
  notes: string | null;
  manual: boolean;
};

export type Resolved = {
  forecastId: number;
  target: NumericTarget;
  targetDate: string;
  horizonMonths: number;
  model: string;
  prediction: number;
  lower: number | null;
  upper: number | null;
  actual: number;
  latest: number | null;
  inInterval: boolean | null;
  category: MissCategory | null;
};

export type TrackRecord = {
  scored: number;
  /** Open live questions, and the earliest month one of them is about. */
  open: number;
  firstOpen: string | null;
  numeric: NumericScore[];
  fomc: FomcScore[];
  categories: Partial<Record<MissCategory, number>>;
  misses: Miss[];
  recent: Resolved[];
};

const RECENT = 15;
const MISSES = 10;

/** Scores, post-mortems and recent resolutions of live (not backtest) forecasts. */
export async function loadTrackRecord(): Promise<TrackRecord> {
  const scored = `NOT f.is_backtest AND f.scored_at IS NOT NULL`;
  const joins = `
    FROM forecasts f
    JOIN questions q ON q.id = f.question_id
    JOIN resolutions r ON r.question_id = q.id
    JOIN model_versions m ON m.id = f.model_version_id`;

  const [counts, numeric, fomc, categories, misses, recent] = await Promise.all([
    db().query<{ scored: number; open: number; first_open: string | null }>(
      `SELECT
         (SELECT count(*)::int FROM forecasts f WHERE ${scored}) AS scored,
         count(DISTINCT q.id)::int AS open,
         min(q.target_date) AS first_open
       FROM questions q
       JOIN forecasts f ON f.question_id = q.id AND NOT f.is_backtest
       LEFT JOIN resolutions r ON r.question_id = q.id
       WHERE r.id IS NULL`,
    ),
    db().query<{
      target: NumericTarget;
      horizon_months: number;
      model: string;
      n: number;
      rmse: number;
      mae: number;
      bias: number;
      coverage: number | null;
      rmse_latest: number | null;
    }>(
      `SELECT q.target::text AS target, q.horizon_months, m.model_type AS model,
              count(*)::int AS n,
              sqrt(avg(f.error ^ 2))::float8 AS rmse,
              avg(abs(f.error))::float8 AS mae,
              avg(f.error)::float8 AS bias,
              avg(f.in_interval::int)::float8 AS coverage,
              sqrt(avg((f.prediction - r.latest_value) ^ 2))::float8 AS rmse_latest
       ${joins}
       WHERE ${scored} AND q.target <> 'fomc_decision'
       GROUP BY 1, 2, 3
       ORDER BY 1, 2, 3`,
    ),
    db().query<{
      horizon_months: number | null;
      lead_days: number | null;
      model: string;
      n: number;
      brier: number;
      hit_rate: number;
    }>(
      `SELECT q.horizon_months, q.lead_days, m.model_type AS model,
              count(*)::int AS n,
              avg(f.brier_score)::float8 AS brier,
              avg(CASE WHEN coalesce((f.probabilities ->> r.actual_outcome)::float8, 0)
                         >= greatest((f.probabilities ->> 'cut')::float8,
                                     (f.probabilities ->> 'hold')::float8,
                                     (f.probabilities ->> 'hike')::float8)
                       THEN 1 ELSE 0 END)::float8 AS hit_rate
       ${joins}
       WHERE ${scored} AND q.target = 'fomc_decision'
       GROUP BY 1, 2, 3
       ORDER BY q.lead_days NULLS LAST, q.horizon_months, m.model_type`,
    ),
    db().query<{ category: MissCategory; n: number }>(
      `SELECT p.miss_category::text AS category, count(*)::int AS n
       FROM postmortems p JOIN forecasts f ON f.id = p.forecast_id
       WHERE NOT f.is_backtest
       GROUP BY 1`,
    ),
    db().query<{
      forecast_id: number;
      target: string;
      target_date: string;
      model: string;
      category: MissCategory;
      notes: string | null;
      classified_by: string;
    }>(
      `SELECT f.id AS forecast_id, q.target::text AS target, q.target_date,
              m.model_type AS model, p.miss_category::text AS category, p.notes,
              p.classified_by
       FROM postmortems p
       JOIN forecasts f ON f.id = p.forecast_id
       JOIN questions q ON q.id = f.question_id
       JOIN model_versions m ON m.id = f.model_version_id
       WHERE NOT f.is_backtest AND p.miss_category <> 'expected_variance'
       ORDER BY q.target_date DESC, q.target, m.model_type
       LIMIT ${MISSES}`,
    ),
    db().query<{
      forecast_id: number;
      target: NumericTarget;
      target_date: string;
      horizon_months: number;
      model: string;
      prediction: number;
      lower: number | null;
      upper: number | null;
      actual: number;
      latest: number | null;
      in_interval: boolean | null;
      category: MissCategory | null;
    }>(
      `SELECT f.id AS forecast_id, q.target::text AS target, q.target_date,
              q.horizon_months, m.model_type AS model,
              f.prediction::float8 AS prediction,
              f.interval_lower::float8 AS lower, f.interval_upper::float8 AS upper,
              r.actual_value::float8 AS actual, r.latest_value::float8 AS latest,
              f.in_interval, p.miss_category::text AS category
       ${joins}
       LEFT JOIN postmortems p ON p.forecast_id = f.id
       WHERE ${scored} AND q.target <> 'fomc_decision'
       ORDER BY r.actual_as_of DESC, q.target_date DESC, q.target, q.horizon_months,
                m.model_type
       LIMIT ${RECENT}`,
    ),
  ]);

  const c = counts.rows[0];
  return {
    scored: c?.scored ?? 0,
    open: c?.open ?? 0,
    firstOpen: c?.first_open ?? null,
    numeric: numeric.rows.map((r) => ({
      target: r.target,
      horizonMonths: r.horizon_months,
      model: r.model,
      n: r.n,
      rmse: r.rmse,
      mae: r.mae,
      bias: r.bias,
      coverage: r.coverage,
      rmseLatest: r.rmse_latest,
    })),
    fomc: fomc.rows.map((r) => ({
      horizonMonths: r.horizon_months,
      leadDays: r.lead_days,
      model: r.model,
      n: r.n,
      brier: r.brier,
      hitRate: r.hit_rate,
    })),
    categories: Object.fromEntries(categories.rows.map((r) => [r.category, r.n])),
    misses: misses.rows.map((r) => ({
      forecastId: r.forecast_id,
      target: r.target,
      targetDate: r.target_date,
      model: r.model,
      category: r.category,
      notes: r.notes,
      manual: r.classified_by === "manual",
    })),
    recent: recent.rows.map((r) => ({
      forecastId: r.forecast_id,
      target: r.target,
      targetDate: r.target_date,
      horizonMonths: r.horizon_months,
      model: r.model,
      prediction: r.prediction,
      lower: r.lower,
      upper: r.upper,
      actual: r.actual,
      latest: r.latest,
      inInterval: r.in_interval,
      category: r.category,
    })),
  };
}

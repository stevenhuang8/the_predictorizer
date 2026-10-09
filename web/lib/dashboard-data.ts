import "server-only";

import { db } from "./db";

export type NumericTarget = "cpi_yoy" | "unemployment";
export type Outcome = "cut" | "hold" | "hike";

export type Point = { date: string; value: number };

export type NumericForecast = {
  questionId: number;
  target: NumericTarget;
  targetDate: string;
  horizonMonths: number;
  model: string;
  forecastDate: string;
  point: number;
  lower: number | null;
  upper: number | null;
};

export type FomcForecast = {
  questionId: number;
  meeting: string;
  horizonMonths: number | null;
  leadDays: number | null;
  model: string;
  forecastDate: string;
  probabilities: Record<Outcome, number>;
};

export type DashboardData = {
  dataThrough: string | null;
  lastRun: string | null;
  numeric: NumericForecast[];
  fomc: FomcForecast[];
  history: Record<NumericTarget, Point[]>;
  policyRate: Point[];
};

/** Months of history shown before the forecasts. */
const HISTORY_MONTHS = 36;

function monthsBefore(date: Date, months: number): string {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() - months, 1));
  return d.toISOString().slice(0, 10);
}

/** The latest vintage of each period of a series, from `since` on. */
async function latestValues(seriesId: string, since: string): Promise<Point[]> {
  const { rows } = await db().query<{ date: string; value: number | null }>(
    `SELECT DISTINCT ON (o.observed_at) o.observed_at AS date, o.value::float8 AS value
     FROM observations o JOIN series s ON s.id = o.series_id
     WHERE s.series_id = $1 AND o.observed_at >= $2
     ORDER BY o.observed_at, o.as_of DESC`,
    [seriesId, since],
  );
  return rows.filter((r): r is Point => r.value !== null);
}

/** CPI year-over-year % change from the index levels. */
function yoy(levels: Point[]): Point[] {
  const byDate = new Map(levels.map((p) => [p.date, p.value]));
  return levels.flatMap((p) => {
    const d = new Date(`${p.date}T00:00:00Z`);
    const yearAgo = new Date(Date.UTC(d.getUTCFullYear() - 1, d.getUTCMonth(), 1));
    const base = byDate.get(yearAgo.toISOString().slice(0, 10));
    return base ? [{ date: p.date, value: (p.value / base - 1) * 100 }] : [];
  });
}

/** A daily series reduced to the days it changed, plus its latest day. */
function changes(points: Point[]): Point[] {
  return points.filter(
    (p, i) => i === 0 || i === points.length - 1 || p.value !== points[i - 1].value,
  );
}

export async function loadDashboard(today: Date): Promise<DashboardData> {
  const since = monthsBefore(today, HISTORY_MONTHS);
  const cpiSince = monthsBefore(today, HISTORY_MONTHS + 12);

  // Each open question's live forecasts from its most recent run.
  const forecasts = db().query<{
    question_id: number;
    target: string;
    target_date: string;
    horizon_months: number | null;
    lead_days: number | null;
    model_type: string;
    forecast_date: string;
    prediction: number | null;
    interval_lower: number | null;
    interval_upper: number | null;
    probabilities: Record<Outcome, number> | null;
  }>(
    `SELECT q.id AS question_id, q.target::text AS target, q.target_date,
            q.horizon_months, q.lead_days, m.model_type, f.forecast_date,
            f.prediction::float8 AS prediction, f.interval_lower::float8 AS interval_lower,
            f.interval_upper::float8 AS interval_upper, f.probabilities
     FROM forecasts f
     JOIN questions q ON q.id = f.question_id
     JOIN model_versions m ON m.id = f.model_version_id
     LEFT JOIN resolutions r ON r.question_id = q.id
     WHERE NOT f.is_backtest AND r.id IS NULL
       AND f.forecast_date = (
         SELECT max(f2.forecast_date) FROM forecasts f2
         WHERE f2.question_id = q.id AND NOT f2.is_backtest)
     ORDER BY q.target_date, q.target, m.model_type`,
  );
  const meta = db().query<{ data_through: string | null; last_run: string | null }>(
    `SELECT (SELECT max(as_of) FROM observations)::date AS data_through,
            (SELECT max(forecast_date) FROM forecasts WHERE NOT is_backtest) AS last_run`,
  );

  const [rows, metaRows, cpi, unrate, fedUpper] = await Promise.all([
    forecasts,
    meta,
    latestValues("CPIAUCSL", cpiSince),
    latestValues("UNRATE", since),
    latestValues("DFEDTARU", since),
  ]);

  const numeric: NumericForecast[] = [];
  const fomc: FomcForecast[] = [];
  for (const r of rows.rows) {
    if (r.target === "fomc_decision" && r.probabilities) {
      fomc.push({
        questionId: r.question_id,
        meeting: r.target_date,
        horizonMonths: r.horizon_months,
        leadDays: r.lead_days,
        model: r.model_type,
        forecastDate: r.forecast_date,
        probabilities: r.probabilities,
      });
    } else if (r.prediction !== null && r.horizon_months !== null) {
      numeric.push({
        questionId: r.question_id,
        target: r.target as NumericTarget,
        targetDate: r.target_date,
        horizonMonths: r.horizon_months,
        model: r.model_type,
        forecastDate: r.forecast_date,
        point: r.prediction,
        lower: r.interval_lower,
        upper: r.interval_upper,
      });
    }
  }

  return {
    dataThrough: metaRows.rows[0]?.data_through ?? null,
    lastRun: metaRows.rows[0]?.last_run ?? null,
    numeric,
    fomc,
    history: {
      cpi_yoy: yoy(cpi).filter((p) => p.date >= since),
      unemployment: unrate,
    },
    policyRate: changes(
      fedUpper.map((p) => ({ date: p.date, value: p.value - 0.125 })),
    ),
  };
}

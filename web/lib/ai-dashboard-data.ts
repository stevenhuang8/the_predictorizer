import "server-only";

import { db } from "./db";

/**
 * Data for the AI section (AI task 24): AI-progress questions, their live
 * forecasts (statistical models, LightGBM, the market, base rates and, when it
 * has run, the Claude forecaster), the track record and LLM costs. Every query
 * is limited to `questions.domain = 'ai'`; the economy pages use
 * dashboard-data.ts and track-record-data.ts.
 */

export type SeriesKind = "count" | "percent" | "log10";
export type QuestionKind = "numeric" | "threshold" | "curated";
export type Point = { date: string; value: number };

export type ModelForecast = {
  model: string;
  forecastDate: string;
  isBacktest: boolean;
  point: number | null;
  lower: number | null;
  upper: number | null;
  pYes: number | null;
};

export type AISeries = {
  seriesId: string;
  name: string;
  units: string;
  kind: SeriesKind;
  history: Point[];
  /** Open live numeric forecasts: one per model and target month. */
  forecasts: (ModelForecast & { targetDate: string; horizonMonths: number })[];
};

export type LLMCosts = {
  calls: number;
  failed: number;
  spentUsd: number;
  forecasts: number;
  byModel: { model: string; calls: number; spentUsd: number }[];
};

export type AIOverview = {
  lastRun: string | null;
  dataThrough: string | null;
  open: Record<QuestionKind, number>;
  liveForecasts: number;
  scored: number;
  llm: LLMCosts;
  series: AISeries[];
};

export type AIQuestionRow = {
  id: number;
  kind: QuestionKind;
  key: string;
  text: string;
  targetDate: string;
  horizonMonths: number | null;
  leadDays: number | null;
  url: string | null;
  forecasts: ModelForecast[];
};

export type AIForecastDetail = ModelForecast & {
  id: number;
  rationale: string | null;
  drivers: [string, number][];
  error: number | null;
  inInterval: boolean | null;
  brier: number | null;
  llmCostUsd: number | null;
  postmortem: {
    category: string;
    notes: string | null;
    draft: string | null;
    confirmed: boolean;
  } | null;
};

export type AIQuestionDetail = {
  id: number;
  kind: QuestionKind;
  key: string;
  text: string;
  rule: string | null;
  targetDate: string;
  horizonMonths: number | null;
  leadDays: number | null;
  url: string | null;
  seriesId: string | null;
  units: string | null;
  resolution: { value: number | null; outcome: string | null; asOf: string } | null;
  /** The series (numeric and threshold) or the market's daily price (curated). */
  history: Point[];
  forecasts: AIForecastDetail[];
};

export type AITrackRecord = {
  includeBacktest: boolean;
  scored: number;
  numeric: {
    seriesId: string;
    name: string;
    horizonMonths: number;
    model: string;
    n: number;
    relativeRmse: number;
    coverage: number | null;
  }[];
  binary: { kind: QuestionKind; model: string; n: number; brier: number }[];
  /** Brier on the curated questions both the model and the market answered. */
  vsMarket: { model: string; n: number; brier: number; marketBrier: number }[];
  calibration: {
    model: string;
    bins: { lower: number; predicted: number; observed: number; n: number }[];
  }[];
  categories: Record<string, number>;
  llm: LLMCosts;
};

const HISTORY_MONTHS = 36;
const KIND_SQL = `CASE WHEN q.question_type = 'numeric_interval' THEN 'numeric'
                       WHEN q.platform_question_id IS NOT NULL THEN 'curated'
                       ELSE 'threshold' END`;

export function seriesKind(units: string): SeriesKind {
  if (units === "Percent") return "percent";
  if (units.startsWith("log10")) return "log10";
  return "count";
}

/** The series a numeric or threshold question resolves from. */
export function seriesOfKey(key: string): string | null {
  if (key.startsWith("polymarket:")) return null;
  return key.split(">=")[0];
}

function monthsBefore(date: Date, months: number): string {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() - months, 1));
  return d.toISOString().slice(0, 10);
}

const FORECAST_COLUMNS = `
  m.model_type AS model, f.forecast_date, f.is_backtest,
  f.prediction::float8 AS point, f.interval_lower::float8 AS lower,
  f.interval_upper::float8 AS upper, (f.probabilities ->> 'yes')::float8 AS p_yes`;

type ForecastRow = {
  model: string;
  forecast_date: string;
  is_backtest: boolean;
  point: number | null;
  lower: number | null;
  upper: number | null;
  p_yes: number | null;
};

function forecast(r: ForecastRow): ModelForecast {
  return {
    model: r.model,
    forecastDate: r.forecast_date,
    isBacktest: r.is_backtest,
    point: r.point,
    lower: r.lower,
    upper: r.upper,
    pYes: r.p_yes,
  };
}

async function latestValues(seriesIds: string[], since: string) {
  const { rows } = await db().query<{ series_id: string; date: string; value: number }>(
    `SELECT DISTINCT ON (s.series_id, o.observed_at)
            s.series_id, o.observed_at AS date, o.value::float8 AS value
     FROM observations o JOIN series s ON s.id = o.series_id
     WHERE s.series_id = ANY($1) AND o.observed_at >= $2 AND o.value IS NOT NULL
     ORDER BY s.series_id, o.observed_at, o.as_of DESC`,
    [seriesIds, since],
  );
  return rows;
}

export async function loadLLMCosts(): Promise<LLMCosts> {
  const [totals, byModel] = await Promise.all([
    db().query<{ calls: number; failed: number; spent: number; forecasts: number }>(
      `SELECT count(*)::int AS calls, count(*) FILTER (WHERE error IS NOT NULL)::int AS failed,
              coalesce(sum(cost_usd), 0)::float8 AS spent,
              count(DISTINCT forecast_id)::int AS forecasts
       FROM llm_calls`,
    ),
    db().query<{ model: string; calls: number; spent: number }>(
      `SELECT model_id AS model, count(*)::int AS calls,
              coalesce(sum(cost_usd), 0)::float8 AS spent
       FROM llm_calls GROUP BY 1 ORDER BY 3 DESC`,
    ),
  ]);
  const t = totals.rows[0];
  return {
    calls: t?.calls ?? 0,
    failed: t?.failed ?? 0,
    spentUsd: t?.spent ?? 0,
    forecasts: t?.forecasts ?? 0,
    byModel: byModel.rows.map((r) => ({
      model: r.model,
      calls: r.calls,
      spentUsd: r.spent,
    })),
  };
}

export async function loadAIOverview(today: Date): Promise<AIOverview> {
  const [meta, open, seriesRows, forecasts, llm] = await Promise.all([
    db().query<{
      last_run: string | null;
      data_through: string | null;
      live: number;
      scored: number;
    }>(
      `SELECT (SELECT max(f.forecast_date) FROM forecasts f JOIN questions q ON q.id = f.question_id
               WHERE NOT f.is_backtest AND q.domain = 'ai') AS last_run,
              (SELECT max(o.as_of) FROM observations o JOIN series s ON s.id = o.series_id
               WHERE s.domain = 'ai')::date AS data_through,
              (SELECT count(*)::int FROM forecasts f JOIN questions q ON q.id = f.question_id
               WHERE NOT f.is_backtest AND q.domain = 'ai') AS live,
              (SELECT count(*)::int FROM forecasts f JOIN questions q ON q.id = f.question_id
               WHERE NOT f.is_backtest AND f.scored_at IS NOT NULL AND q.domain = 'ai') AS scored`,
    ),
    db().query<{ kind: QuestionKind; n: number }>(
      `SELECT ${KIND_SQL} AS kind, count(*)::int AS n
       FROM questions q LEFT JOIN resolutions r ON r.question_id = q.id
       WHERE q.domain = 'ai' AND r.id IS NULL AND q.target_date > $1
       GROUP BY 1`,
      [today.toISOString().slice(0, 10)],
    ),
    db().query<{ series_id: string; name: string; units: string }>(
      `SELECT DISTINCT s.series_id, s.name, s.units
       FROM series s JOIN questions q ON q.question_key = s.series_id
       WHERE s.domain = 'ai' AND q.question_type = 'numeric_interval'
       ORDER BY s.series_id`,
    ),
    db().query<
      ForecastRow & { series_id: string; target_date: string; horizon_months: number }
    >(
      `SELECT q.question_key AS series_id, q.target_date, q.horizon_months, ${FORECAST_COLUMNS}
       FROM forecasts f
       JOIN questions q ON q.id = f.question_id
       JOIN model_versions m ON m.id = f.model_version_id
       LEFT JOIN resolutions r ON r.question_id = q.id
       WHERE q.domain = 'ai' AND q.question_type = 'numeric_interval'
         AND NOT f.is_backtest AND r.id IS NULL
         AND f.forecast_date = (SELECT max(f2.forecast_date) FROM forecasts f2
                                WHERE f2.question_id = q.id AND NOT f2.is_backtest)
       ORDER BY q.question_key, q.target_date, m.model_type`,
    ),
    loadLLMCosts(),
  ]);
  const ids = seriesRows.rows.map((r) => r.series_id);
  const history = await latestValues(ids, monthsBefore(today, HISTORY_MONTHS));
  const counts = Object.fromEntries(open.rows.map((r) => [r.kind, r.n]));
  const m = meta.rows[0];
  return {
    lastRun: m?.last_run ?? null,
    dataThrough: m?.data_through ?? null,
    open: {
      numeric: counts.numeric ?? 0,
      threshold: counts.threshold ?? 0,
      curated: counts.curated ?? 0,
    },
    liveForecasts: m?.live ?? 0,
    scored: m?.scored ?? 0,
    llm,
    series: seriesRows.rows.map((s) => ({
      seriesId: s.series_id,
      name: s.name,
      units: s.units,
      kind: seriesKind(s.units),
      history: history
        .filter((h) => h.series_id === s.series_id)
        .map((h) => ({ date: h.date, value: h.value })),
      forecasts: forecasts.rows
        .filter((f) => f.series_id === s.series_id)
        .map((f) => ({
          ...forecast(f),
          targetDate: f.target_date,
          horizonMonths: f.horizon_months,
        })),
    })),
  };
}

/** Open AI questions (target still ahead, not resolved) with each model's latest live forecast. */
export async function loadAIQuestions(
  today: Date,
  kind?: QuestionKind,
  limit = 200,
): Promise<{ rows: AIQuestionRow[]; total: number }> {
  const { rows } = await db().query<{
    id: number;
    kind: QuestionKind;
    key: string;
    text: string | null;
    target_date: string;
    horizon_months: number | null;
    lead_days: number | null;
    url: string | null;
    total: number;
  }>(
    `SELECT q.id, ${KIND_SQL} AS kind, q.question_key AS key, q.question_text AS text,
            q.target_date, q.horizon_months, q.lead_days, p.url, count(*) OVER ()::int AS total
     FROM questions q
     LEFT JOIN resolutions r ON r.question_id = q.id
     LEFT JOIN platform_questions p ON p.id = q.platform_question_id
     WHERE q.domain = 'ai' AND r.id IS NULL AND q.target_date > $1
       AND ($2::text IS NULL OR ${KIND_SQL} = $2)
     ORDER BY q.target_date, q.id
     LIMIT $3`,
    [today.toISOString().slice(0, 10), kind ?? null, limit],
  );
  const ids = rows.map((r) => r.id);
  const forecasts = ids.length
    ? await db().query<ForecastRow & { question_id: number }>(
        `SELECT DISTINCT ON (f.question_id, m.model_type) f.question_id, ${FORECAST_COLUMNS}
         FROM forecasts f JOIN model_versions m ON m.id = f.model_version_id
         WHERE f.question_id = ANY($1) AND NOT f.is_backtest
         ORDER BY f.question_id, m.model_type, f.forecast_date DESC`,
        [ids],
      )
    : { rows: [] };
  return {
    total: rows[0]?.total ?? 0,
    rows: rows.map((r) => ({
      id: r.id,
      kind: r.kind,
      key: r.key,
      text: r.text ?? r.key,
      targetDate: r.target_date,
      horizonMonths: r.horizon_months,
      leadDays: r.lead_days,
      url: r.url,
      forecasts: forecasts.rows.filter((f) => f.question_id === r.id).map(forecast),
    })),
  };
}

/** One AI question with every forecast, rationale, drivers, score and post-mortem. */
export async function loadAIQuestion(id: number): Promise<AIQuestionDetail | null> {
  const q = await db().query<{
    id: number;
    kind: QuestionKind;
    key: string;
    text: string | null;
    rule: string | null;
    target_date: string;
    horizon_months: number | null;
    lead_days: number | null;
    url: string | null;
    platform_question_id: number | null;
    value: number | null;
    outcome: string | null;
    actual_as_of: string | null;
  }>(
    `SELECT q.id, ${KIND_SQL} AS kind, q.question_key AS key, q.question_text AS text,
            q.resolution_rule AS rule, q.target_date, q.horizon_months, q.lead_days, p.url,
            q.platform_question_id, r.actual_value::float8 AS value, r.actual_outcome AS outcome,
            r.actual_as_of
     FROM questions q
     LEFT JOIN resolutions r ON r.question_id = q.id
     LEFT JOIN platform_questions p ON p.id = q.platform_question_id
     WHERE q.id = $1 AND q.domain = 'ai'`,
    [id],
  );
  const row = q.rows[0];
  if (!row) return null;
  const seriesId = seriesOfKey(row.key);

  const [forecasts, history, units] = await Promise.all([
    db().query<
      ForecastRow & {
        id: number;
        rationale: string | null;
        shap_values: { contributions?: Record<string, number> } | null;
        error: number | null;
        in_interval: boolean | null;
        brier: number | null;
        llm_cost: number | null;
        category: string | null;
        notes: string | null;
        draft: string | null;
        confirmed: boolean | null;
      }
    >(
      `SELECT f.id, ${FORECAST_COLUMNS}, f.rationale, f.shap_values,
              f.error::float8 AS error, f.in_interval, f.brier_score::float8 AS brier,
              (SELECT sum(c.cost_usd)::float8 FROM llm_calls c WHERE c.forecast_id = f.id) AS llm_cost,
              pm.miss_category::text AS category, pm.notes,
              CASE WHEN pm.llm_drafted THEN pm.draft_note END AS draft,
              pm.human_confirmed AS confirmed
       FROM forecasts f
       JOIN model_versions m ON m.id = f.model_version_id
       LEFT JOIN postmortems pm ON pm.forecast_id = f.id
       WHERE f.question_id = $1
       ORDER BY f.is_backtest, f.forecast_date DESC, m.model_type`,
      [id],
    ),
    seriesId
      ? latestValues(
          [seriesId],
          monthsBefore(new Date(`${row.target_date}T00:00:00Z`), HISTORY_MONTHS),
        )
      : row.platform_question_id
        ? db()
            .query<{ date: string; value: number }>(
              `SELECT as_of AS date, (probability * 100)::float8 AS value
               FROM platform_probabilities WHERE question_id = $1 ORDER BY as_of`,
              [row.platform_question_id],
            )
            .then((r) => r.rows)
        : Promise.resolve([]),
    seriesId
      ? db().query<{ units: string }>(`SELECT units FROM series WHERE series_id = $1`, [
          seriesId,
        ])
      : Promise.resolve({ rows: [] as { units: string }[] }),
  ]);

  return {
    id: row.id,
    kind: row.kind,
    key: row.key,
    text: row.text ?? row.key,
    rule: row.rule,
    targetDate: row.target_date,
    horizonMonths: row.horizon_months,
    leadDays: row.lead_days,
    url: row.url,
    seriesId,
    units: units.rows[0]?.units ?? null,
    resolution: row.actual_as_of
      ? { value: row.value, outcome: row.outcome, asOf: row.actual_as_of }
      : null,
    history: history.map((h) => ({ date: h.date, value: h.value })),
    forecasts: forecasts.rows.map((f) => ({
      ...forecast(f),
      id: f.id,
      rationale: f.rationale,
      drivers: Object.entries(f.shap_values?.contributions ?? {})
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 5),
      error: f.error,
      inInterval: f.in_interval,
      brier: f.brier,
      llmCostUsd: f.llm_cost,
      postmortem: f.category
        ? {
            category: f.category,
            notes: f.notes,
            draft: f.draft,
            confirmed: !!f.confirmed,
          }
        : null,
    })),
  };
}

export async function loadAITrackRecord(
  includeBacktest = false,
): Promise<AITrackRecord> {
  const scope = `q.domain = 'ai' AND f.scored_at IS NOT NULL AND ($1 OR NOT f.is_backtest)`;
  const joins = `FROM forecasts f
    JOIN questions q ON q.id = f.question_id
    JOIN resolutions r ON r.question_id = q.id
    JOIN model_versions m ON m.id = f.model_version_id`;
  const [numeric, binary, vsMarket, calibration, categories, llm] = await Promise.all([
    db().query<{
      series_id: string;
      name: string;
      horizon_months: number;
      model: string;
      n: number;
      relative_rmse: number;
      coverage: number | null;
    }>(
      `SELECT q.question_key AS series_id, s.name, q.horizon_months, m.model_type AS model,
              count(*)::int AS n,
              sqrt(avg((f.error / nullif(r.actual_value, 0)) ^ 2))::float8 AS relative_rmse,
              avg(f.in_interval::int)::float8 AS coverage
       ${joins} JOIN series s ON s.series_id = q.question_key
       WHERE ${scope} AND q.question_type = 'numeric_interval'
       GROUP BY 1, 2, 3, 4 ORDER BY 1, 3, 4`,
      [includeBacktest],
    ),
    db().query<{ kind: QuestionKind; model: string; n: number; brier: number }>(
      `SELECT ${KIND_SQL} AS kind, m.model_type AS model, count(*)::int AS n,
              avg(f.brier_score)::float8 AS brier
       ${joins}
       WHERE ${scope} AND q.question_type = 'binary'
       GROUP BY 1, 2 ORDER BY 1, 4`,
      [includeBacktest],
    ),
    db().query<{ model: string; n: number; brier: number; market_brier: number }>(
      `SELECT m.model_type AS model, count(*)::int AS n,
              avg(f.brier_score)::float8 AS brier, avg(fm.brier_score)::float8 AS market_brier
       ${joins}
       JOIN forecasts fm ON fm.question_id = q.id AND fm.forecast_date = f.forecast_date
            AND fm.scored_at IS NOT NULL
       JOIN model_versions mm ON mm.id = fm.model_version_id AND mm.model_type = 'ai_market_price'
       WHERE ${scope} AND q.platform_question_id IS NOT NULL
         AND m.model_type <> 'ai_market_price'
       GROUP BY 1 ORDER BY 3`,
      [includeBacktest],
    ),
    db().query<{
      model: string;
      bin: number;
      predicted: number;
      observed: number;
      n: number;
    }>(
      `SELECT m.model_type AS model,
              least(floor((f.probabilities ->> 'yes')::float8 * 10), 9)::int AS bin,
              avg((f.probabilities ->> 'yes')::float8)::float8 AS predicted,
              avg((r.actual_outcome = 'yes')::int)::float8 AS observed, count(*)::int AS n
       ${joins}
       WHERE ${scope} AND q.question_type = 'binary'
       GROUP BY 1, 2 ORDER BY 1, 2`,
      [includeBacktest],
    ),
    db().query<{ category: string; n: number }>(
      `SELECT p.miss_category::text AS category, count(*)::int AS n
       FROM postmortems p JOIN forecasts f ON f.id = p.forecast_id
       JOIN questions q ON q.id = f.question_id
       WHERE q.domain = 'ai' AND ($1 OR NOT f.is_backtest)
       GROUP BY 1`,
      [includeBacktest],
    ),
    loadLLMCosts(),
  ]);
  const models = [...new Set(calibration.rows.map((r) => r.model))];
  return {
    includeBacktest,
    scored:
      numeric.rows.reduce((s, r) => s + r.n, 0) +
      binary.rows.reduce((s, r) => s + r.n, 0),
    numeric: numeric.rows.map((r) => ({
      seriesId: r.series_id,
      name: r.name,
      horizonMonths: r.horizon_months,
      model: r.model,
      n: r.n,
      relativeRmse: r.relative_rmse,
      coverage: r.coverage,
    })),
    binary: binary.rows,
    vsMarket: vsMarket.rows.map((r) => ({
      model: r.model,
      n: r.n,
      brier: r.brier,
      marketBrier: r.market_brier,
    })),
    calibration: models.map((model) => ({
      model,
      bins: calibration.rows
        .filter((r) => r.model === model)
        .map((r) => ({
          lower: r.bin / 10,
          predicted: r.predicted,
          observed: r.observed,
          n: r.n,
        })),
    })),
    categories: Object.fromEntries(categories.rows.map((r) => [r.category, r.n])),
    llm,
  };
}

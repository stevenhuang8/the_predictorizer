import { monthYear } from "../_components/format";
import { HorizonChart } from "../_components/horizon-chart";
import { modelName, modelTone } from "../forecasts/models";
import type { NumericTarget } from "@/lib/dashboard-data";
import { type BacktestResults, loadBacktest } from "@/lib/backtest-results";
import { CoverageMeter, ModelLabel, ValueBar } from "./bars";

const TITLES: Record<NumericTarget, string> = {
  cpi_yoy: "Inflation (CPI, year over year)",
  unemployment: "Unemployment rate",
};

/** Display order: the live models first, then the extra baselines. */
const ORDER = ["lightgbm", "arima", "random_walk", "ets", "historical_mean"];
const FOMC_ORDER = ["fomc_lightgbm", "persistence", "climatology", "always_hold"];

function sortBy<T extends { model: string }>(rows: T[], order: string[]): T[] {
  const rank = (m: string) => (order.includes(m) ? order.indexOf(m) : order.length);
  return [...rows].sort((a, b) => rank(a.model) - rank(b.model));
}

function modelsOf(results: BacktestResults, target: NumericTarget): string[] {
  const models = [
    ...new Set(results.metrics.filter((m) => m.target === target).map((m) => m.model)),
  ];
  return sortBy(
    models.map((model) => ({ model })),
    ORDER,
  ).map((m) => m.model);
}

/** RMSE by horizon, one line per model: how fast each one's error grows. */
function HorizonCharts({
  results,
  targets,
}: {
  results: BacktestResults;
  targets: NumericTarget[];
}) {
  return (
    <div>
      <h3 className="font-semibold">Misses by forecast horizon</h3>
      <p className="mt-1 max-w-2xl text-sm text-muted">
        Typical miss (RMSE, percentage points) at each horizon. A model whose line sits
        under the orange &ldquo;no change&rdquo; line beat simply assuming nothing
        changes.
      </p>
      <div className="mt-4 grid gap-8 md:grid-cols-2">
        {targets.map((t) => (
          <HorizonChart
            key={t}
            title={TITLES[t]}
            series={modelsOf(results, t).map((model) => ({
              id: model,
              label: modelName(model),
              tone: modelTone(model),
              points: results.metrics
                .filter((m) => m.target === t && m.model === model)
                .sort((a, b) => a.horizon - b.horizon)
                .map((m) => ({ horizon: m.horizon, value: m.rmse })),
            }))}
          />
        ))}
      </div>
    </div>
  );
}

function NumericTable({
  target,
  results,
}: {
  target: NumericTarget;
  results: BacktestResults;
}) {
  const metrics = results.metrics.filter((m) => m.target === target);
  const horizons = [...new Set(metrics.map((m) => m.horizon))].sort((a, b) => a - b);
  const models = modelsOf(results, target);
  const max = Math.max(...metrics.map((m) => m.rmse));
  const best = new Map(
    horizons.map((h) => [
      h,
      Math.min(...metrics.filter((m) => m.horizon === h).map((m) => m.rmse)),
    ]),
  );
  const coverage = (model: string) =>
    results.calibration.find(
      (c) => c.target === target && c.model === model && c.nominal === 0.8,
    )?.empirical;

  return (
    <div>
      <h4 className="font-semibold">{TITLES[target]}</h4>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[40rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Model</th>
              {horizons.map((h) => (
                <th key={h} className="py-2 pr-4 font-medium">
                  {h} month{h === 1 ? "" : "s"} ahead
                </th>
              ))}
              <th className="py-2 font-medium">80% ranges held</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const share = coverage(model);
              return (
                <tr key={model} className="border-b border-line">
                  <td className="py-2 pr-4">
                    <ModelLabel name={modelName(model)} tone={modelTone(model)} />
                  </td>
                  {horizons.map((h) => {
                    const m = metrics.find((x) => x.model === model && x.horizon === h);
                    return (
                      <td key={h} className="py-2 pr-4">
                        {m ? (
                          <ValueBar
                            value={m.rmse}
                            max={max}
                            tone={modelTone(model)}
                            best={m.rmse === best.get(h)}
                            label={`${modelName(model)}, ${h}-month RMSE`}
                          />
                        ) : (
                          "–"
                        )}
                      </td>
                    );
                  })}
                  <td className="py-2">
                    {share != null ? (
                      <CoverageMeter share={share} tone={modelTone(model)} />
                    ) : (
                      "–"
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FomcTable({ results }: { results: BacktestResults }) {
  const rows = sortBy(results.fomc, FOMC_ORDER);
  const max = Math.max(...rows.map((r) => r.brier));
  const best = Math.min(...rows.map((r) => r.brier));
  const changes = rows[0]?.n_changes ?? 0;
  return (
    <div>
      <h4 className="font-semibold">Federal Reserve rate decisions</h4>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[34rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Brier score</th>
              <th className="py-2 pr-4 font-medium">Right call</th>
              <th className="py-2 font-medium">
                Brier when rates moved ({changes} meetings)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className="border-b border-line">
                <td className="py-2 pr-4">
                  <ModelLabel name={modelName(r.model)} tone={modelTone(r.model)} />
                </td>
                <td className="py-2 pr-4">
                  <ValueBar
                    value={r.brier}
                    max={max}
                    tone={modelTone(r.model)}
                    best={r.brier === best}
                    label={`${modelName(r.model)}, Brier score`}
                  />
                </td>
                <td className="py-2 pr-4 tabular-nums">
                  {Math.round(r.accuracy * 100)}%
                </td>
                <td className="py-2 tabular-nums">{r.brier_on_changes.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Backtest() {
  const results = loadBacktest();
  if (!results) {
    return (
      <p className="text-muted">
        No backtest report yet. Run{" "}
        <code className="font-mono">uv run eco-forecast backtest</code> to generate one.
      </p>
    );
  }
  const { settings, success_criterion: goal } = results;
  const n = results.metrics[0]?.n;
  const fomcN = results.fomc[0]?.n_resolved;
  const targets = [...new Set(results.metrics.map((m) => m.target))];

  return (
    <div className="space-y-8">
      <p className="max-w-2xl text-muted">
        How each model would have done from {monthYear(settings.start)} to{" "}
        {monthYear(settings.end)}, replayed month by month with only the data published
        at the time{n ? ` (${n} forecasts per model and horizon` : ""}
        {fomcN ? `, ${fomcN} FOMC meetings)` : n ? ")" : ""}. This is a simulation, not
        a live record.
      </p>

      <div className="rounded-lg border border-line bg-panel p-4 text-sm">
        <p className="font-semibold">
          The project&apos;s goal was {goal.passed ? "met" : "not met"}.
        </p>
        <p className="mt-1 text-muted">
          The aim was for LightGBM to beat the &ldquo;no change&rdquo; baseline on
          3-month-ahead CPI. It scored an RMSE of {goal.lightgbm_rmse.toFixed(2)}{" "}
          against {goal.random_walk_rmse.toFixed(2)}
          {goal.passed ? "" : ", so the simpler forecast did better"}.
        </p>
      </div>

      <HorizonCharts results={results} targets={targets} />
      {targets.map((t) => (
        <NumericTable key={t} target={t} results={results} />
      ))}
      {results.fomc.length > 0 && <FomcTable results={results} />}

      <div className="text-sm text-muted">
        <h4 className="font-semibold text-foreground">Findings</h4>
        <ul className="mt-2 list-disc space-y-1.5 pl-5">
          {results.findings.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

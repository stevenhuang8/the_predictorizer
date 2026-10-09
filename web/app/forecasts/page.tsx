import type { Metadata } from "next";
import { ComingSoon } from "../_components/coming-soon";

export const metadata: Metadata = { title: "Forecasts" };

export default function ForecastsPage() {
  return (
    <ComingSoon
      title="Forecasts"
      summary="The latest live forecast for every open question, from every model."
      planned={[
        "CPI and unemployment: point forecasts and 80% intervals by model and horizon",
        "FOMC: cut / hold / hike probabilities for upcoming meetings",
        "The top SHAP drivers behind each LightGBM forecast",
        "The data vintage each forecast was made with",
      ]}
      source="questions, forecasts, model_versions, feature_snapshots"
    />
  );
}

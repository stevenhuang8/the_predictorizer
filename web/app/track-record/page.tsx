import type { Metadata } from "next";
import { ComingSoon } from "../_components/coming-soon";

export const metadata: Metadata = { title: "Track record" };

export default function TrackRecordPage() {
  return (
    <ComingSoon
      title="Track record"
      summary="How every live forecast turned out, and why the misses missed."
      planned={[
        "Accuracy by model and horizon: RMSE, MAE, interval coverage, Brier score",
        "Calibration: stated interval coverage against how often actuals landed inside",
        "Week-ahead and month-ahead FOMC forecasts scored separately",
        "Post-mortems: each miss classified as bad data, bad model, regime change or expected variance",
      ]}
      source="resolutions, forecasts (scores), postmortems"
    />
  );
}

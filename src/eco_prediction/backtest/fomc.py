"""Walk-forward backtesting of FOMC decision probabilities.

Usage:
    from eco_prediction.backtest.fomc import FOMCBacktest, summarize_fomc

    bt = FOMCBacktest(date(2015, 1, 1), date(2024, 12, 31), "annually")
    results = bt.run(FOMCForecaster(lead_days=7))
    summarize_fomc(results)  # {'n_resolved': 80, 'brier': ..., 'accuracy': ...}

Every scheduled meeting with a decision day in [start_date, end_date] is
forecast at its `forecast_cutoff` (`lead_days` before, using data through the
day before). The model is refit at the first meeting's cutoff and again once
`retrain_frequency` has passed since the last fit ("every_meeting" refits each
time).

Models never see the full store: `fit` gets `store.until(fit cutoff)` and
`predict_proba` gets `store.until(forecast cutoff)`, so nothing published
later is reachable. Each forecast is scored against the decision as known in
the latest data; meetings still to come get no outcome and no score.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd

from eco_prediction.backtest import metrics
from eco_prediction.backtest.data import VintageStore
from eco_prediction.backtest.harness import months_between
from eco_prediction.data.fomc import (
    OUTCOMES,
    RANGE_UPPER_SERIES,
    RELEASE_LAGS,
    TARGET_SERIES,
    decision,
    fed_funds_target,
    meetings_between,
)
from eco_prediction.models.fomc_model import FOMCModel, forecast_cutoff

FOMCRetrain = Literal["every_meeting", "quarterly", "annually"]
RETRAIN_MONTHS = {"every_meeting": 0, "quarterly": 3, "annually": 12}


class FOMCBacktest:
    def __init__(
        self,
        start_date: date,
        end_date: date,
        retrain_frequency: FOMCRetrain = "annually",
        *,
        store: VintageStore | None = None,
    ) -> None:
        if end_date < start_date:
            raise ValueError("end_date is before start_date")
        if retrain_frequency not in RETRAIN_MONTHS:
            raise ValueError(f"retrain_frequency must be one of {list(RETRAIN_MONTHS)}")
        self.start_date = start_date
        self.end_date = end_date
        self.retrain_frequency = retrain_frequency
        self.store = store or VintageStore.from_db(
            backfill=True, release_lags=RELEASE_LAGS
        )

    def resolve_as_of(self) -> date:
        """The latest date any target data was published: outcomes are as of then."""
        dates = [
            self.store.latest_vintage(s) for s in (TARGET_SERIES, RANGE_UPPER_SERIES)
        ]
        known = [d for d in dates if d is not None]
        if not known:
            raise ValueError("no fed funds target data in the store")
        return max(known)

    def run(self, model: FOMCModel) -> pd.DataFrame:
        """Forecast, resolve and score every meeting. One row per meeting."""
        resolve_as_of = self.resolve_as_of()
        target = fed_funds_target(self.store, resolve_as_of)
        every = RETRAIN_MONTHS[self.retrain_frequency]

        rows: list[dict[str, Any]] = []
        trained_as_of: date | None = None
        for meeting in meetings_between(self.start_date, self.end_date):
            cutoff = forecast_cutoff(meeting, model.lead_days)
            if trained_as_of is None or months_between(trained_as_of, cutoff) >= every:
                model.fit(self.store.until(cutoff), cutoff)
                trained_as_of = cutoff
            probs = model.predict_proba(self.store.until(cutoff), meeting)
            if set(probs) != set(OUTCOMES):
                raise ValueError(f"model returned categories {sorted(probs)}")
            outcome = decision(self.store, meeting, resolve_as_of, target)
            rows.append(
                {
                    "meeting": meeting,
                    "data_as_of": cutoff,
                    "trained_as_of": trained_as_of,
                    **{f"p_{o}": probs[o] for o in OUTCOMES},
                    "outcome": outcome,
                    "brier": None
                    if outcome is None
                    else metrics.brier_score(probs, outcome),
                }
            )
        frame = pd.DataFrame(
            rows,
            columns=[
                "meeting",
                "data_as_of",
                "trained_as_of",
                *(f"p_{o}" for o in OUTCOMES),
                "outcome",
                "brier",
            ],
        )
        frame["brier"] = frame["brier"].astype("float64")
        return frame


def probabilities(results: pd.DataFrame) -> list[dict[str, float]]:
    """The forecasts of a `run` as {category: probability} dicts."""
    return [
        {o: float(row[f"p_{o}"]) for o in OUTCOMES} for _, row in results.iterrows()
    ]


def summarize_fomc(results: pd.DataFrame) -> dict[str, float | int | None]:
    """Mean Brier score and hit rate over the resolved meetings of a `run`."""
    resolved = results[results["outcome"].notna()]
    n = len(resolved)
    if not n:
        return {"n_forecasts": len(results), "n_resolved": 0, "brier": None}
    probs = resolved[[f"p_{o}" for o in OUTCOMES]].to_numpy()
    predicted = [OUTCOMES[i] for i in probs.argmax(axis=1)]
    changes = resolved["outcome"] != "hold"
    return {
        "n_forecasts": len(results),
        "n_resolved": n,
        "brier": metrics.mean_brier_score(
            probabilities(resolved), list(resolved["outcome"])
        ),
        "accuracy": float(
            (pd.Series(predicted, index=resolved.index) == resolved["outcome"]).mean()
        ),
        "n_changes": int(changes.sum()),
        "brier_on_changes": float(resolved.loc[changes, "brier"].mean())
        if changes.any()
        else None,
    }

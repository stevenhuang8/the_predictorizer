"""Blending the LLM's forecasts with the statistical ones (AI task 19).

    from eco_prediction.ai.llm.ensemble import LLMEnsembleForecaster

    ens = LLMEnsembleForecaster("learned")
    blended = ens.walk_forward_binary(frame)     # one row per question, p_yes blended
    blended = ens.walk_forward_numeric(frame)    # one row per forecast, point/interval blended

**Weights** (`method`):
- `equal`: 1/k each.
- `inverse_variance`: proportional to 1 / mean squared error, Brier for
  binary forecasts, squared relative error (forecast / actual - 1) for
  numeric ones, so series of different sizes pool.
- `learned`: the point on a grid over the weight simplex (steps of
  `GRID_STEP`, 0.05) with the lowest past error; with more than
  `MAX_GRID_COMPONENTS` (3) components, inverse variance instead.
Every method falls back to equal weights until `min_history` (10) past
forecasts have resolved.

**Walk-forward.** Each forecast's weights come only from forecasts resolved
before it was made: for binary questions, outcomes resolved before the ask
date (`resolved_at < ask_date`); for numeric ones, actuals published by the
forecast's data cutoff (`actual_as_of <= data_as_of`). A component missing
for a forecast (an LLM failure, a model that couldn't fit) drops out and the
others' weights are renormalized.

Binary forecasts are a linear pool (a weighted average of probabilities).
Numeric forecasts average points, and lower and upper bounds separately
(quantile averaging), which keeps an 80% interval an 80% interval when the
components agree on its width.

Input frames are long: one row per (forecast, component), with a
`component` column naming the model. See `walk_forward_binary` and
`walk_forward_numeric` for the columns.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["equal", "inverse_variance", "learned"]
GRID_STEP = 0.05
MAX_GRID_COMPONENTS = 3
MIN_HISTORY = 10

__all__ = ["LLMEnsembleForecaster", "simplex_grid"]


def simplex_grid(k: int, step: float = GRID_STEP) -> np.ndarray:
    """Every weight vector of length k on a `step` grid, summing to 1."""
    n = round(1 / step)
    points = [c for c in itertools.product(range(n + 1), repeat=k - 1) if sum(c) <= n]
    return np.array([[*c, n - sum(c)] for c in points], dtype=float) / n


class LLMEnsembleForecaster:
    """See the module docstring."""

    def __init__(
        self, method: Method = "learned", *, min_history: int = MIN_HISTORY
    ) -> None:
        if method not in ("equal", "inverse_variance", "learned"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.min_history = min_history

    # --- weights ---------------------------------------------------------------------

    def weights(
        self, predictions: pd.DataFrame, target: pd.Series, *, binary: bool
    ) -> dict[str, float]:
        """Weights from past forecasts: `predictions` has one column per
        component (NaN where it made none), `target` the outcome (0/1) or actual."""
        components = list(predictions.columns)
        complete = predictions.notna().all(axis=1) & target.notna()
        p, y = predictions[complete], target[complete]
        if self.method == "equal" or len(p) < self.min_history:
            return dict.fromkeys(components, 1 / len(components))
        if self.method == "learned" and len(components) <= MAX_GRID_COMPONENTS:
            grid = simplex_grid(len(components))
            losses = [_loss(p.to_numpy() @ w, y.to_numpy()) for w in grid]
            best = grid[int(np.argmin(losses))]
            return dict(zip(components, best.tolist()))
        errors = {c: _loss(p[c].to_numpy(), y.to_numpy()) for c in components}
        inverse = {c: 1 / max(e, 1e-12) for c, e in errors.items()}
        total = sum(inverse.values())
        return {c: v / total for c, v in inverse.items()}

    @staticmethod
    def combine(
        values: Mapping[str, float | None], weights: Mapping[str, float]
    ) -> float | None:
        """The weighted average of the components present, renormalized."""
        present = {c: v for c, v in values.items() if v is not None and not pd.isna(v)}
        total = sum(weights.get(c, 0.0) for c in present)
        if not present:
            return None
        if total <= 0:  # every present component got weight 0: fall back to equal
            return float(np.mean(list(present.values())))
        return float(sum(weights.get(c, 0.0) * v for c, v in present.items()) / total)

    # --- walk-forward --------------------------------------------------------------------

    def walk_forward_binary(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Blend binary forecasts question by question, in ask-date order.

        `frame` columns: `question_id`, `ask_date`, `component`, `p_yes`,
        `outcome` ('yes'/'no'/None), `resolved_at` (when the outcome became
        known; NaT if unresolved). Returns one row per question with the
        blended `p_yes`, `outcome`, `brier` and the `weights` used.
        """
        wide = frame.pivot_table(
            index=["question_id", "ask_date"],
            columns="component",
            values="p_yes",
            aggfunc="first",
        )
        meta = frame.groupby("question_id")[["outcome", "resolved_at"]].first()
        # Compare in naive UTC: database times are tz-aware, ask dates aren't.
        meta["resolved_at"] = pd.to_datetime(
            meta["resolved_at"], utc=True
        ).dt.tz_convert(None)
        wide = wide.reset_index().sort_values(["ask_date", "question_id"])
        components = [c for c in wide.columns if c not in ("question_id", "ask_date")]
        y = meta["outcome"].map({"yes": 1.0, "no": 0.0})
        rows = []
        for _, row in wide.iterrows():
            past = wide[
                wide["question_id"].map(meta["resolved_at"])
                < pd.Timestamp(row["ask_date"])
            ]
            w = self.weights(
                past.set_index("question_id")[components],
                y.reindex(past["question_id"]).set_axis(past["question_id"]),
                binary=True,
            )
            p = self.combine({c: row[c] for c in components}, w)
            outcome = meta.at[row["question_id"], "outcome"]
            rows.append({
                "question_id": row["question_id"], "ask_date": row["ask_date"],
                "p_yes": p, "outcome": outcome,
                "brier": None if p is None or outcome is None else (p - (outcome == "yes")) ** 2,
                "weights": w, "n_history": len(past),
            })  # fmt: skip
        return pd.DataFrame(rows)

    def walk_forward_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Blend numeric forecasts in forecast-date order.

        `frame` columns: `series_id`, `forecast_date`, `target_period`,
        `data_as_of`, `component`, `prediction`, `lower`, `upper`, `actual`,
        `actual_as_of` (harness results with `series_id` and `component`
        added). Returns one row per forecast with blended `prediction`,
        `lower`, `upper`, the `actual`, and the `weights` used.
        """
        key = ["series_id", "forecast_date", "target_period"]
        wide = {
            col: frame.pivot_table(
                index=key, columns="component", values=col, aggfunc="first"
            )
            for col in ("prediction", "lower", "upper")
        }
        meta = frame.groupby(key)[["data_as_of", "actual", "actual_as_of"]].first()
        rows = []
        for idx in sorted(meta.index, key=lambda k: (k[1], k[0])):
            cutoff = meta.at[idx, "data_as_of"]
            known = meta[
                meta["actual_as_of"].notna() & (meta["actual_as_of"] <= cutoff)
            ].index
            past = wide["prediction"].loc[known]
            actual = meta.loc[known, "actual"]
            w = self.weights(
                past.div(actual, axis=0) - 1, pd.Series(0.0, index=known), binary=False
            )
            blended = {
                col: self.combine(wide[col].loc[idx].to_dict(), w)
                for col in ("prediction", "lower", "upper")
            }
            rows.append({
                **dict(zip(key, idx)), **blended,
                "actual": meta.at[idx, "actual"], "weights": w, "n_history": len(known),
            })  # fmt: skip
        return pd.DataFrame(rows)


def _loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error: Brier for probabilities, else of the relative errors."""
    return float(np.mean((prediction - target) ** 2))

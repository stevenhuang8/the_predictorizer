"""SHAP explanations of LightGBM forecasts.

Usage:
    from eco_prediction.models.explainability import SHAPExplainer

    model.fit(train)
    explainer = SHAPExplainer(model)            # once per fit
    explanation = explainer.explain(view, target_period)
    explanation.top(5)                          # [('cpi_yoy', 0.41), ('ppi_3m', -0.12), ...]
    explanation.to_json()                       # for forecasts.shap_values

TreeSHAP (`shap.TreeExplainer`) is exact for tree models, so the
contributions add up to the model's output:

    prediction = offset + expected_value + sum(contributions)

`expected_value` is the model's mean output over its training rows. With
`predict_change` the model outputs a change, and `offset` is the latest known
target value it is added to; for a level model `offset` is 0. So "the latest
CPI YoY was 3.1, a typical 3-month change is +0.0, and these features moved
the forecast by ..." is how an explanation reads.

Contributions are in the target's units (percentage points for CPI YoY and
unemployment), one per feature the model was trained on. A feature that was
NaN in the forecast row still gets a contribution: LightGBM routes missing
values down a learned branch, and SHAP attributes the effect of that choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import shap  # type: ignore[import-untyped]

from eco_prediction.backtest.data import PointInTimeData
from eco_prediction.models.baselines import NotFittedError
from eco_prediction.models.lightgbm_model import LightGBMModel


@dataclass(frozen=True)
class Explanation:
    """Feature contributions to one forecast; see the module docstring."""

    expected_value: float
    offset: float
    contributions: dict[str, float]

    @property
    def prediction(self) -> float:
        return self.offset + self.expected_value + sum(self.contributions.values())

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        """The n largest contributions by absolute size, signed."""
        ranked = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))
        return ranked[:n]

    def to_json(self) -> dict[str, Any]:
        """A JSON object for `forecasts.shap_values`."""
        return {
            "expected_value": self.expected_value,
            "offset": self.offset,
            "contributions": dict(self.contributions),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Explanation:
        return cls(
            float(value["expected_value"]),
            float(value["offset"]),
            {k: float(v) for k, v in value["contributions"].items()},
        )


class SHAPExplainer:
    """TreeSHAP over a fitted `LightGBMModel`. Build a new one after each fit."""

    def __init__(self, model: LightGBMModel) -> None:
        if model.model is None:
            raise NotFittedError("fit the model before explaining it")
        self.model = model
        self.booster = model.model
        self.explainer = shap.TreeExplainer(model.model)

    def explain(self, data: PointInTimeData, target_period: date) -> Explanation:
        """Explain the forecast `model.predict(data, target_period)` makes."""
        if self.model.model is not self.booster:
            raise ValueError("the model was refit; build a new SHAPExplainer")
        row, offset = self.model.prediction_inputs(data, target_period)
        values = self.explainer.shap_values(row)[0]
        return Explanation(
            expected_value=float(self.explainer.expected_value),
            offset=offset,
            contributions={
                name: float(v) for name, v in zip(row.columns, values, strict=True)
            },
        )

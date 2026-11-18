"""Remaining Useful Life (RUL) estimation and NASA asymmetric scoring metric."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.telemetry.sensor_window import UnitDegradationSummary


@dataclass(frozen=True)
class LinearRulModel:
    """Immutable linear RUL regressor from degradation slope."""

    intercept: float
    slope_weight: float


def fit_rul_regressor(
    summaries: tuple[UnitDegradationSummary, ...],
    observed_ruls: tuple[float, ...],
) -> LinearRulModel:
    """Fit OLS RUL regressor against unit temperature degradation slope."""
    if len(summaries) != len(observed_ruls) or len(summaries) < 3:
        raise ValueError("At least 3 matched summaries and RUL targets are required.")

    xs = [s.temp_slope for s in summaries]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(observed_ruls) / len(observed_ruls)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 1e-12:
        return LinearRulModel(intercept=mean_y, slope_weight=0.0)

    weight = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, observed_ruls)) / var_x
    intercept = mean_y - weight * mean_x
    return LinearRulModel(intercept=intercept, slope_weight=weight)


def predict_rul(model: LinearRulModel, summary: UnitDegradationSummary) -> float:
    """Predict non-negative Remaining Useful Life in cycles."""
    raw = model.intercept + model.slope_weight * summary.temp_slope
    return max(0.0, raw)


def compute_nasa_asymmetric_score(
    predictions: tuple[float, ...],
    targets: tuple[float, ...],
) -> float:
    """Compute NASA C-MAPSS asymmetric penalty (late predictions penalized more heavily)."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and equal length.")

    total = 0.0
    for pred, true in zip(predictions, targets):
        diff = pred - true
        # Late prediction (diff > 0) uses tighter denominator 10 vs 13 for early warning.
        denom = 10.0 if diff >= 0 else 13.0
        total += math.exp(abs(diff) / denom) - 1.0
    return total / len(predictions)

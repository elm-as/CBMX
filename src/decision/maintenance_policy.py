"""Cost-optimal preventive maintenance threshold selection under asymmetric risk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceCostConfig:
    """Asymmetric cost parameters for preventive vs unplanned failure interventions."""

    preventive_cost_eur: float = 1_500.0
    unplanned_failure_cost_eur: float = 12_000.0


@dataclass(frozen=True)
class PolicyEvaluation:
    """Expected cost and selected failure-probability threshold."""

    optimal_threshold: float
    expected_cost_eur: float


def select_optimal_threshold(
    failure_probs: tuple[float, ...],
    actual_failures: tuple[bool, ...],
    cost_config: MaintenanceCostConfig,
) -> PolicyEvaluation:
    """Select probability threshold minimizing total expected maintenance cost."""
    if len(failure_probs) != len(actual_failures) or not failure_probs:
        raise ValueError("failure_probs and actual_failures must be non-empty and equal length.")

    best_threshold = 0.5
    best_cost = float("inf")
    for step in range(1, 20):
        threshold = step * 0.05
        cost = _evaluate_threshold_cost(failure_probs, actual_failures, threshold, cost_config)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return PolicyEvaluation(optimal_threshold=round(best_threshold, 2), expected_cost_eur=best_cost)


def _evaluate_threshold_cost(
    failure_probs: tuple[float, ...],
    actual_failures: tuple[bool, ...],
    threshold: float,
    cost_config: MaintenanceCostConfig,
) -> float:
    """Compute total fleet cost for a candidate intervention probability threshold."""
    total_cost = 0.0
    for prob, failed in zip(failure_probs, actual_failures):
        if prob >= threshold:
            total_cost += cost_config.preventive_cost_eur
        elif failed:
            total_cost += cost_config.unplanned_failure_cost_eur
    return total_cost / len(failure_probs)

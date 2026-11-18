"""Kaplan-Meier survival estimator and Harrell's C-index for right-censored data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurvivalPoint:
    """Step on the Kaplan-Meier survival curve."""

    time: int
    survival_prob: float
    at_risk: int
    events: int


def fit_kaplan_meier(
    durations: tuple[int, ...],
    events: tuple[bool, ...],
) -> tuple[SurvivalPoint, ...]:
    """Compute exact Kaplan-Meier product-limit survival curve for right-censored data."""
    if len(durations) != len(events) or not durations:
        raise ValueError("durations and events must be non-empty and equal length.")

    unique_event_times = sorted({t for t, ev in zip(durations, events) if ev})
    survival = 1.0
    curve: list[SurvivalPoint] = []
    for t in unique_event_times:
        at_risk = sum(1 for d in durations if d >= t)
        n_events = sum(1 for d, ev in zip(durations, events) if d == t and ev)
        if at_risk > 0:
            survival *= 1.0 - (n_events / at_risk)
        curve.append(SurvivalPoint(time=t, survival_prob=survival, at_risk=at_risk, events=n_events))
    return tuple(curve)


def compute_concordance_index(
    risk_scores: tuple[float, ...],
    durations: tuple[int, ...],
    events: tuple[bool, ...],
) -> float:
    """Compute Harrell's C-index for right-censored survival predictions."""
    if not (len(risk_scores) == len(durations) == len(events)) or not durations:
        raise ValueError("All inputs must be non-empty and of equal length.")

    concordant = 0.0
    permissible = 0.0
    n = len(durations)
    for i in range(n):
        if not events[i]:
            continue
        concordant_i, permissible_i = _count_pairs_for_event(i, risk_scores, durations, events)
        concordant += concordant_i
        permissible += permissible_i

    if permissible == 0.0:
        raise ValueError("No permissible pairs found for C-index calculation.")
    return concordant / permissible


def _count_pairs_for_event(
    idx: int,
    risk_scores: tuple[float, ...],
    durations: tuple[int, ...],
    events: tuple[bool, ...],
) -> tuple[float, float]:
    """Count concordant and permissible pairs relative to an observed failure at idx."""
    concordant = 0.0
    permissible = 0.0
    for j in range(len(durations)):
        if durations[j] <= durations[idx]:
            continue
        permissible += 1.0
        if risk_scores[idx] > risk_scores[j]:
            concordant += 1.0
        elif risk_scores[idx] == risk_scores[j]:
            concordant += 0.5
    return concordant, permissible

"""CLI entrypoint for CBMX (Condition-Based Maintenance & Survival Intelligence)."""

from __future__ import annotations

from src.decision.maintenance_policy import MaintenanceCostConfig, select_optimal_threshold
from src.evaluation.group_split import split_by_unit_id
from src.survival.kaplan_meier import compute_concordance_index, fit_kaplan_meier
from src.telemetry.sensor_window import generate_synthetic_fleet, summarize_unit_degradation


def run_pipeline() -> dict[str, float]:
    """Execute end-to-end CBMX benchmark: unit-grouped survival + cost-optimal maintenance."""
    fleet = generate_synthetic_fleet(n_units=24, seed=42)
    summaries = summarize_unit_degradation(fleet)
    split = split_by_unit_id(tuple(s.unit_id for s in summaries), test_ratio=0.25, seed=42)

    test_units = tuple(s for s in summaries if s.unit_id in split.test_unit_ids)
    durations = tuple(s.observed_cycles for s in test_units)
    events = tuple(s.event_observed for s in test_units)
    risks = tuple(s.temp_slope for s in test_units)

    km_curve = fit_kaplan_meier(durations, events)
    c_index = compute_concordance_index(risks, durations, events)

    max_slope = max(risks) if risks else 1.0
    probs = tuple(min(0.95, max(0.05, r / max(max_slope, 1e-6))) for r in risks)
    policy = select_optimal_threshold(probs, events, MaintenanceCostConfig())

    return {
        "km_steps": float(len(km_curve)),
        "c_index": round(c_index, 4),
        "optimal_threshold": policy.optimal_threshold,
        "expected_cost_eur": round(policy.expected_cost_eur, 2),
    }


if __name__ == "__main__":
    metrics = run_pipeline()
    print("[CBMX] Pipeline completed:", metrics)

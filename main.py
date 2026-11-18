"""CLI entrypoint for CBMX (Condition-Based Maintenance & Survival Intelligence)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.decision.maintenance_policy import MaintenanceCostConfig, select_optimal_threshold
from src.evaluation.group_split import split_by_unit_id
from src.survival.kaplan_meier import compute_concordance_index, fit_kaplan_meier
from src.telemetry.sensor_window import (
    EngineCycle,
    generate_synthetic_fleet,
    summarize_unit_degradation,
)

DATA_PATH = Path("data/turbofan_fleet.csv")
RESULTS_PATH = Path("artifacts/results.json")


def ensure_dataset(csv_path: Path) -> tuple[EngineCycle, ...]:
    """Load existing fleet CSV or generate and persist a 60-unit turbofan benchmark."""
    if csv_path.exists():
        rows: list[EngineCycle] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    EngineCycle(
                        unit_id=int(row["unit_id"]),
                        cycle=int(row["cycle"]),
                        temp_sensor=float(row["temp_sensor"]),
                        vibration_sensor=float(row["vibration_sensor"]),
                        failed=row["failed"].lower() == "true",
                    )
                )
        return tuple(rows)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fleet = generate_synthetic_fleet(n_units=60, seed=42)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit_id", "cycle", "temp_sensor", "vibration_sensor", "failed"])
        for row in fleet:
            writer.writerow([row.unit_id, row.cycle, row.temp_sensor, row.vibration_sensor, row.failed])
    return fleet


def run_pipeline() -> dict[str, float]:
    """Execute end-to-end CBMX benchmark: unit-grouped survival + cost-optimal maintenance."""
    fleet = ensure_dataset(DATA_PATH)
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

    metrics = {
        "n_units": float(len(summaries)),
        "n_cycles": float(len(fleet)),
        "km_steps": float(len(km_curve)),
        "c_index": round(c_index, 4),
        "optimal_threshold": policy.optimal_threshold,
        "expected_cost_eur": round(policy.expected_cost_eur, 2),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    output = run_pipeline()
    print("[CBMX] Pipeline completed:", output)

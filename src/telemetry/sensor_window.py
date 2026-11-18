"""Multi-sensor engine telemetry validation and causal rolling degradation features."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCycle:
    """Immutable sensor snapshot for a single industrial unit at a given cycle."""

    unit_id: int
    cycle: int
    temp_sensor: float
    vibration_sensor: float
    failed: bool


@dataclass(frozen=True)
class UnitDegradationSummary:
    """Causal summary of a unit's telemetry up to its latest observed cycle."""

    unit_id: int
    observed_cycles: int
    latest_temp: float
    temp_slope: float
    event_observed: bool


def validate_cycle_monotonicity(cycles: tuple[EngineCycle, ...]) -> None:
    """Ensure cycles within each unit_id are strictly increasing without duplicates."""
    last_seen: dict[int, int] = {}
    for row in cycles:
        prev = last_seen.get(row.unit_id)
        if prev is not None and row.cycle <= prev:
            raise ValueError(
                f"Non-monotonic cycle for unit {row.unit_id}: {row.cycle} <= {prev}"
            )
        last_seen[row.unit_id] = row.cycle


def summarize_unit_degradation(
    cycles: tuple[EngineCycle, ...],
) -> tuple[UnitDegradationSummary, ...]:
    """Compute causal degradation slope per unit using only observed cycles."""
    validate_cycle_monotonicity(cycles)
    by_unit: dict[int, list[EngineCycle]] = {}
    for row in cycles:
        by_unit.setdefault(row.unit_id, []).append(row)

    summaries: list[UnitDegradationSummary] = []
    for unit_id in sorted(by_unit.keys()):
        summaries.append(_compute_unit_summary(unit_id, by_unit[unit_id]))
    return tuple(summaries)


def _compute_unit_summary(unit_id: int, unit_rows: list[EngineCycle]) -> UnitDegradationSummary:
    """Compute linear temperature degradation slope for one unit."""
    first = unit_rows[0]
    last = unit_rows[-1]
    span = max(1, last.cycle - first.cycle)
    slope = (last.temp_sensor - first.temp_sensor) / span
    return UnitDegradationSummary(
        unit_id=unit_id,
        observed_cycles=last.cycle,
        latest_temp=last.temp_sensor,
        temp_slope=slope,
        event_observed=any(r.failed for r in unit_rows),
    )


def generate_synthetic_fleet(n_units: int = 24, seed: int = 42) -> tuple[EngineCycle, ...]:
    """Generate a reproducible C-MAPSS-like fleet with right-censored units."""
    if n_units < 4:
        raise ValueError("n_units must be at least 4.")

    rng = random.Random(seed)
    rows: list[EngineCycle] = []
    for unit_id in range(1, n_units + 1):
        censored = unit_id % 4 == 0
        total_cycles = rng.randint(45, 110)
        rate = rng.uniform(0.08, 0.25)
        for cycle in range(1, total_cycles + 1):
            temp = 600.0 + rate * cycle + rng.gauss(0.0, 1.2)
            vib = 0.20 + 0.004 * rate * cycle + rng.gauss(0.0, 0.01)
            is_fail = (not censored) and (cycle == total_cycles)
            rows.append(EngineCycle(unit_id, cycle, round(temp, 3), round(vib, 4), is_fail))
    return tuple(rows)

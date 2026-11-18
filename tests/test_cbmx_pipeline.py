"""Automated test suite for CBMX: unit-group split, Kaplan-Meier, C-index, NASA score, policy."""

from __future__ import annotations

import pytest

from src.decision.maintenance_policy import MaintenanceCostConfig, select_optimal_threshold
from src.evaluation.group_split import split_by_unit_id
from src.rul.rul_calibrator import compute_nasa_asymmetric_score, fit_rul_regressor, predict_rul
from src.survival.kaplan_meier import compute_concordance_index, fit_kaplan_meier
from src.telemetry.sensor_window import (
    EngineCycle,
    generate_synthetic_fleet,
    summarize_unit_degradation,
    validate_cycle_monotonicity,
)


def test_group_split_has_zero_unit_overlap() -> None:
    split = split_by_unit_id((1, 1, 2, 2, 3, 4, 5, 6, 7, 8), test_ratio=0.25, seed=9)
    assert split.train_unit_ids.isdisjoint(split.test_unit_ids)
    assert len(split.train_unit_ids) + len(split.test_unit_ids) == 8


def test_kaplan_meier_is_monotonically_non_increasing() -> None:
    curve = fit_kaplan_meier((10, 20, 25, 30, 40), (True, False, True, True, False))
    assert len(curve) == 3
    probs = [pt.survival_prob for pt in curve]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < probs[-1] < 1.0


def test_concordance_index_scores_perfect_ranking_as_one() -> None:
    c_index = compute_concordance_index(
        risk_scores=(0.9, 0.6, 0.3, 0.1),
        durations=(10, 20, 30, 40),
        events=(True, True, True, False),
    )
    assert c_index == pytest.approx(1.0)


def test_nasa_asymmetric_score_penalizes_late_prediction_more() -> None:
    early_penalty = compute_nasa_asymmetric_score((15.0,), (20.0,))
    late_penalty = compute_nasa_asymmetric_score((25.0,), (20.0,))
    assert late_penalty > early_penalty


def test_non_monotonic_cycles_raise_value_error() -> None:
    bad_rows = (
        EngineCycle(1, 2, 600.0, 0.2, False),
        EngineCycle(1, 2, 601.0, 0.2, True),
    )
    with pytest.raises(ValueError):
        validate_cycle_monotonicity(bad_rows)


def test_end_to_end_rul_and_maintenance_policy() -> None:
    fleet = generate_synthetic_fleet(n_units=16, seed=11)
    summaries = summarize_unit_degradation(fleet)
    model = fit_rul_regressor(summaries[:8], tuple(float(s.observed_cycles) for s in summaries[:8]))
    pred = predict_rul(model, summaries[8])
    assert pred >= 0.0

    policy = select_optimal_threshold(
        failure_probs=(0.9, 0.7, 0.2, 0.1),
        actual_failures=(True, True, False, False),
        cost_config=MaintenanceCostConfig(),
    )
    assert 0.05 <= policy.optimal_threshold <= 0.95

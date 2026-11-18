"""Group-aware splitting by unit_id to prevent cross-cycle equipment leakage."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitGroupSplit:
    """Disjoint sets of unit identifiers for train and test evaluation."""

    train_unit_ids: frozenset[int]
    test_unit_ids: frozenset[int]


def split_by_unit_id(
    unit_ids: tuple[int, ...],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> UnitGroupSplit:
    """Split unique unit_ids into strictly disjoint train and test sets."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must lie strictly between 0 and 1.")

    unique_units = sorted(set(unit_ids))
    if len(unique_units) < 2:
        raise ValueError("At least 2 distinct units are required for group splitting.")

    shuffled = list(unique_units)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_set = frozenset(shuffled[:n_test])
    train_set = frozenset(shuffled[n_test:])
    return UnitGroupSplit(train_unit_ids=train_set, test_unit_ids=test_set)

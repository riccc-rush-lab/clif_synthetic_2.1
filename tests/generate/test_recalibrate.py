"""Tests for spine acuity recalibration (population shaping)."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.generate.recalibrate import recalibrate_spine_acuity
from clifforge.generate.spine import sample_spine


def _high_acuity_fraction(pack, *, seed: int, n: int = 400) -> float:
    rng = np.random.default_rng(seed)
    levels = [
        lvl
        for i in range(n)
        for lvl in sample_spine(pack, rng, hospitalization_id=f"H{i}").support_level
    ]
    return sum(lvl >= 3 for lvl in levels) / len(levels)


def test_deescalation_reduces_high_acuity_time(pack) -> None:
    base = _high_acuity_fraction(pack, seed=1)
    recal = _high_acuity_fraction(recalibrate_spine_acuity(pack, deescalation=0.5), seed=1)
    assert recal < base  # less time in IMV-and-above states


def test_transition_rows_stay_valid_probability_distributions(pack) -> None:
    out = recalibrate_spine_acuity(pack, deescalation=0.4)
    matrix = out.tables["spine"]["params"]["support_level_transition_matrix"]
    for level, row in matrix.items():
        assert abs(sum(row.values()) - 1.0) < 1e-9, f"row {level} does not sum to 1"
        assert all(p >= 0 for p in row.values())
    start = out.tables["spine"]["params"]["support_level_start_dist"]
    assert abs(sum(start.values()) - 1.0) < 1e-9


def test_does_not_mutate_the_input_pack(pack) -> None:
    before = pack.tables["spine"]["params"]["support_level_transition_matrix"]["4"].copy()
    recalibrate_spine_acuity(pack, deescalation=0.5)
    assert pack.tables["spine"]["params"]["support_level_transition_matrix"]["4"] == before


def test_zero_deescalation_leaves_acuity_unchanged(pack) -> None:
    assert _high_acuity_fraction(recalibrate_spine_acuity(pack, deescalation=0.0), seed=3) == (
        _high_acuity_fraction(pack, seed=3)
    )


def test_rejects_out_of_range_deescalation(pack) -> None:
    with pytest.raises(ValueError, match="deescalation"):
        recalibrate_spine_acuity(pack, deescalation=1.0)

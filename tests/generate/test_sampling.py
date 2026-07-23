"""Unit tests for the shared categorical sampler (U7, R22)."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.generate.sampling import categorical


def test_categorical_is_deterministic_under_fixed_seed() -> None:
    marginal = {"a": 0.2, "b": 0.3, "c": 0.5}
    a = [categorical(marginal, np.random.default_rng(0)) for _ in range(1)]
    b = [categorical(marginal, np.random.default_rng(0)) for _ in range(1)]
    assert a == b


def test_categorical_order_independent() -> None:
    # Two dicts with identical mass but different insertion order must produce
    # the same draw for the same seed (keys are sorted before the CDF search).
    m1 = {"a": 0.2, "b": 0.3, "c": 0.5}
    m2 = {"c": 0.5, "a": 0.2, "b": 0.3}
    assert categorical(m1, np.random.default_rng(7)) == categorical(m2, np.random.default_rng(7))


def test_categorical_recovers_the_marginal() -> None:
    marginal = {"x": 0.1, "y": 0.6, "z": 0.3}
    rng = np.random.default_rng(11)
    draws = [categorical(marginal, rng) for _ in range(20_000)]
    for key, prob in marginal.items():
        assert abs(draws.count(key) / len(draws) - prob) < 0.02


def test_categorical_renormalizes_unnormalized_input() -> None:
    # A conditioned sub-marginal (mass < 1) is renormalized on the fly.
    marginal = {"a": 1.0, "b": 3.0}  # -> 0.25 / 0.75
    rng = np.random.default_rng(3)
    draws = [categorical(marginal, rng) for _ in range(20_000)]
    assert abs(draws.count("b") / len(draws) - 0.75) < 0.02


def test_categorical_rejects_empty_and_zero_mass() -> None:
    with pytest.raises(ValueError, match="empty"):
        categorical({}, np.random.default_rng(0))
    with pytest.raises(ValueError, match="non-positive"):
        categorical({"a": 0.0, "b": 0.0}, np.random.default_rng(0))

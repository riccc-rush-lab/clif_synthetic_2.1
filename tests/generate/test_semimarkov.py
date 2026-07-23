"""Unit tests for the hand-rolled semi-Markov engine (U6, KTD-3).

Exercised on hand-built transition/sojourn structures — no parameter pack and no
real data — so the engine's contract (determinism, absorption, horizon
censoring, non-geometric dwell times) is verified in isolation from the spine.
"""

from __future__ import annotations

import numpy as np

from clifforge.generate import semimarkov
from clifforge.generate.semimarkov import Visit, make_sojourn_sampler, sample

# A tiny two-level ladder that always discharges from level 1.
_TRANSITIONS = {
    0: {1: 1.0},
    1: {0: 0.5, "discharge": 0.5},
}
_ABSORBING = {"discharge"}
_START = {0: 1.0}


def _constant_sojourns(hours: float) -> dict[object, object]:
    return {0: lambda rng: hours, 1: lambda rng: hours}


def test_sample_is_deterministic_under_fixed_seed() -> None:
    sojourns = _constant_sojourns(3.0)
    a = sample(_TRANSITIONS, sojourns, _START, _ABSORBING, np.random.default_rng(42), horizon=1000)
    b = sample(_TRANSITIONS, sojourns, _START, _ABSORBING, np.random.default_rng(42), horizon=1000)
    assert [(v.state, v.duration, v.censored) for v in a] == [
        (v.state, v.duration, v.censored) for v in b
    ]


def test_trajectory_terminates_at_absorbing_state_within_horizon() -> None:
    # Constant 3h sojourns, generous horizon: termination must be by absorption,
    # not by censoring, and the terminal visit is the discharge state.
    sojourns = _constant_sojourns(3.0)
    visits = sample(
        _TRANSITIONS, sojourns, _START, _ABSORBING, np.random.default_rng(1), horizon=10_000
    )
    terminal = visits[-1]
    assert terminal.state == "discharge"
    assert terminal.duration == 0.0
    assert not terminal.censored
    # Discharge is absorbing: it appears only once, at the very end.
    assert [v.state for v in visits].count("discharge") == 1


def test_horizon_censors_the_final_visit() -> None:
    # A single long sojourn that overruns a short horizon must truncate and flag.
    sojourns = {0: lambda rng: 100.0, 1: lambda rng: 100.0}
    visits = sample(
        _TRANSITIONS, sojourns, _START, _ABSORBING, np.random.default_rng(0), horizon=5.0
    )
    assert len(visits) == 1
    assert visits[0].state == 0
    assert visits[0].duration == 5.0
    assert visits[0].censored
    # Total dwell time never exceeds the horizon.
    assert sum(v.duration for v in visits) <= 5.0


def test_non_absorbing_dead_end_exits_cleanly() -> None:
    # A state whose transition row is empty is an implicit exit, not a hang.
    visits = sample(
        {0: {}}, {0: lambda rng: 2.0}, {0: 1.0}, set(), np.random.default_rng(0), horizon=1000
    )
    assert [v.state for v in visits] == [0]
    assert not visits[0].censored


def test_lognormal_sojourns_are_non_geometric() -> None:
    # The whole point of a semi-Markov engine: dwell times follow the configured
    # continuous family, not a memoryless geometric law. A lognormal sampler must
    # reproduce lognormal moments (heavy right skew, mean matching exp(mu+s^2/2)).
    s, loc, scale = 0.5, 0.0, np.exp(1.0)  # lognorm(s=0.5, scale=e^1) -> median e
    sampler = make_sojourn_sampler("lognormal", [s, loc, scale])
    rng = np.random.default_rng(7)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    expected_mean = float(np.exp(1.0 + s**2 / 2))
    assert abs(draws.mean() - expected_mean) / expected_mean < 0.05
    assert (draws > 0).all()  # positive support
    # Right-skewed: mean well above median, unlike a symmetric or geometric law.
    assert draws.mean() > np.median(draws)


def test_empirical_mean_fallback_samples_exponential_with_that_mean() -> None:
    sampler = make_sojourn_sampler("empirical_mean", [4.0])
    rng = np.random.default_rng(3)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    assert abs(draws.mean() - 4.0) / 4.0 < 0.05
    assert (draws >= 0).all()


def test_family_factory_round_trips_all_known_families() -> None:
    # Every family the fit stage can emit must build a working sampler.
    rng = np.random.default_rng(11)
    specs = {
        "exponential": [0.0, 2.0],
        "gamma": [2.0, 0.0, 1.5],
        "lognormal": [0.6, 0.0, 3.0],
        "weibull": [1.3, 0.0, 5.0],
    }
    for family, params in specs.items():
        assert family in semimarkov.SOJOURN_FAMILIES
        draw = make_sojourn_sampler(family, params)(rng)
        assert draw >= 0.0 and np.isfinite(draw)


def test_visit_is_frozen_value_object() -> None:
    v = Visit(state=1, duration=2.0, censored=False)
    assert (v.state, v.duration, v.censored) == (1, 2.0, False)

"""Hand-rolled semi-Markov trajectory sampler (KTD-3).

A plain Markov chain gives only geometric dwell times; ICU state durations are
not geometric. This module samples a **semi-Markov** trajectory: an embedded
(jump) chain over states plus a per-state sojourn distribution drawn from a
positive-support parametric family (exponential / gamma / lognormal / Weibull /
an exponential fallback for a mean-only fit). It is a few dozen lines, has no
heavy dependency (numpy + scipy only), and is fully seedable — every draw comes
from a passed-in ``numpy.random.Generator``, so a fixed seed reproduces a
trajectory byte-for-byte.

The engine is deliberately generic over the state type: states are any hashable
value (the U6 spine uses ``int`` support levels plus the string ``"discharge"``
as an absorbing exit), and it knows nothing about CLIF, support levels, or the
parameter pack. ``clifforge.generate.spine`` adapts pack params into engine
inputs; keeping the engine pack-agnostic is what makes it unit-testable on
hand-built transition/sojourn structures.

Termination is either **absorption** (the chain enters a state in ``absorbing``,
which has no sojourn and no outgoing row) or **horizon censoring** (the running
total duration reaches ``horizon`` mid-sojourn, truncating the final visit). A
non-absorbing state with no outgoing transition row is treated as an implicit
exit — the trajectory ends rather than dead-looping.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "SOJOURN_FAMILIES",
    "Visit",
    "make_sojourn_sampler",
    "sample",
]

#: A callable that draws one non-negative dwell time from a passed rng.
SojournSampler = Callable[[np.random.Generator], float]

#: Positive-support families the fit stage selects among by AIC, keyed by the
#: family name stored in the pack. Each maps to the scipy distribution whose
#: ``rvs(*params, random_state=rng)`` reproduces a draw; the stored ``params``
#: are exactly that distribution's ``fit`` tuple (shape(s), loc, scale).
SOJOURN_FAMILIES: dict[str, stats.rv_continuous] = {
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
}


@dataclass(frozen=True)
class Visit:
    """One dwell in a state along a sampled trajectory.

    ``duration`` is the time spent in ``state`` (in the same units as the
    sojourn samplers and ``horizon``). ``censored`` is ``True`` only for a final
    visit truncated by the horizon — a natural jump or absorption is not
    censored. An absorbing state is recorded as a terminal ``Visit`` with
    ``duration == 0.0`` so the caller can see *why* the trajectory ended.
    """

    state: Hashable
    duration: float
    censored: bool


def make_sojourn_sampler(family: str, params: Sequence[float]) -> SojournSampler:
    """Build a seedable dwell-time sampler for a fitted sojourn family.

    ``family``/``params`` come straight from a pack's ``support_level_sojourn``
    block. Recognized families sample from the matching scipy distribution via
    ``rvs(*params, random_state=rng)``. The mean-only fallback the fit stage
    emits when no family converges (``"empirical_mean"`` with a single mean
    param) is sampled as an exponential with that mean — the maximum-entropy
    choice given only a mean, and never a degenerate constant.
    """
    if family in SOJOURN_FAMILIES:
        dist = SOJOURN_FAMILIES[family]
        args = tuple(float(p) for p in params)

        def _sample(rng: np.random.Generator) -> float:
            return float(dist.rvs(*args, random_state=rng))

        return _sample

    if family == "empirical_mean":
        mean = float(params[0]) if len(params) else 0.0
        scale = max(mean, 0.0)

        def _sample_mean(rng: np.random.Generator) -> float:
            return float(rng.exponential(scale)) if scale > 0 else 0.0

        return _sample_mean

    raise ValueError(f"unknown sojourn family {family!r}")


def _choice(dist: Mapping[Hashable, float], rng: np.random.Generator) -> Hashable:
    """Draw one key from a (possibly unnormalized) categorical via inverse-CDF.

    Uses a single ``rng.random()`` and a manual cumulative search so the state
    keys can be any hashable (int levels, the ``"discharge"`` string) rather
    than requiring the numeric-array coercion ``rng.choice`` would impose.
    """
    states = list(dist.keys())
    probs = np.asarray([dist[s] for s in states], dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("categorical distribution has non-positive total mass")
    cumulative = np.cumsum(probs / total)
    idx = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return states[min(idx, len(states) - 1)]


def sample(
    transitions: Mapping[Hashable, Mapping[Hashable, float]],
    sojourns: Mapping[Hashable, SojournSampler],
    start_dist: Mapping[Hashable, float],
    absorbing: Container[Hashable],
    rng: np.random.Generator,
    horizon: float,
    *,
    max_jumps: int = 100_000,
) -> list[Visit]:
    """Sample one semi-Markov trajectory as an ordered list of :class:`Visit`.

    Args:
        transitions: embedded jump chain, ``{from: {to: prob}}``. Rows need not
            be exactly normalized (they are renormalized on draw). A state with
            no row (or an empty row) ends the trajectory as an implicit exit.
        sojourns: per-state dwell-time sampler (see :func:`make_sojourn_sampler`).
            Only non-absorbing states are looked up.
        start_dist: initial-state law ``{state: prob}``.
        absorbing: states that terminate the trajectory on entry; they have no
            sojourn and no outgoing row.
        rng: the single source of randomness — a fixed seed is fully
            reproducible.
        horizon: maximum total duration; a sojourn crossing it truncates the
            final visit, which is then marked ``censored``.
        max_jumps: safety cap on the number of visits (guards against a
            pathological near-absorbing loop); reaching it ends the trajectory.

    Returns:
        The visit sequence in order. The terminal visit is either the absorbing
        state (``duration == 0.0``, not censored), a horizon-censored dwell
        (``censored is True``), or the last dwell before an implicit exit.
    """
    state = _choice(start_dist, rng)
    visits: list[Visit] = []
    elapsed = 0.0

    for _ in range(max_jumps):
        if state in absorbing:
            visits.append(Visit(state=state, duration=0.0, censored=False))
            return visits

        duration = max(0.0, sojourns[state](rng))
        remaining = horizon - elapsed
        if duration >= remaining:
            visits.append(Visit(state=state, duration=remaining, censored=True))
            return visits

        visits.append(Visit(state=state, duration=duration, censored=False))
        elapsed += duration

        row = transitions.get(state)
        if not row:  # non-absorbing dead-end: exit rather than loop forever
            return visits
        state = _choice(row, rng)

    return visits

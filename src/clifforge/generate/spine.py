"""Latent state spine sampler — Tier 0 of the generate stage (U6, KTD-6).

Each synthetic hospitalization first gets an internal trajectory: an acuity
(organ-support) level over time, four organ-failure flags, and a terminal
outcome. This spine is the **sole cross-table channel** — every downstream table
generator reads ``(spine, param_pack, rng)`` and never another table's output,
which is what pairs vasopressors with hypotension, sedation with IMV, and prone
with severe hypoxemia while keeping the generators decoupled in code.

The spine is *not* a CLIF table. It is optionally retained as ``_truth.parquet``
for benchmarking (free ground-truth acuity/flag/outcome labels).

Everything here is sampled offline from the parameter pack the fit stage (U5)
emits — no real data is present. The pack's ``spine`` block supplies every input:

* ``support_level_start_dist`` — the initial-state law.
* ``support_level_transition_matrix`` — the embedded jump chain with an absorbing
  ``discharge`` exit, so trajectories terminate naturally.
* ``support_level_sojourn`` — per-level dwell-time family.
* ``expired_rate_by_peak_level`` / ``outcome_marginal`` — mortality coupled to
  peak acuity (falling back to the cohort marginal when a peak level was gated).
* ``flag_prevalence_by_level`` — per-level organ-failure flag prevalences.

A fixed ``numpy.random.Generator`` makes a sampled spine reproducible
byte-for-byte (R22): the trajectory, then per-run flags, then the outcome are
all drawn from that one generator in a fixed order.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.estimators import DISCHARGE_STATE
from clifforge.fit.param_pack import ParamPack
from clifforge.generate import semimarkov
from clifforge.generate.semimarkov import SojournSampler

__all__ = [
    "FLAG_NAMES",
    "SpineFrame",
    "sample_spine",
    "truth_frame",
]

#: The four organ-failure flags carried per interval, in a stable order.
FLAG_NAMES: tuple[str, ...] = ("resp_flag", "cv_flag", "renal_flag", "neuro_flag")


@dataclass(frozen=True)
class SpineFrame:
    """One hospitalization's latent trajectory (per-interval arrays + outcome).

    ``support_level`` and the four flag lists are all length ``n_intervals`` and
    aligned interval-by-interval. ``outcome`` is ``"expired"`` or ``"alive"``.
    """

    hospitalization_id: str
    support_level: list[int]
    resp_flag: list[bool]
    cv_flag: list[bool]
    renal_flag: list[bool]
    neuro_flag: list[bool]
    outcome: str

    @property
    def n_intervals(self) -> int:
        return len(self.support_level)

    @property
    def peak_level(self) -> int:
        return max(self.support_level) if self.support_level else 0

    def to_polars(self) -> pl.DataFrame:
        """Long-format frame: one row per interval, plus scalar labels broadcast."""
        n = self.n_intervals
        return pl.DataFrame(
            {
                "hospitalization_id": [self.hospitalization_id] * n,
                "interval_idx": list(range(n)),
                "support_level": self.support_level,
                "resp_flag": self.resp_flag,
                "cv_flag": self.cv_flag,
                "renal_flag": self.renal_flag,
                "neuro_flag": self.neuro_flag,
                "outcome": [self.outcome] * n,
            }
        )


def _spine_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'spine' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _int_key_dist(dist: dict[str, float]) -> dict[Hashable, float]:
    return {int(level): float(prob) for level, prob in dist.items()}


def _transitions(matrix: dict[str, dict[str, float]]) -> dict[Hashable, dict[Hashable, float]]:
    """Parse the string-keyed pack matrix into int levels + the discharge exit."""
    out: dict[Hashable, dict[Hashable, float]] = {}
    for frm, row in matrix.items():
        parsed: dict[Hashable, float] = {}
        for to, prob in row.items():
            key: Hashable = DISCHARGE_STATE if to == DISCHARGE_STATE else int(to)
            parsed[key] = float(prob)
        out[int(frm)] = parsed
    return out


def _referenced_levels(
    start_dist: dict[Hashable, float], transitions: dict[Hashable, dict[Hashable, float]]
) -> set[int]:
    """Every integer support level the trajectory could dwell in."""
    levels: set[int] = {k for k in start_dist if isinstance(k, int)}
    for frm, row in transitions.items():
        if isinstance(frm, int):
            levels.add(frm)
        levels.update(to for to in row if isinstance(to, int))
    return levels


def _sojourn_samplers(
    sojourn_block: dict[str, dict[str, Any]], levels: set[int]
) -> dict[Hashable, SojournSampler]:
    """A dwell-time sampler for every referenced level.

    A level whose sojourn was gated out (n < 20) has no fitted family; it falls
    back to an exponential whose mean is the average of the fitted levels'
    means, so a reachable-but-unfit level can still be dwelt in rather than
    raising. If no level was fit at all, the fallback is a unit-mean exponential.
    """
    samplers: dict[Hashable, SojournSampler] = {}
    fitted_means: list[float] = []
    for level_str, fit in sojourn_block.items():
        samplers[int(level_str)] = semimarkov.make_sojourn_sampler(fit["family"], fit["params"])
        mean = fit.get("mean_hours")
        if isinstance(mean, int | float) and mean > 0:
            fitted_means.append(float(mean))

    fallback_mean = float(np.mean(fitted_means)) if fitted_means else 1.0
    fallback = semimarkov.make_sojourn_sampler("empirical_mean", [fallback_mean])
    for level in levels:
        samplers.setdefault(level, fallback)
    return samplers


def _visits_to_intervals(
    visits: list[semimarkov.Visit], grid_step_hours: float, horizon_intervals: int
) -> list[tuple[int, int]]:
    """Expand (level, hours) visits into per-run (level, interval_count) pairs.

    A sub-grid dwell still occupies at least one interval; the whole timeline is
    capped at ``horizon_intervals`` so rounding can never overrun the fit grid.
    The absorbing ``discharge`` visit (a non-int state) contributes nothing.
    """
    runs: list[tuple[int, int]] = []
    total = 0
    for visit in visits:
        if not isinstance(visit.state, int):  # the discharge terminal marker
            continue
        n_int = max(1, round(visit.duration / grid_step_hours))
        if total + n_int > horizon_intervals:
            n_int = horizon_intervals - total
        if n_int <= 0:
            break
        runs.append((visit.state, n_int))
        total += n_int
        if total >= horizon_intervals:
            break
    return runs


def sample_spine(
    pack: ParamPack, rng: np.random.Generator, *, hospitalization_id: str = "H0"
) -> SpineFrame:
    """Sample one hospitalization's latent spine from the pack (KTD-6, R22).

    Draws, in order from ``rng``: the support-level trajectory (semi-Markov),
    then each run's organ-failure flags (Bernoulli at the run's per-level
    prevalence, held constant across the run for within-run coherence), then the
    terminal outcome (Bernoulli at the mortality rate for the trajectory's peak
    acuity). Same seed in -> identical :class:`SpineFrame` out.
    """
    params = _spine_params(pack)
    state_model = params["state_model"]
    grid_step_hours = float(state_model["grid_step_hours"])
    horizon_intervals = int(state_model["horizon_intervals"])
    horizon_hours = horizon_intervals * grid_step_hours

    start_dist = _int_key_dist(params["support_level_start_dist"])
    transitions = _transitions(params["support_level_transition_matrix"])
    levels = _referenced_levels(start_dist, transitions)
    sojourns = _sojourn_samplers(params["support_level_sojourn"], levels)
    flag_prevalence: dict[str, dict[str, float]] = params["flag_prevalence_by_level"]

    visits = semimarkov.sample(
        transitions,
        sojourns,
        start_dist,
        {DISCHARGE_STATE},
        rng,
        horizon_hours,
    )
    runs = _visits_to_intervals(visits, grid_step_hours, horizon_intervals)

    support_level: list[int] = []
    flags: dict[str, list[bool]] = {name: [] for name in FLAG_NAMES}
    for level, n_int in runs:
        prevalence = flag_prevalence.get(str(level), {})
        # One draw per flag per run, broadcast across the run's intervals.
        run_flags = {
            name: bool(rng.random() < float(prevalence.get(name, 0.0))) for name in FLAG_NAMES
        }
        support_level.extend([level] * n_int)
        for name in FLAG_NAMES:
            flags[name].extend([run_flags[name]] * n_int)

    peak = max(support_level) if support_level else 0
    outcome = _sample_outcome(params, peak, rng)

    return SpineFrame(
        hospitalization_id=hospitalization_id,
        support_level=support_level,
        resp_flag=flags["resp_flag"],
        cv_flag=flags["cv_flag"],
        renal_flag=flags["renal_flag"],
        neuro_flag=flags["neuro_flag"],
        outcome=outcome,
    )


def _sample_outcome(params: dict[str, Any], peak_level: int, rng: np.random.Generator) -> str:
    """Expired/alive, coupled to peak acuity with a cohort-marginal fallback."""
    by_peak: dict[str, dict[str, float]] = params.get("expired_rate_by_peak_level", {})
    peak_cell = by_peak.get(str(peak_level))
    if peak_cell is not None:
        expired_rate = float(peak_cell["expired_rate"])
    else:
        expired_rate = float(params.get("outcome_marginal", {}).get("expired", 0.0))
    return "expired" if rng.random() < expired_rate else "alive"


def truth_frame(spines: list[SpineFrame]) -> pl.DataFrame:
    """Stack sampled spines into one long ``_truth`` frame for benchmarking."""
    if not spines:
        return pl.DataFrame(
            schema={
                "hospitalization_id": pl.String,
                "interval_idx": pl.Int64,
                "support_level": pl.Int64,
                "resp_flag": pl.Boolean,
                "cv_flag": pl.Boolean,
                "renal_flag": pl.Boolean,
                "neuro_flag": pl.Boolean,
                "outcome": pl.String,
            }
        )
    return pl.concat([s.to_polars() for s in spines], how="vertical")

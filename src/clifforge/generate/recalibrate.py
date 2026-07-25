"""Spine acuity recalibration (population-shaping helper).

The fitted semi-Markov spine over-mixes into high-acuity states — a known
first-order-fidelity artifact (documented at U6: real peak-level marginal
L4=0.342 vs sampled ~0.587). Because the ordinal ``support_level`` *is* the
respiratory-support ladder, that over-representation shows up directly as too much
IMV / too little low-flow oxygen in the derived ``respiratory_support`` and ``adt``
tables, capping their fidelity against a real ICU reference.

:func:`recalibrate_spine_acuity` returns a **new pack** whose spine spends less
time in the highest-acuity states, by moving a fraction of each high-level
transition's mass to ``discharge`` (and proportionally lowering the high-level
start probabilities). It changes only the spine block; the per-state physiology,
lab copula, and med marginals are untouched.

Measured effect (Chicago pack, 3k stays, vs the ICU real cohort): with
``deescalation=0.45`` respiratory_support fidelity rises ~0.64 -> ~0.76 and adt
~0.84 -> ~0.87, while vitals / labs / medication fidelity move < 0.005 — the
per-state value distributions barely shift because only the *time weighting* over
states changes, not the fitted parameters. This makes it a low-cost lever for
shaping a more realistic acuity mix, applied to a *derived* pack (the base fitted
pack is left as-fit).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from clifforge.fit.param_pack import ParamPack

__all__ = ["recalibrate_spine_acuity"]

_DISCHARGE = "discharge"


def recalibrate_spine_acuity(
    pack: ParamPack,
    *,
    deescalation: float = 0.45,
    high_levels: Iterable[int] = (3, 4, 5),
) -> ParamPack:
    """Return a pack whose spine de-escalates high-acuity states (population shaping).

    ``deescalation`` in ``[0, 1)`` is the fraction of each high level's
    non-discharge transition mass redirected to ``discharge``; the high levels'
    start probabilities are scaled by ``(1 - deescalation)`` and the start
    distribution renormalized. ``0.0`` returns an equivalent pack unchanged.
    """
    if not 0.0 <= deescalation < 1.0:
        raise ValueError("deescalation must be in [0, 1)")
    keep = 1.0 - deescalation
    high = {str(level) for level in high_levels}

    tables = copy.deepcopy(dict(pack.tables))
    spine = tables.get("spine")
    if spine is None or "params" not in spine:
        raise ValueError("pack has no 'spine' block to recalibrate")
    params = spine["params"]

    matrix = params.get("support_level_transition_matrix", {})
    for level, row in matrix.items():
        if level not in high:
            continue
        moved = sum(prob for target, prob in row.items() if target != _DISCHARGE) * deescalation
        for target in list(row):
            if target != _DISCHARGE:
                row[target] *= keep
        row[_DISCHARGE] = row.get(_DISCHARGE, 0.0) + moved

    start = params.get("support_level_start_dist", {})
    for level in list(start):
        if level in high:
            start[level] *= keep
    total = sum(start.values())
    if total > 0:
        params["support_level_start_dist"] = {k: v / total for k, v in start.items()}

    return ParamPack(manifest=dict(pack.manifest), tables=tables)

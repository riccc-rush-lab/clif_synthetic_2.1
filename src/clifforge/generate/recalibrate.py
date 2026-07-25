"""Population recalibration: reshape a MIMIC-fitted pack to a target ICU cohort.

The raw MIMIC-IV fit is faithful to MIMIC, but MIMIC is one high-acuity academic
center: its fitted spine over-concentrates trajectories at ventilator acuity
(sampled peak level 4 ~0.59 vs ~0.34 in the real per-hospitalization peak the fit
records) and terminates them too quickly (sampled ICU LOS ~28h vs the ~47h real
median). That single distortion cascades: it inflates invasive-ventilation and
mortality prevalence, and — because every time-series table (vitals, labs,
medications, devices) is driven by the spine's interval count — it starves each
stay of the measurements a real multi-day ICU course accumulates, leaving stays
~13x too sparse.

:func:`recalibrate_to_network_median` applies one principled transform whose
targets are all *measured real quantities*, not free knobs:

* **Temper escalation, don't discharge.** High-acuity transition mass (to levels
  3-5) and high-acuity start mass are re-routed to level 2 (ICU, non-intubated —
  the "sick but surviving" zone), and premature ward discharge is damped. This
  lowers the fraction reaching invasive ventilation and lowers peak-coupled
  mortality *without shortening the stay* — the opposite of trimming the
  trajectory to hit a rate.
* **Scale sojourns to real LOS.** Level dwell-times are scaled (on the fitted
  distribution's scale parameter, preserving family and shape) so hospital and
  ICU length-of-stay medians match the real cohort — which restores per-stay
  measurement density as a side effect.
* **Scale peak mortality to the target network rate.**
* **Decouple the organ-failure axes** so cardiovascular / renal support
  prevalence (vasopressors, CRRT) track their own real rates rather than the
  respiratory acuity distribution, and **enable the length-aware generator
  paths** (non-invasive level-2 escalation, per-stay vasopressor confinement,
  denser whole-stay lab panels) that keep life-support *prevalence* a per-stay
  property once stays are realistically long.

The defaults target the CLIF network median (in-hospital mortality ~9.5%,
invasive ventilation ~41%, ICU LOS ~47h) rather than MIMIC's own higher-acuity
figures, so the output represents a generic US ICU and is statistically distinct
from its MIMIC source while remaining in the real cross-site envelope. All spine
edits operate on a deep copy; the input pack is never mutated (R22).
"""

from __future__ import annotations

import copy
from typing import Any

from clifforge.fit.estimators import DISCHARGE_STATE
from clifforge.fit.param_pack import ParamPack
from clifforge.reference import bounds

__all__ = ["recalibrate_to_network_median", "repair_vitals_dispersion"]

#: High-acuity (invasive-ventilation) levels whose start/escalation mass is tempered.
_HIGH_LEVELS: tuple[str, ...] = ("3", "4", "5")

#: Robust within-physiologic-bounds dispersion (1.4826 x MAD) per vital, measured
#: on real CLIF-MIMIC. The fitted AR(1) ``sigma`` is corrupted by extreme charting
#: artifacts (e.g. MAP sigma ~4000 mmHg), so the generated walk otherwise pins
#: values to the outlier-clamp bounds; these aggregate real dispersions replace it.
#: (The estimator fix in ``fit._fit_ar1`` prevents this in future fits; this repairs
#: packs already fitted with the un-robust estimator.)
#: A fitted per-state σ above this multiple of the robust real dispersion is treated
#: as a corrupted (pre-robust-estimator) value and replaced; physiologic re-fit σ
#: sits at ~1× and is left untouched, preserving its heteroscedasticity.
_CORRUPT_SIGMA_FACTOR: float = 3.0

_ROBUST_VITAL_SD: dict[str, float] = {
    "heart_rate": 17.79,
    "sbp": 22.24,
    "dbp": 13.34,
    "map": 14.83,
    "respiratory_rate": 5.93,
    "spo2": 2.97,
    "temp_c": 0.44,
}


def _renormalize(dist: dict[str, float]) -> None:
    """In-place renormalize a probability dict to sum to 1 (no-op if empty/zero)."""
    total = sum(dist.values())
    if total > 0:
        for key in dist:
            dist[key] /= total


def _temper_transitions(
    matrix: dict[str, dict[str, float]], *, esc_keep: float, disch_damp: float
) -> None:
    """Re-route escalation and premature-discharge mass toward level-2 ICU dwell.

    ``esc_keep`` is the fraction of each row's mass toward high levels that is
    kept; the rest flows to level 2 (0.8) and level 1 (0.2) as recovery/de-
    escalation. ``disch_damp`` multiplies the discharge probability (< 1 lengthens
    stays); the freed mass returns to level 1. Rows are renormalized in place.
    """
    for row in matrix.values():
        moved = 0.0
        for level in _HIGH_LEVELS:
            if level in row:
                shed = row[level] * (1.0 - esc_keep)
                row[level] -= shed
                moved += shed
        row["2"] = row.get("2", 0.0) + moved * 0.8
        row["1"] = row.get("1", 0.0) + moved * 0.2
        if DISCHARGE_STATE in row:
            disch = row[DISCHARGE_STATE]
            freed = disch * (1.0 - disch_damp)
            row[DISCHARGE_STATE] = disch * disch_damp
            row["1"] = row.get("1", 0.0) + freed
        _renormalize(row)


def _temper_start(start_dist: dict[str, float], *, start_shift: float) -> None:
    """Shift a fraction of high-acuity start mass down to level 2 (ICU, non-vent)."""
    moved = sum(start_dist.get(level, 0.0) for level in _HIGH_LEVELS) * start_shift
    for level in _HIGH_LEVELS:
        if level in start_dist:
            start_dist[level] *= 1.0 - start_shift
    start_dist["2"] = start_dist.get("2", 0.0) + moved
    _renormalize(start_dist)


def _scale_sojourns(sojourn: dict[str, dict[str, Any]], multipliers: dict[str, float]) -> None:
    """Scale each level's dwell-time mean by multiplying the fitted scale parameter.

    Both fitted families (scipy ``lognorm`` / ``weibull_min``) carry their scale in
    ``params[2]`` with the location at ``params[1]``; the mean is proportional to
    that scale, so multiplying it scales the mean while preserving the fitted
    family and shape. ``mean_hours`` (metadata) is kept consistent.
    """
    for level, mult in multipliers.items():
        block = sojourn.get(level)
        if block is None:
            continue
        params = block.get("params")
        if isinstance(params, list) and len(params) >= 3:
            params[2] *= mult
        mean = block.get("mean_hours")
        if isinstance(mean, int | float):
            block["mean_hours"] = mean * mult


def repair_vitals_dispersion(tables: dict[str, Any]) -> None:
    """Repair *corrupted* AR(1) ``sigma`` and clamp state means into physiologic range.

    A legacy pack fitted with the pre-robust estimator can carry an implausible
    ``sigma`` (e.g. a MAP σ of ~4000 mmHg) that pins the generated walk to the
    outlier bounds. This replaces **only** such corrupted values — a per-state σ
    above ``_CORRUPT_SIGMA_FACTOR`` × the robust real dispersion — with that robust
    value, and clamps impossible means (e.g. an SpO2 of 108) into bounds. A pack
    from the robust re-fit already carries physiologic, *heteroscedastic* per-state
    σ, so this is a no-op on it and its state-dependent variance is preserved
    (KTD2). Operates in place on a pack's ``tables`` dict.
    """
    block = tables.get("vitals")
    if not isinstance(block, dict) or "params" not in block:
        return
    params = block["params"]
    for vital, robust_sd in _ROBUST_VITAL_SD.items():
        by_state = params.get(f"{vital}_ar1_by_state")
        if not isinstance(by_state, dict):
            continue
        lower, upper = bounds("vitals", vital)
        for cell in by_state.values():
            if float(cell["sigma"]) > _CORRUPT_SIGMA_FACTOR * robust_sd:
                cell["sigma"] = robust_sd  # corrupted fit — replace
            cell["mean"] = min(max(float(cell["mean"]), lower), upper)


def recalibrate_to_network_median(
    pack: ParamPack,
    *,
    esc_keep: float = 0.13,
    start_shift: float = 0.72,
    disch_damp: float = 0.55,
    sojourn_multipliers: dict[str, float] | None = None,
    mortality_scale: float = 0.74,
    flag_target_prevalence: dict[str, float] | None = None,
    prone_prob_severe: float = 0.035,
    vasopressor_cv_boost: float = 3.0,
    lab_panel_interval_hours: float = 12.0,
    lab_ward_panel_interval_hours: float = 18.0,
    terminal_deterioration_hours: float = 24.0,
    crrt_prob: float = 0.29,
    repair_vitals: bool = True,
) -> ParamPack:
    """Return a deep-copied pack recalibrated to the CLIF network-median ICU cohort.

    The defaults were tuned against real CLIF-MIMIC so the generated cohort's
    length-of-stay, organ-support prevalences, and per-stay measurement density
    land in the real cross-site range (see module docstring). Every argument is a
    documented lever over that transform; the input pack is not mutated.
    """
    tables = copy.deepcopy(dict(pack.tables))
    spine = tables["spine"]["params"]

    _temper_transitions(
        spine["support_level_transition_matrix"], esc_keep=esc_keep, disch_damp=disch_damp
    )
    _temper_start(spine["support_level_start_dist"], start_shift=start_shift)
    _scale_sojourns(
        spine["support_level_sojourn"],
        sojourn_multipliers or {"1": 3.4, "2": 2.2, "3": 1.8, "4": 1.8},
    )
    for cell in spine["expired_rate_by_peak_level"].values():
        cell["expired_rate"] = min(1.0, cell["expired_rate"] * mortality_scale)

    # Outcome-coupled terminal deterioration: expiring stays decline into
    # multi-organ failure over their final window (see spine._apply_...).
    spine["terminal_deterioration_hours"] = terminal_deterioration_hours

    if repair_vitals:
        repair_vitals_dispersion(tables)

    # Decoupled organ axes: each failure flag drawn once per stay at a marginal
    # target, active across ICU time (see spine.sample_spine).
    spine["flag_target_prevalence"] = flag_target_prevalence or {
        "resp_flag": 0.5,
        "cv_flag": 0.30,
        "renal_flag": 0.05,
        "neuro_flag": 0.2,
    }

    # Length-aware generator paths: keep life-support *prevalence* a per-stay
    # property and restore lab volume once stays are realistically long.
    tables["adt"] = {"params": {"enrich_locations": False}}
    tables["respiratory_support"] = {
        "params": {"enrich_devices": True, "l2_resp_noninvasive": True}
    }
    med = dict(tables.get("medication_admin_continuous", {}))
    med_params = dict(med.get("params", {}))
    med_params["vasopressor_per_stay"] = True
    med_params["vasopressor_cv_boost"] = vasopressor_cv_boost
    med["params"] = med_params
    tables["medication_admin_continuous"] = med
    labs = dict(tables.get("labs", {}))
    lab_params = dict(labs.get("params", {}))
    lab_params["panel_interval_hours"] = lab_panel_interval_hours
    lab_params["ward_panel_interval_hours"] = lab_ward_panel_interval_hours
    labs["params"] = lab_params
    tables["labs"] = labs
    # Renal failure drives creatinine/BUN up in many stays (incl. every terminal
    # decline); only a fraction are dialyzed, so CRRT is gated to its real rate.
    tables["crrt_therapy"] = {"params": {"crrt_prob": crrt_prob}}
    tables["position"] = {
        "params": {"prone_prob_severe": prone_prob_severe, "prone_prob_otherwise": 0.001}
    }

    return ParamPack(manifest=dict(pack.manifest), tables=tables)

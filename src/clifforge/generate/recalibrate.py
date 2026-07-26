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

* **Temper escalation and realign the peak-acuity shape.** High-acuity transition
  mass (to levels 3-5) is tempered toward recovery, and the peak-acuity
  distribution is shaped through the start law to an ICU-conditioned, real-shape
  target at the network-median ventilation rate (see :func:`_network_median_peak_target`)
  — so the fraction reaching invasive ventilation and peak-coupled mortality land
  at the network median *without shortening the stay* and with a realistically
  spread acuity mix, the opposite of trimming the trajectory to hit a rate.
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
_ROBUST_VITAL_SD: dict[str, float] = {
    "heart_rate": 17.79,
    "sbp": 22.24,
    "dbp": 13.34,
    "map": 14.83,
    "respiratory_rate": 5.93,
    "spo2": 2.97,
    "temp_c": 0.44,
}

#: A fitted per-state σ above this multiple of the robust real dispersion is treated
#: as a corrupted (pre-robust-estimator) value and replaced; physiologic re-fit σ
#: sits at ~1× and is left untouched, preserving its heteroscedasticity.
_CORRUPT_SIGMA_FACTOR: float = 3.0


def _renormalize(dist: dict[str, float]) -> None:
    """In-place renormalize a probability dict to sum to 1 (no-op if empty/zero)."""
    total = sum(dist.values())
    if total > 0:
        for key in dist:
            dist[key] /= total


def _temper_transitions(
    matrix: dict[str, dict[str, float]], *, esc_keep: float, disch_damp: float
) -> None:
    """Temper high-acuity escalation and premature discharge.

    ``esc_keep`` is the fraction of each row's mass toward high levels that is
    kept; the rest flows to level 1 (0.7) as recovery/de-escalation and level 2
    (0.3), so the peak-acuity distribution is driven by the start law (shaped by
    :func:`_apply_peak_target`) rather than by escalation. ``disch_damp`` multiplies
    the discharge probability (< 1 lengthens stays); the freed mass returns to
    level 1. Rows are renormalized in place.
    """
    for row in matrix.values():
        moved = 0.0
        for level in _HIGH_LEVELS:
            if level in row:
                shed = row[level] * (1.0 - esc_keep)
                row[level] -= shed
                moved += shed
        row["1"] = row.get("1", 0.0) + moved * 0.7  # de-escalate toward recovery (L1)
        row["2"] = row.get("2", 0.0) + moved * 0.3
        if DISCHARGE_STATE in row:
            disch = row[DISCHARGE_STATE]
            freed = disch * (1.0 - disch_damp)
            row[DISCHARGE_STATE] = disch * disch_damp
            row["1"] = row.get("1", 0.0) + freed
        _renormalize(row)


def _real_peak_profile(spine_params: dict[str, Any]) -> dict[str, float]:
    """Real per-hospitalization peak-level distribution, from the fit's peak counts."""
    by_peak = spine_params.get("expired_rate_by_peak_level", {})
    counts = {k: float(v.get("n_hospitalizations", 0)) for k, v in by_peak.items()}
    total = sum(counts.values()) or 1.0
    return {k: n / total for k, n in counts.items()}


def _network_median_peak_target(real: dict[str, float], imv_target: float) -> dict[str, float]:
    """ICU-conditioned network-median peak-level target.

    The dataset is an **ICU** cohort, so every stay peaks at the ICU floor (level 2)
    or above — the full-cohort real profile's large level-0/1 mass is non-ICU floor
    stays and must not be targeted. This puts the non-ventilated ICU mass at level 2
    (``1 - imv_target``) and spreads the ventilated mass (``imv_target``, the
    network-median reaches-IMV rate) across levels 3-5 in the **real** high-acuity
    proportions — which corrects the earlier shape's under-representation of the
    highest level while holding the ventilation rate.
    """
    high = {k: v for k, v in real.items() if int(k) >= 3}
    hs = sum(high.values()) or 1.0
    target = {k: imv_target * v / hs for k, v in high.items()}
    target["2"] = 1.0 - imv_target  # ICU floor: non-ventilated ICU stays peak at L2
    return target


def _apply_peak_target(start_dist: dict[str, float], target: dict[str, float]) -> None:
    """Shape the start distribution to the target peak profile (in place).

    With escalation heavily tempered the trajectory's peak tracks its start level,
    so setting the start law to the desired peak profile realigns the sampled
    peak-acuity distribution across levels in real proportions instead of piling
    it at level 2 (which the earlier level-2 routing produced).
    """
    start_dist.clear()
    for level, prob in target.items():
        start_dist[str(level)] = float(prob)
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


def _expected_mortality(
    peak_dist: dict[str, float],
    expired_by_peak: dict[str, dict[str, float]],
    marginal_expired: float,
    scale: float,
) -> float:
    """Analytic cohort mortality under a peak distribution and a mortality scale.

    Because escalation is heavily tempered, a stay's peak acuity tracks its start
    level, so ``peak_dist`` (the start law) approximates the sampled peak
    distribution. Each level contributes its (scaled, rate-capped) expired rate.
    """
    e = 0.0
    for level, prob in peak_dist.items():
        cell = expired_by_peak.get(str(level))
        rate = float(cell["expired_rate"]) if cell else marginal_expired
        e += prob * min(1.0, rate * scale)
    return e


def _solve_mortality_scale(
    peak_dist: dict[str, float],
    expired_by_peak: dict[str, dict[str, float]],
    marginal_expired: float,
    target: float,
) -> float:
    """Bisection for the mortality scale that lands cohort mortality on ``target``.

    Monotone in ``scale`` (each term is non-decreasing), so bisection converges.
    Falls back to the upper bracket when the target exceeds what capping allows.
    """
    lo, hi = 0.0, 100.0
    if _expected_mortality(peak_dist, expired_by_peak, marginal_expired, hi) < target:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _expected_mortality(peak_dist, expired_by_peak, marginal_expired, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


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
    esc_keep: float = 0.04,
    peak_imv_target: float = 0.28,
    peak_target: dict[str, float] | None = None,
    disch_damp: float = 0.55,
    sojourn_multipliers: dict[str, float] | None = None,
    mortality_scale: float = 0.66,
    mortality_target: float | None = None,
    flag_target_prevalence: dict[str, float] | None = None,
    prone_prob_severe: float = 0.026,
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
    # Realign the peak-acuity distribution to a realistic (real-shape) profile at the
    # network-median ventilation rate, driven through the start law (peak tracks start
    # once escalation is tempered) instead of piling peak mass at level 2.
    target = peak_target or _network_median_peak_target(_real_peak_profile(spine), peak_imv_target)
    _apply_peak_target(spine["support_level_start_dist"], target)
    _scale_sojourns(
        spine["support_level_sojourn"],
        sojourn_multipliers or {"1": 3.2, "2": 4.6, "3": 2.8, "4": 2.8},
    )
    # Mortality: scale peak-coupled expired rates. ``mortality_target`` (an exact
    # cohort in-hospital mortality) takes precedence — the scale is solved
    # analytically against the peak-target distribution — otherwise the fixed
    # ``mortality_scale`` is applied.
    if mortality_target is not None:
        eff_mortality_scale = _solve_mortality_scale(
            target,
            spine["expired_rate_by_peak_level"],
            float(spine.get("outcome_marginal", {}).get("expired", 0.0)),
            mortality_target,
        )
    else:
        eff_mortality_scale = mortality_scale
    for cell in spine["expired_rate_by_peak_level"].values():
        cell["expired_rate"] = min(1.0, cell["expired_rate"] * eff_mortality_scale)

    # Outcome-coupled terminal deterioration: expiring stays decline into
    # multi-organ failure over their final window (see spine._apply_...).
    spine["terminal_deterioration_hours"] = terminal_deterioration_hours

    if repair_vitals:
        repair_vitals_dispersion(tables)

    # Decoupled organ axes: each failure flag drawn once per stay at a marginal
    # target, active across ICU time (see spine.sample_spine).
    spine["flag_target_prevalence"] = flag_target_prevalence or {
        "resp_flag": 0.5,
        "cv_flag": 0.27,
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

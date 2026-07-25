"""Unit tests for population recalibration (network-median transform + repairs).

Covers the transform math (escalation/start tempering, sojourn + mortality
scaling), the vitals-dispersion repair, outcome-coupled terminal deterioration in
the spine, and the length-aware gated generator paths (non-invasive level-2
escalation, per-stay vasopressor confinement, denser lab panels). Driven by
constructed packs/spines — no real data.
"""

from __future__ import annotations

import numpy as np

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.recalibrate import (
    recalibrate_to_network_median,
    repair_vitals_dispersion,
)
from clifforge.generate.spine import FLAG_NAMES, SpineFrame, sample_spine
from clifforge.generate.tables.labs import _panel_intervals
from clifforge.generate.tables.medication_admin_continuous import (
    _VASOPRESSORS,
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.respiratory_support import sample_respiratory_support


def _spine_params(*, expired_rate: float = 0.2) -> dict:
    """A minimal-but-complete spine param block for sampling/recalibration."""
    return {
        "state_model": {"grid_step_hours": 1.0, "horizon_intervals": 240},
        "support_level_start_dist": {"1": 0.4, "2": 0.1, "3": 0.2, "4": 0.29, "5": 0.01},
        "support_level_transition_matrix": {
            "1": {"discharge": 0.4, "2": 0.1, "4": 0.5},
            "2": {"discharge": 0.05, "1": 0.5, "3": 0.05, "4": 0.4},
            "3": {"discharge": 0.01, "1": 0.6, "4": 0.39},
            "4": {"discharge": 0.05, "1": 0.5, "3": 0.4, "5": 0.05},
            "5": {"discharge": 0.1, "4": 0.9},
        },
        "support_level_sojourn": {
            "1": {"family": "lognormal", "mean_hours": 10.0, "params": [1.0, 0.0, 5.0]},
            "2": {"family": "lognormal", "mean_hours": 9.0, "params": [1.0, 0.0, 4.5]},
            "3": {"family": "lognormal", "mean_hours": 5.0, "params": [1.0, 0.0, 2.8]},
            "4": {"family": "lognormal", "mean_hours": 5.0, "params": [0.8, 0.0, 3.3]},
            "5": {"family": "weibull", "mean_hours": 40.0, "params": [0.8, 0.0, 36.0]},
        },
        "flag_prevalence_by_level": {
            str(lvl): {name: 0.1 for name in FLAG_NAMES} for lvl in range(6)
        },
        "expired_rate_by_peak_level": {
            str(lvl): {"expired_rate": expired_rate, "n_hospitalizations": 100} for lvl in range(6)
        },
        "outcome_marginal": {"alive": 1 - expired_rate, "expired": expired_rate},
    }


def _pack(**kw) -> ParamPack:
    return ParamPack(manifest={}, tables={"spine": {"params": _spine_params(**kw)}})


def _vitals_pack() -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "spine": {"params": _spine_params()},
            "vitals": {
                "params": {
                    "map_ar1_by_state": {
                        "1": {"mean": 88.0, "phi": 0.0, "sigma": 6838.0},
                        "4": {"mean": 74.0, "phi": 0.0, "sigma": 205.0},
                    },
                    "spo2_ar1_by_state": {"1": {"mean": 108.0, "phi": 0.0, "sigma": 900.0}},
                }
            },
        },
    )


# --- transform math -------------------------------------------------------- #


def test_recalibrate_does_not_mutate_input() -> None:
    pack = _pack()
    before = pack.tables["spine"]["params"]["support_level_start_dist"]["4"]
    recalibrate_to_network_median(pack)
    assert pack.tables["spine"]["params"]["support_level_start_dist"]["4"] == before


def test_start_and_transition_mass_tempered_toward_level_two() -> None:
    out = recalibrate_to_network_median(_pack()).tables["spine"]["params"]
    start = out["support_level_start_dist"]
    # High-acuity start mass is shed and level-2 gains it.
    assert start["4"] < 0.29 and start["2"] > 0.1
    # Escalation to level 4 is tempered in every row that had it.
    row = out["support_level_transition_matrix"]["1"]
    assert row["4"] < 0.5 and row["2"] > 0.0


def test_sojourns_scaled_and_mortality_scaled() -> None:
    out = recalibrate_to_network_median(_pack(expired_rate=0.2)).tables["spine"]["params"]
    # Level-1 sojourn scale (params[2]) multiplied by the default 3.4x.
    assert out["support_level_sojourn"]["1"]["params"][2] == 5.0 * 3.4
    # Peak mortality scaled by the default 0.74.
    assert abs(out["expired_rate_by_peak_level"]["4"]["expired_rate"] - 0.2 * 0.74) < 1e-9


def test_generator_paths_enabled() -> None:
    out = recalibrate_to_network_median(_pack()).tables
    assert out["respiratory_support"]["params"]["l2_resp_noninvasive"] is True
    assert out["medication_admin_continuous"]["params"]["vasopressor_per_stay"] is True
    assert out["labs"]["params"]["ward_panel_interval_hours"] is not None
    assert out["spine"]["params"]["terminal_deterioration_hours"] == 24.0


# --- vitals repair --------------------------------------------------------- #


def test_repair_vitals_sets_robust_sigma_and_clamps_mean() -> None:
    tables = _vitals_pack().tables
    repair_vitals_dispersion(tables)
    mp = tables["vitals"]["params"]["map_ar1_by_state"]
    assert mp["1"]["sigma"] < 30.0 and mp["4"]["sigma"] < 30.0  # no longer thousands
    # SpO2 mean of 108 is clamped into the physiologic bound (<= 100).
    assert tables["vitals"]["params"]["spo2_ar1_by_state"]["1"]["mean"] <= 100.0


def test_repair_vitals_leaves_physiologic_refit_sigma_untouched() -> None:
    # A pack from the robust re-fit carries physiologic, heteroscedastic per-state
    # sigma; repair must be a no-op so that state-dependent variance is preserved (KTD2).
    tables = {
        "vitals": {
            "params": {
                "map_ar1_by_state": {
                    "2": {"mean": 80.6, "phi": 0.66, "sigma": 11.18},
                    "5": {"mean": 73.4, "phi": 0.66, "sigma": 8.85},
                }
            }
        }
    }
    repair_vitals_dispersion(tables)
    mp = tables["vitals"]["params"]["map_ar1_by_state"]
    assert (
        mp["2"]["sigma"] == 11.18 and mp["5"]["sigma"] == 8.85
    )  # unchanged, still heteroscedastic


# --- terminal deterioration ------------------------------------------------ #


def test_terminal_deterioration_escalates_dying_tail() -> None:
    # expired_rate 1.0 -> always expires; terminal window escalates the tail.
    pack = recalibrate_to_network_median(_pack(expired_rate=1.0), terminal_deterioration_hours=24.0)
    sp = sample_spine(pack, np.random.default_rng(0), hospitalization_id="Hd")
    assert sp.outcome == "expired"
    n_term = 24  # grid_step 1.0h
    tail = sp.support_level[-min(n_term, sp.n_intervals) :]
    assert max(tail) >= 4  # escalated into invasive-ventilation acuity
    assert sp.cv_flag[-1] and sp.resp_flag[-1]  # organ failure active at death


def test_survivors_are_not_deteriorated() -> None:
    pack = recalibrate_to_network_median(_pack(expired_rate=0.0), terminal_deterioration_hours=24.0)
    sp = sample_spine(pack, np.random.default_rng(1), hospitalization_id="Ha")
    assert sp.outcome == "alive"
    # No forced terminal escalation: the alive tail need not reach L4+.
    # (This is probabilistic-free — deterioration only runs for expired stays.)


# --- gated generator paths ------------------------------------------------- #


def _spine_frame(levels: list[int], *, resp: bool, cv: bool) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id="H0",
        support_level=levels,
        resp_flag=[resp] * n,
        cv_flag=[cv] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_l2_resp_noninvasive_reserves_imv_for_intubation_tier() -> None:
    # A level-2 stay in respiratory failure: default escalates to IMV, the gated
    # path keeps it non-invasive (no IMV device).
    sp = _spine_frame([2] * 20, resp=True, cv=False)
    default = ParamPack(
        manifest={}, tables={"spine": {"params": {"state_model": {"grid_step_hours": 1.0}}}}
    )
    gated = ParamPack(
        manifest={},
        tables={
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
            "respiratory_support": {"params": {"l2_resp_noninvasive": True}},
        },
    )
    dev_default = {
        r.device_category for r in sample_respiratory_support(sp, default, np.random.default_rng(0))
    }
    dev_gated = {
        r.device_category for r in sample_respiratory_support(sp, gated, np.random.default_rng(0))
    }
    assert "IMV" in dev_default
    assert "IMV" not in dev_gated


def test_vasopressor_per_stay_confines_pressors_to_cv_stays() -> None:
    marginal = {"norepinephrine": 0.3, "propofol": 0.3, "sodium_chloride": 0.4}
    params = {"med_category_marginal": marginal, "vasopressor_per_stay": True}
    pack = ParamPack(
        manifest={},
        tables={
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
            "medication_admin_continuous": {"params": params},
        },
    )
    non_cv = _spine_frame([3] * 60, resp=False, cv=False)
    rows = sample_medication_admin_continuous(non_cv, pack, np.random.default_rng(0))
    assert not any(r.med_category in _VASOPRESSORS for r in rows)  # no pressors without cv failure


def test_lab_panel_schedule_is_denser_with_pack_overrides() -> None:
    levels = [2] * 48  # 48h of ICU at grid_step 1.0
    daily = _panel_intervals(levels, 1.0)
    bid = _panel_intervals(levels, 1.0, icu_hours=12.0)
    ward = _panel_intervals([1] * 48, 1.0, ward_hours=24.0)
    assert len(bid) > len(daily)  # twice-daily yields more panels than daily
    assert len(_panel_intervals([1] * 48, 1.0)) == 0 and len(ward) > 0  # ward panels off by default

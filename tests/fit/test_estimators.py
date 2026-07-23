"""Estimator tests on fabricated aggregates (U5d).

Every input is synthetic; no real data path is touched (KTD-1). Tests assert the
invariants the parameter pack and downstream generator rely on: stochastic-matrix
transition rows with a zero diagonal, finite sojourn parameters, stationary AR1
coefficients, a symmetric positive-definite copula, and — critically — that a
sub-threshold (n<20) cell is suppressed rather than leaked (R2).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.fit import estimators


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #
def _cyclic_timeline(n_hosp: int = 30) -> pl.DataFrame:
    """Each hospitalization walks 0 -> 3 -> 4 -> 0 over four intervals."""
    rows = []
    for h in range(n_hosp):
        for interval, level in enumerate([0, 3, 4, 0]):
            rows.append(
                {"hospitalization_id": f"H{h}", "interval_idx": interval, "support_level": level}
            )
    return pl.DataFrame(rows)


def test_transition_rows_sum_to_one_zero_diagonal() -> None:
    params, _ = estimators.fit_transitions(_cyclic_timeline())
    matrix = params["support_level_transition_matrix"]
    assert matrix  # non-empty
    for from_level, row in matrix.items():
        assert from_level not in row  # zero diagonal (self-transition never emitted)
        assert abs(sum(row.values()) - 1.0) < 1e-9  # row-stochastic


def test_transition_below_gate_suppressed() -> None:
    # Only 5 hospitalizations -> every transition pair has n=5 < 20 -> all gated.
    params, audit = estimators.fit_transitions(_cyclic_timeline(n_hosp=5))
    assert params["support_level_transition_matrix"] == {}
    assert audit  # suppression recorded
    assert all(r.fallback_kind == "none" for r in audit)


# --------------------------------------------------------------------------- #
# Sojourns
# --------------------------------------------------------------------------- #
def test_sojourn_family_has_finite_params() -> None:
    params, _ = estimators.fit_sojourns(_cyclic_timeline(), grid_step_hours=1.0)
    sojourns = params["support_level_sojourn"]
    assert sojourns
    for fit in sojourns.values():
        assert fit["family"]
        assert all(np.isfinite(p) for p in fit["params"])
        assert np.isfinite(fit["mean_hours"])


# --------------------------------------------------------------------------- #
# Spine attributes: outcome + flags
# --------------------------------------------------------------------------- #
def _flagged_timeline(n_hosp: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Half the cohort peaks at level 4 and expires; half peaks at 1 and lives."""
    tl_rows, out_rows = [], []
    for h in range(n_hosp):
        sick = h % 2 == 0
        hid = f"H{h}"
        levels = [0, 4] if sick else [0, 1]
        for interval, level in enumerate(levels):
            tl_rows.append(
                {
                    "hospitalization_id": hid,
                    "interval_idx": interval,
                    "support_level": level,
                    "resp_flag": level >= 2,
                    "cv_flag": level >= 4,
                    "renal_flag": False,
                    "neuro_flag": level >= 4,
                }
            )
        out_rows.append({"hospitalization_id": hid, "outcome": "expired" if sick else "alive"})
    return pl.DataFrame(tl_rows), pl.DataFrame(out_rows)


def test_outcome_rates_couple_expiry_to_peak_acuity() -> None:
    timeline, outcomes = _flagged_timeline()
    params, _ = estimators.fit_outcome_rates(timeline, outcomes, min_n=20)
    marginal = params["outcome_marginal"]
    assert abs(marginal["alive"] + marginal["expired"] - 1.0) < 1e-9
    by_level = params["expired_rate_by_peak_level"]
    # Everyone peaking at level 4 expired; everyone peaking at level 1 survived.
    assert by_level["4"]["expired_rate"] == 1.0
    assert by_level["1"]["expired_rate"] == 0.0


def test_flag_prevalence_below_gate_suppressed() -> None:
    # 5 hospitalizations -> each level's interval count < 20 -> all gated out.
    timeline, _ = _flagged_timeline(n_hosp=5)
    params, audit = estimators.fit_flag_prevalence(timeline, min_n=20)
    assert params == {}
    assert audit
    assert all(r.fallback_kind == "none" for r in audit)


def test_flag_prevalence_is_probability_by_level() -> None:
    timeline, _ = _flagged_timeline()
    params, _ = estimators.fit_flag_prevalence(timeline, min_n=20)
    prevalence = params["flag_prevalence_by_level"]
    for level_prev in prevalence.values():
        for prob in level_prev.values():
            assert 0.0 <= prob <= 1.0


# --------------------------------------------------------------------------- #
# AR1
# --------------------------------------------------------------------------- #
def _ar1_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    v_rows = []
    s_rows = []
    for h in range(40):
        hid = f"H{h}"
        x = 80.0
        for interval in range(6):
            x = 82.0 + 0.7 * (x - 82.0) + rng.normal(0, 3.0)
            v_rows.append(
                {
                    "hospitalization_id": hid,
                    "interval_idx": interval,
                    "vital_category": "heart_rate",
                    "value": x,
                }
            )
            s_rows.append({"hospitalization_id": hid, "interval_idx": interval, "support_level": 3})
    return pl.DataFrame(v_rows), pl.DataFrame(s_rows)


def test_ar1_phi_is_stationary() -> None:
    vitals, timeline = _ar1_frames()
    params, _ = estimators.fit_ar1_by_state(vitals, timeline, vitals=["heart_rate"])
    fit = params["heart_rate_ar1_by_state"]["3"]
    assert -1.0 < fit["phi"] < 1.0
    assert fit["sigma"] >= 0.0
    assert np.isfinite(fit["mean"])


# --------------------------------------------------------------------------- #
# Lab copula
# --------------------------------------------------------------------------- #
def _lab_frame() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for h in range(60):
        base = rng.normal(0, 1)
        for interval in range(4):
            creat = np.expm1(1.0 + 0.5 * base + rng.normal(0, 0.2))
            lactate = np.expm1(0.8 + 0.5 * base + rng.normal(0, 0.2))
            rows.append(
                {
                    "hospitalization_id": f"H{h}",
                    "interval_idx": interval,
                    "lab_category": "creatinine",
                    "value": max(creat, 0.0),
                }
            )
            rows.append(
                {
                    "hospitalization_id": f"H{h}",
                    "interval_idx": interval,
                    "lab_category": "lactate",
                    "value": max(lactate, 0.0),
                }
            )
    return pl.DataFrame(rows)


def test_lab_copula_symmetric_positive_definite() -> None:
    params, _ = estimators.fit_lab_copula(_lab_frame(), n_hospitalizations=60)
    corr = np.asarray(params["lab_correlation"], dtype=float)
    assert corr.shape[0] == corr.shape[1] == len(params["lab_order"])
    assert np.allclose(corr, corr.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(corr) > 0)  # positive definite
    assert np.allclose(np.diag(corr), 1.0)  # unit diagonal


def test_nearest_pd_repairs_indefinite_matrix() -> None:
    indefinite = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    repaired = estimators.nearest_positive_definite_correlation(indefinite)
    assert np.allclose(repaired, repaired.T)
    assert np.all(np.linalg.eigvalsh(repaired) > 0)


# --------------------------------------------------------------------------- #
# Marginals + suppression
# --------------------------------------------------------------------------- #
def test_categorical_marginal_suppresses_rare_cell() -> None:
    # "Male" x40, "Female" x40 clear the gate; "Unknown" x3 must be suppressed.
    values = ["Male"] * 40 + ["Female"] * 40 + ["Unknown"] * 3
    df = pl.DataFrame({"sex_category": values})
    params, audit = estimators.fit_categorical_marginals(df, ["sex_category"], min_n=20)
    marginal = params["sex_category_marginal"]
    assert "Unknown" not in marginal  # rare cell not leaked
    assert set(marginal) == {"Male", "Female"}
    assert abs(sum(marginal.values()) - 1.0) < 1e-9  # renormalized
    assert any(rec.cell == ("sex_category", "Unknown") for rec in audit)


def test_continuous_marginal_emits_bounded_edges() -> None:
    df = pl.DataFrame({"age": list(range(18, 91))})
    params, _ = estimators.fit_continuous_marginals(df, ["age"], n_bins=10, min_n=20)
    edges = params["age_quantile_bin_edges"]
    assert len(edges) <= 11  # n_bins + 1, never approaches n_records
    assert edges == sorted(edges)

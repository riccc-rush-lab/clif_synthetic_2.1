"""Tests for the TSTR utility evaluation (U22, R19).

Uses two independently-seeded synthetic datasets (one standing in for "real") so
the whole train-on-A / test-on-B machinery runs in CI without real data: a finite
reproducible AUC gap, correct feature/label assembly, and the leakage guard that
recomputes holdout membership from the pack manifest split spec.
"""

from __future__ import annotations

import pytest

from clifforge.eval.tstr import (
    TstrReport,
    assert_holdout_disjoint,
    build_wide_features,
    run_tstr,
)
from clifforge.fit.param_pack import ParamPack
from clifforge.fit.run_fit import _holdout_mask
from clifforge.generate.orchestrator import generate_dataset


@pytest.fixture
def synthetic(pack: ParamPack) -> dict:
    return generate_dataset(pack, n_patients=120, seed=1).tables


@pytest.fixture
def real(pack: ParamPack) -> dict:
    # A second, differently-seeded draw stands in for real data in CI.
    return generate_dataset(pack, n_patients=120, seed=2).tables


def test_build_wide_features_shape_and_label(synthetic: dict) -> None:
    feat = build_wide_features(synthetic)
    assert feat.height == synthetic["hospitalization"].height  # one row per hospitalization
    assert "label" in feat.columns
    assert set(feat["label"].unique().to_list()) <= {0, 1}
    # label matches the Expired discharge rate
    expired = (
        synthetic["hospitalization"]
        .filter(synthetic["hospitalization"]["discharge_category"] == "Expired")
        .height
    )
    assert feat["label"].sum() == expired
    assert any(c.startswith("vital_") for c in feat.columns)
    assert any(c.startswith("lab_") for c in feat.columns)


def test_run_tstr_produces_finite_gap(synthetic: dict, real: dict) -> None:
    report = run_tstr(synthetic, real, seed=0)
    assert isinstance(report, TstrReport)
    for auc in (report.tstr_auc, report.trtr_auc):
        assert 0.0 <= auc <= 1.0
    assert report.auc_gap == pytest.approx(report.trtr_auc - report.tstr_auc)
    assert report.n_features > 0 and report.n_test_real > 0


def test_run_tstr_is_deterministic(synthetic: dict, real: dict) -> None:
    a = run_tstr(synthetic, real, seed=7)
    b = run_tstr(synthetic, real, seed=7)
    assert a == b


# --- leakage guard ----------------------------------------------------------- #
def _split_manifest(seed: int = 20260723, fraction: float = 0.2) -> dict:
    return {"split": {"method": "sha1_mod_10000", "seed": seed, "holdout_fraction": fraction}}


def _partition(patient_ids: list[str], seed: int, fraction: float) -> tuple[list[str], list[str]]:
    holdout = [p for p in patient_ids if _holdout_mask(p, seed, fraction)]
    train = [p for p in patient_ids if not _holdout_mask(p, seed, fraction)]
    return train, holdout


def test_holdout_guard_passes_for_holdout_only_patients() -> None:
    manifest = _split_manifest()
    ids = [f"P{i}" for i in range(500)]
    _train, holdout = _partition(ids, 20260723, 0.2)
    assert holdout  # the fixture must actually produce some holdout patients
    assert_holdout_disjoint(holdout, manifest)  # must not raise


def test_holdout_guard_raises_on_leaked_patient() -> None:
    manifest = _split_manifest()
    ids = [f"P{i}" for i in range(500)]
    train, holdout = _partition(ids, 20260723, 0.2)
    assert train
    with pytest.raises(ValueError, match="leakage"):
        assert_holdout_disjoint(holdout + train[:1], manifest)  # one training patient leaks in


def test_holdout_guard_raises_when_split_missing() -> None:
    with pytest.raises(ValueError, match="no 'split' spec"):
        assert_holdout_disjoint(["P0"], {"clif_version": "2.1.0"})


def test_holdout_guard_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unsupported split method"):
        assert_holdout_disjoint(
            ["P0"], {"split": {"method": "random", "seed": 1, "holdout_fraction": 0.2}}
        )

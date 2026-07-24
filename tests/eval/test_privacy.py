"""Tests for the standalone privacy metrics (U23, R20).

Two independently-seeded synthetic datasets (one standing in for "real") exercise
the metrics in CI: finite in-range values, aggregate-only cleanliness (DCR well
above zero — no memorization), a memorization control that the metrics actually
detect, determinism, and an import-containment assertion (no torch/synthcity).
"""

from __future__ import annotations

import sys

from clifforge.eval.privacy import PrivacyReport, privacy_metrics
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset


def _tables(pack: ParamPack, seed: int, n: int = 120) -> dict:
    return generate_dataset(pack, n_patients=n, seed=seed).tables


def test_metrics_are_finite_and_in_range(pack: ParamPack) -> None:
    report = privacy_metrics(_tables(pack, 1), _tables(pack, 2))
    assert isinstance(report, PrivacyReport)
    assert report.dcr_median >= 0 and report.dcr_p5 >= 0
    assert 0.0 <= report.nndr_median <= 1.0  # d1 <= d2 so the ratio is bounded by 1
    assert 0.0 <= report.identifiability <= 1.0
    assert report.n_synthetic > 0 and report.n_real > 0


def test_aggregate_only_synthetic_is_clean(pack: ParamPack) -> None:
    # Generators sample from aggregate params and never copy a real row, so even
    # the closest synthetic record keeps a positive distance to every real one.
    report = privacy_metrics(_tables(pack, 1), _tables(pack, 2))
    assert report.dcr_p5 > 0.0  # no memorization in the privacy-relevant tail


def test_memorization_is_detected(pack: ParamPack) -> None:
    # Control: if the "synthetic" set IS the real set, DCR collapses to zero and
    # every real record is identifiable — proving the metrics discriminate.
    real = _tables(pack, 3)
    report = privacy_metrics(real, real)
    assert report.dcr_median == 0.0
    assert report.identifiability > 0.9


def test_is_deterministic(pack: ParamPack) -> None:
    a = privacy_metrics(_tables(pack, 5), _tables(pack, 6))
    b = privacy_metrics(_tables(pack, 5), _tables(pack, 6))
    assert a == b


def test_no_torch_or_synthcity_imported(pack: ParamPack) -> None:
    # R20 containment: the privacy path pulls in no torch / synthcity tree.
    import clifforge.eval.privacy  # noqa: F401

    privacy_metrics(_tables(pack, 1), _tables(pack, 2))
    assert "torch" not in sys.modules
    assert "synthcity" not in sys.modules

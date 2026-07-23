"""Unit tests for the Tier 3 labs generator (U11, R9/R12/KTD-4/R22).

Driven by a small hand-built copula pack plus directly-constructed ``SpineFrame``s
(no real data, no run_fit): generated pairwise correlations recover the pack
copula, values stay within outlier bounds, per-lab presence tracks the fitted
rates with no imputation of absent labs, creatinine rises under the renal-failure
flag, categories are exact mCIDE members, the frame passes the gate, and output
is seed-reproducible.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import labs
from clifforge.generate.tables.labs import LabObservation, labs_frame, sample_labs
from clifforge.reference import bounds, categories

_GRID = 24.0  # one lab panel per ICU interval, for dense sampling

# Three real mCIDE labs so category/bounds checks are meaningful. creatinine and
# bun are positively correlated (0.6); sodium is ~uncorrelated with creatinine.
_ORDER = ["creatinine", "bun", "sodium"]
_CORR = [
    [1.0, 0.6, 0.0],
    [0.6, 1.0, -0.3],
    [0.0, -0.3, 1.0],
]
_MARGINALS = {
    "creatinine": {"log_mean": 0.79, "log_sd": 0.4},
    "bun": {"log_mean": 2.8, "log_sd": 0.5},
    "sodium": {"log_mean": 4.95, "log_sd": 0.03},
}


def _pack(
    presence: dict[str, float] | None = None,
    correlation: list[list[float]] | None = None,
    grid_step_hours: float = _GRID,
) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "labs": {
                "n_records": 1000,
                "fitted": True,
                "params": {
                    "lab_order": _ORDER,
                    "lab_correlation": correlation or _CORR,
                    "lab_marginals": _MARGINALS,
                    "lab_presence": presence or {"creatinine": 1.0, "bun": 1.0, "sodium": 1.0},
                },
            },
            "spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}},
        },
    )


def _spine(levels: list[int], hid: str = "H0", renal: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[renal] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _spearman(x: list[float], y: list[float]) -> float:
    xr = np.argsort(np.argsort(np.asarray(x)))
    yr = np.argsort(np.argsort(np.asarray(y)))
    return float(np.corrcoef(xr, yr)[0, 1])


def test_is_deterministic_under_fixed_seed() -> None:
    pack = _pack()
    sp = _spine([3] * 10)
    a = sample_labs(sp, pack, np.random.default_rng(4))
    b = sample_labs(sp, pack, np.random.default_rng(4))
    assert a == b


def test_frame_reproducible_byte_for_byte() -> None:
    pack = _pack()
    sp = _spine([3] * 8, hid="H1")
    a = labs_frame(sample_labs(sp, pack, np.random.default_rng(5)))
    b = labs_frame(sample_labs(sp, pack, np.random.default_rng(5)))
    assert a.equals(b)


def test_generated_correlation_recovers_copula() -> None:
    pack = _pack()
    rng = np.random.default_rng(0)
    # Pair labs measured in the same panel (same hid + order time).
    panels: dict[tuple[str, object], dict[str, float]] = {}
    for i in range(40):
        for o in sample_labs(_spine([3] * 30, hid=f"H{i}"), pack, rng):
            panels.setdefault((o.hospitalization_id, o.lab_order_dttm), {})[o.lab_category] = (
                o.lab_value_numeric
            )
    creat = [p["creatinine"] for p in panels.values()]
    bun = [p["bun"] for p in panels.values()]
    sod = [p["sodium"] for p in panels.values()]
    assert _spearman(creat, bun) > 0.45  # strong positive, ~0.6 in the pack
    assert abs(_spearman(creat, sod)) < 0.15  # ~uncorrelated in the pack


def test_all_values_within_outlier_bounds() -> None:
    pack = _pack()
    rng = np.random.default_rng(1)
    for i in range(20):
        for o in sample_labs(_spine([3] * 20, hid=f"H{i}"), pack, rng):
            lo, hi = bounds("labs", o.lab_category)
            assert lo <= o.lab_value_numeric <= hi


def test_presence_rates_match_pack() -> None:
    pack = _pack(presence={"creatinine": 0.7, "bun": 0.4, "sodium": 0.9})
    rng = np.random.default_rng(2)
    n_stays = 3000
    stays_with = {"creatinine": 0, "bun": 0, "sodium": 0}
    for i in range(n_stays):
        seen = {o.lab_category for o in sample_labs(_spine([3], hid=f"H{i}"), pack, rng)}
        for lab in stays_with:
            if lab in seen:
                stays_with[lab] += 1
    for lab, rate in {"creatinine": 0.7, "bun": 0.4, "sodium": 0.9}.items():
        assert abs(stays_with[lab] / n_stays - rate) < 0.04


def test_masked_absent_lab_never_emitted() -> None:
    pack = _pack(presence={"creatinine": 1.0, "bun": 1.0, "sodium": 0.0})
    rng = np.random.default_rng(3)
    for i in range(50):
        for o in sample_labs(_spine([3] * 5, hid=f"H{i}"), pack, rng):
            assert o.lab_category != "sodium"  # sample-then-mask: absent -> no row


def test_creatinine_rises_under_renal_failure() -> None:
    pack = _pack()
    healthy = _spine([3] * 60, hid="Hok", renal=False)
    failing = _spine([3] * 60, hid="Hbad", renal=True)
    ok = [
        o.lab_value_numeric
        for o in sample_labs(healthy, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    bad = [
        o.lab_value_numeric
        for o in sample_labs(failing, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    sod_ok = [
        o.lab_value_numeric
        for o in sample_labs(healthy, pack, np.random.default_rng(1))
        if o.lab_category == "sodium"
    ]
    sod_bad = [
        o.lab_value_numeric
        for o in sample_labs(failing, pack, np.random.default_rng(1))
        if o.lab_category == "sodium"
    ]
    assert float(np.mean(bad)) > float(np.mean(ok))  # renal coupling raises creatinine
    assert abs(float(np.mean(sod_bad)) - float(np.mean(sod_ok))) < 1.0  # sodium uncoupled


def test_no_icu_time_yields_no_labs() -> None:
    pack = _pack()
    ward_only = _spine([0, 1, 1, 0], hid="Hward")  # never reaches ICU threshold
    assert sample_labs(ward_only, pack, np.random.default_rng(0)) == []


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    ok = set(categories("labs", "lab_category"))
    seen = {o.lab_category for o in sample_labs(_spine([3] * 10), pack, np.random.default_rng(0))}
    assert seen and seen <= ok


def test_names_echo_categories() -> None:
    pack = _pack()
    for o in sample_labs(_spine([3] * 6), pack, np.random.default_rng(0)):
        assert o.lab_name == o.lab_category


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    obs: list[LabObservation] = []
    for i in range(20):
        obs += sample_labs(_spine([2, 3, 4, 3, 2], hid=f"H{i}"), pack, np.random.default_rng(i))
    frame = labs_frame(obs)
    dtype = frame.schema["lab_order_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    report = gate.validate(frame, "labs", run_secondary=False)
    assert report.pandera_passed


def test_missing_labs_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={"spine": {"params": {"state_model": {}}}})
    try:
        sample_labs(_spine([3, 3]), empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "labs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no labs block")


def test_module_exports() -> None:
    assert set(labs.__all__) == {"LabObservation", "labs_frame", "sample_labs"}

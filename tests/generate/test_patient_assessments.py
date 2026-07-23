"""Unit tests for the Tier 4 patient_assessments generator (U14, R12/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import patient_assessments as pa
from clifforge.generate.tables.patient_assessments import (
    patient_assessments_frame,
    sample_patient_assessments,
)
from clifforge.reference import categories

_GRID = 4.0  # one assessment per ICU interval


def _pack(grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(levels: list[int], hid: str = "H0", neuro: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[neuro] * n,
        outcome="alive",
    )


def _values(rows: list, category: str) -> list[float]:
    return [r.numerical_value for r in rows if r.assessment_category == category]


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine([2, 3, 4, 2])
    a = sample_patient_assessments(sp, pack, np.random.default_rng(0))
    b = sample_patient_assessments(sp, pack, np.random.default_rng(0))
    assert a == b


def test_rass_shifts_negative_under_sedation() -> None:
    pack = _pack()
    sedated = sample_patient_assessments(_spine([3] * 40, hid="Hs"), pack, np.random.default_rng(0))
    awake = sample_patient_assessments(_spine([2] * 40, hid="Ha"), pack, np.random.default_rng(0))
    assert float(np.mean(_values(sedated, "RASS"))) < float(np.mean(_values(awake, "RASS")))


def test_gcs_lower_with_neuro_failure() -> None:
    pack = _pack()
    impaired = sample_patient_assessments(
        _spine([3] * 40, hid="Hi", neuro=True), pack, np.random.default_rng(0)
    )
    intact = sample_patient_assessments(
        _spine([3] * 40, hid="Hk", neuro=False), pack, np.random.default_rng(0)
    )
    assert float(np.mean(_values(impaired, "gcs_total"))) < float(
        np.mean(_values(intact, "gcs_total"))
    )


def test_scores_within_valid_ranges() -> None:
    pack = _pack()
    sp = _spine([2, 3, 4, 5] * 10, hid="Hr", neuro=True)
    for r in sample_patient_assessments(sp, pack, np.random.default_rng(1)):
        if r.assessment_category == "RASS":
            assert -5 <= r.numerical_value <= 4
        else:
            assert 3 <= r.numerical_value <= 15


def test_no_icu_time_yields_no_assessments() -> None:
    pack = _pack()
    ward = _spine([0, 1, 1, 0], hid="Hw")
    assert sample_patient_assessments(ward, pack, np.random.default_rng(0)) == []


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    ok = set(categories("patient_assessments", "assessment_category"))
    seen = {
        r.assessment_category
        for r in sample_patient_assessments(_spine([3] * 6), pack, np.random.default_rng(0))
    }
    assert seen == {"RASS", "gcs_total"}
    assert seen <= ok


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    rows: list = []
    for i in range(15):
        rows += sample_patient_assessments(
            _spine([2, 3, 4, 3, 2], hid=f"H{i}", neuro=bool(i % 2)), pack, np.random.default_rng(i)
        )
    frame = patient_assessments_frame(rows)
    dtype = frame.schema["recorded_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "patient_assessments", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(pa.__all__) == {
        "AssessmentRow",
        "patient_assessments_frame",
        "sample_patient_assessments",
    }

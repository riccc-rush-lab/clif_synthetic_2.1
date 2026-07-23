"""Unit tests for the Tier 4 medication_admin_continuous generator (U13, R11/AE3/R22)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import medication_admin_continuous as mac
from clifforge.generate.tables.medication_admin_continuous import (
    medication_admin_continuous_frame,
    sample_medication_admin_continuous,
)
from clifforge.reference import categories

_GRID = 1.0


def _pack(stop_hazard: float = 0.0, grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "medication_admin_continuous": {
                "params": {
                    "infusion_hazards": {
                        "norepinephrine": {"mean_run_intervals": 2.0, "stop_hazard": stop_hazard},
                        "propofol": {"mean_run_intervals": 2.0, "stop_hazard": stop_hazard},
                    }
                }
            },
            "spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}},
        },
    )


def _spine(levels: list[int], cv: list[bool], hid: str = "H0") -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=cv,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_is_deterministic() -> None:
    pack = _pack(stop_hazard=0.5)
    sp = _spine([3] * 6, [True, True, False, True, False, False])
    a = sample_medication_admin_continuous(sp, pack, np.random.default_rng(0))
    b = sample_medication_admin_continuous(sp, pack, np.random.default_rng(0))
    assert a == b


def test_ae3_stop_is_new_zero_dose_row_no_bolus() -> None:
    pack = _pack(stop_hazard=0.0)
    sp = _spine([0] * 8, [True] * 5 + [False] * 3, hid="Hae3")
    rows = sample_medication_admin_continuous(sp, pack, np.random.default_rng(0))
    norepi = [r for r in rows if r.med_category == "norepinephrine"]
    starts = [r for r in norepi if r.mar_action_category == "start"]
    stops = [r for r in norepi if r.mar_action_category == "stop"]
    assert len(starts) == 1 and len(stops) == 1
    assert starts[0].med_dose > 0
    assert stops[0].med_dose == 0  # stop is a distinct zero-dose row
    assert stops[0].admin_dttm > starts[0].admin_dttm
    assert all(r.mar_action_category in {"start", "stop"} for r in rows)  # no bolus


def test_norepinephrine_starts_only_under_cv_failure() -> None:
    pack = _pack(stop_hazard=0.3)
    cv = [False, True, True, False, True, False, True, True]
    sp = _spine([2] * len(cv), cv, hid="Hcv")
    rows = sample_medication_admin_continuous(sp, pack, np.random.default_rng(2))
    for r in rows:
        if r.med_category == "norepinephrine" and r.mar_action_category == "start":
            idx = int((r.admin_dttm - datetime(2020, 1, 1, tzinfo=UTC)).total_seconds() / 3600)
            assert cv[idx]  # every vasopressor start sits in a cv-failure interval


def test_sedative_runs_during_invasive_ventilation() -> None:
    pack = _pack(stop_hazard=0.0)
    # No cv failure -> no vasopressor; IMV throughout -> propofol runs.
    sp = _spine([3] * 6, [False] * 6, hid="Hsed")
    rows = sample_medication_admin_continuous(sp, pack, np.random.default_rng(0))
    cats = {r.med_category for r in rows}
    assert "propofol" in cats
    assert "norepinephrine" not in cats


def test_doses_non_negative_and_within_range() -> None:
    pack = _pack(stop_hazard=0.3)
    sp = _spine([3] * 20, [True, False] * 10, hid="Hd")
    for r in sample_medication_admin_continuous(sp, pack, np.random.default_rng(1)):
        assert r.med_dose >= 0
        if r.med_category == "norepinephrine" and r.med_dose > 0:
            assert 0.02 <= r.med_dose <= 0.5
        if r.med_category == "propofol" and r.med_dose > 0:
            assert 5.0 <= r.med_dose <= 50.0


def test_no_coupling_yields_no_infusions() -> None:
    pack = _pack()
    sp = _spine([0, 1, 1, 0], [False] * 4, hid="Hnone")  # never cv-failure, never IMV
    assert sample_medication_admin_continuous(sp, pack, np.random.default_rng(0)) == []


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack(stop_hazard=0.3)
    med_ok = set(categories("medication_admin_continuous", "med_category"))
    route_ok = set(categories("medication_admin_continuous", "med_route_category"))
    action_ok = set(categories("medication_admin_continuous", "mar_action_category"))
    sp = _spine([3] * 10, [True] * 5 + [False] * 5, hid="Hc")
    for r in sample_medication_admin_continuous(sp, pack, np.random.default_rng(0)):
        assert r.med_category in med_ok
        assert r.med_route_category in route_ok
        assert r.mar_action_category in action_ok


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack(stop_hazard=0.4)
    rows: list = []
    for i in range(15):
        cv = [bool((i + t) % 2) for t in range(8)]
        rows += sample_medication_admin_continuous(
            _spine([3] * 8, cv, hid=f"H{i}"), pack, np.random.default_rng(i)
        )
    frame = medication_admin_continuous_frame(rows)
    dtype = frame.schema["admin_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "medication_admin_continuous", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(mac.__all__) == {
        "MedAdminRow",
        "medication_admin_continuous_frame",
        "sample_medication_admin_continuous",
    }

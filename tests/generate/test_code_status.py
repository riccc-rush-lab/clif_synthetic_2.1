"""Unit tests for the Tier 5 code_status generator (U19, R12/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import code_status as cs
from clifforge.generate.tables.code_status import code_status_frame, sample_code_status
from clifforge.reference import categories

_DEESCALATION = {"DNR/DNI", "AND"}


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(outcome: str, n: int = 48, hid: str = "H0") -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[3] * n,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome=outcome,
    )


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine("expired")
    a = sample_code_status(sp, pack, np.random.default_rng(0), patient_id="P1")
    b = sample_code_status(sp, pack, np.random.default_rng(0), patient_id="P1")
    assert a == b


def test_everyone_starts_full() -> None:
    pack = _pack()
    for outcome in ("alive", "expired"):
        events = sample_code_status(_spine(outcome), pack, np.random.default_rng(3), patient_id="P")
        assert events[0].code_status_category == "Full"


def test_deescalation_concentrated_before_death() -> None:
    pack = _pack()
    rng = np.random.default_rng(1)
    expired_deesc = sum(
        any(
            e.code_status_category in _DEESCALATION
            for e in sample_code_status(_spine("expired"), pack, rng, patient_id=f"E{i}")
        )
        for i in range(400)
    )
    rng = np.random.default_rng(2)
    survivor_deesc = sum(
        any(
            e.code_status_category in _DEESCALATION
            for e in sample_code_status(_spine("alive"), pack, rng, patient_id=f"S{i}")
        )
        for i in range(400)
    )
    assert expired_deesc / 400 > 0.5  # most expired trajectories de-escalate
    assert survivor_deesc / 400 < 0.15  # survivors rarely do
    assert expired_deesc > survivor_deesc


def test_start_times_are_ordered() -> None:
    pack = _pack()
    for i in range(50):
        events = sample_code_status(
            _spine("expired"), pack, np.random.default_rng(i), patient_id=f"P{i}"
        )
        times = [e.start_dttm for e in events]
        assert times == sorted(times)
        assert len(set(times)) == len(times)  # strictly increasing


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    ok = set(categories("code_status", "code_status_category"))
    seen: set[str] = set()
    for i in range(200):
        for e in sample_code_status(
            _spine("expired"), pack, np.random.default_rng(i), patient_id=f"P{i}"
        ):
            seen.add(e.code_status_category)
    assert seen and seen <= ok


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    events: list = []
    for i in range(30):
        events += sample_code_status(
            _spine("expired" if i % 2 else "alive"),
            pack,
            np.random.default_rng(i),
            patient_id=f"P{i}",
        )
    frame = code_status_frame(events)
    dtype = frame.schema["start_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "code_status", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(cs.__all__) == {"CodeStatusEvent", "code_status_frame", "sample_code_status"}

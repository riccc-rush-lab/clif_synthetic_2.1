"""Unit tests for the Tier 4 position generator (U15, R12/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import position as pos
from clifforge.generate.tables.position import position_frame, sample_position
from clifforge.reference import categories

_GRID = 6.0  # one position check per ICU interval


def _pack(grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(levels: list[int], hid: str = "H0", resp: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[resp] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _prone_rate(rows: list) -> float:
    return sum(r.position_category == "prone" for r in rows) / len(rows)


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine([2, 3, 4, 3], resp=True)
    a = sample_position(sp, pack, np.random.default_rng(0))
    b = sample_position(sp, pack, np.random.default_rng(0))
    assert a == b


def test_prone_elevated_during_severe_hypoxemia() -> None:
    pack = _pack()
    severe = sample_position(_spine([3] * 200, hid="Hs", resp=True), pack, np.random.default_rng(0))
    mild = sample_position(_spine([3] * 200, hid="Hm", resp=False), pack, np.random.default_rng(0))
    assert _prone_rate(severe) > 0.4  # concentrated proning
    assert _prone_rate(mild) < 0.15  # rarely prone without severe hypoxemia
    assert _prone_rate(severe) > _prone_rate(mild)


def test_prone_requires_invasive_ventilation() -> None:
    pack = _pack()
    # resp failure but only high-flow (level 2, not intubated) -> proning stays rare.
    rows = sample_position(_spine([2] * 200, hid="Hh", resp=True), pack, np.random.default_rng(0))
    assert _prone_rate(rows) < 0.15


def test_one_position_per_row() -> None:
    pack = _pack()
    ok = set(categories("position", "position_category"))
    for r in sample_position(_spine([2, 3, 4], resp=True), pack, np.random.default_rng(0)):
        assert r.position_category in ok  # a single mutually-exclusive category


def test_no_icu_time_yields_no_positions() -> None:
    pack = _pack()
    assert sample_position(_spine([0, 1, 1, 0], hid="Hw"), pack, np.random.default_rng(0)) == []


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    rows: list = []
    for i in range(15):
        rows += sample_position(
            _spine([2, 3, 4, 3, 2], hid=f"H{i}", resp=bool(i % 2)), pack, np.random.default_rng(i)
        )
    frame = position_frame(rows)
    dtype = frame.schema["recorded_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "position", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(pos.__all__) == {"PositionRow", "position_frame", "sample_position"}

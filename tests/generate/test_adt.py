"""Unit tests for the Tier 2 adt generator (U9, R8).

Driven by directly-constructed ``SpineFrame``s (no real data, no run_fit) so the
generator's contract is checked in isolation: contiguous non-overlapping
movements within the hospitalization window, ICU segments aligned with
high-acuity spine intervals, exact mCIDE categories, gate pass, and determinism.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import adt
from clifforge.generate.tables.adt import (
    ICU_MIN_SUPPORT_LEVEL,
    adt_frame,
    icu_windows,
    sample_adt,
)
from clifforge.reference import categories

_GRID = 2.0  # hours per interval, for easy arithmetic


def _pack(grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(levels: list[int], hid: str = "H0") -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine([1, 1, 3, 3, 0])
    a = sample_adt(sp, pack, np.random.default_rng(0))
    b = sample_adt(sp, pack, np.random.default_rng(999))  # rng is unused -> same result
    assert a == b


def test_movements_are_contiguous_and_span_the_window() -> None:
    pack = _pack()
    admit = datetime(2021, 1, 1, tzinfo=UTC)
    sp = _spine([0, 1, 3, 3, 4, 1])  # ward(2), icu(3), ward(1) -> 6 intervals * 2h = 12h
    moves = sample_adt(sp, pack, np.random.default_rng(0), admit_dttm=admit)
    assert moves[0].in_dttm == admit
    for prev, nxt in zip(moves, moves[1:], strict=False):
        assert prev.out_dttm == nxt.in_dttm  # contiguous, no gaps or overlaps
    total_hours = (moves[-1].out_dttm - moves[0].in_dttm).total_seconds() / 3600
    assert total_hours == 6 * _GRID  # last out_dttm == discharge (matches U8 LOS)


def test_rle_collapses_runs() -> None:
    pack = _pack()
    sp = _spine([0, 1, 3, 3, 4, 1])
    cats = [m.location_category for m in sample_adt(sp, pack, np.random.default_rng(0))]
    assert cats == ["ward", "icu", "ward"]


def test_icu_windows_align_with_high_acuity_segments() -> None:
    pack = _pack()
    admit = datetime(2021, 1, 1, tzinfo=UTC)
    # intervals 2..4 (0-indexed) are >= ICU threshold -> exactly one ICU window.
    sp = _spine([1, 1, 3, 4, 3, 0], hid="H7")
    moves = sample_adt(sp, pack, np.random.default_rng(0), admit_dttm=admit)
    windows = icu_windows(moves)
    assert list(windows) == ["H7"]
    ((icu_in, icu_out),) = windows["H7"]
    # ICU begins after the two ward intervals (2 * 2h) and lasts three intervals.
    assert (icu_in - admit).total_seconds() / 3600 == 2 * _GRID
    assert (icu_out - icu_in).total_seconds() / 3600 == 3 * _GRID


def test_all_ward_has_no_icu_window() -> None:
    pack = _pack()
    sp = _spine([0, 1, 1, 0], hid="H1")
    moves = sample_adt(sp, pack, np.random.default_rng(0))
    assert [m.location_category for m in moves] == ["ward"]
    assert icu_windows(moves) == {}


def test_all_icu_is_single_window() -> None:
    pack = _pack()
    sp = _spine([3, 4, 5, 3], hid="H2")
    moves = sample_adt(sp, pack, np.random.default_rng(0))
    assert [m.location_category for m in moves] == ["icu"]
    assert all(m.location_type == "medical_icu" for m in moves)
    assert len(icu_windows(moves)["H2"]) == 1


def test_categories_and_types_are_mcide_members() -> None:
    pack = _pack()
    loc_ok = set(categories("adt", "location_category"))
    type_ok = set(categories("adt", "location_type"))
    hosp_ok = set(categories("adt", "hospital_type_category"))
    sp = _spine([0, 2, 3, 1, 4])
    for m in sample_adt(sp, pack, np.random.default_rng(0)):
        assert m.location_category in loc_ok
        assert m.hospital_type in hosp_ok
        if m.location_type is not None:
            assert m.location_type in type_ok


def test_ward_rows_have_null_location_type() -> None:
    pack = _pack()
    sp = _spine([0, 1, 3])
    moves = sample_adt(sp, pack, np.random.default_rng(0))
    ward = next(m for m in moves if m.location_category == "ward")
    icu = next(m for m in moves if m.location_category == "icu")
    assert ward.location_type is None
    assert icu.location_type == "medical_icu"


def test_hospitalization_id_defaults_to_spine_id() -> None:
    pack = _pack()
    moves = sample_adt(_spine([3, 3], hid="Hxyz"), pack, np.random.default_rng(0))
    assert all(m.hospitalization_id == "Hxyz" for m in moves)


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    moves: list = []
    for i in range(30):
        levels = [0, 1, 3, 3, 4, 1, 0][: 2 + (i % 5)]
        moves += sample_adt(_spine(levels, hid=f"H{i}"), pack, np.random.default_rng(0))
    frame = adt_frame(moves)
    for col in ("in_dttm", "out_dttm"):
        dtype = frame.schema[col]
        assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    report = gate.validate(frame, "adt", run_secondary=False)
    assert report.pandera_passed


def test_threshold_constant_matches_semantics() -> None:
    # A level exactly at the threshold is ICU; one below is ward.
    pack = _pack()
    at = _spine([ICU_MIN_SUPPORT_LEVEL])
    below = _spine([ICU_MIN_SUPPORT_LEVEL - 1])
    assert sample_adt(at, pack, np.random.default_rng(0))[0].location_category == "icu"
    assert sample_adt(below, pack, np.random.default_rng(0))[0].location_category == "ward"


def test_module_exports() -> None:
    assert set(adt.__all__) == {
        "ICU_MIN_SUPPORT_LEVEL",
        "AdtMovement",
        "adt_frame",
        "icu_windows",
        "sample_adt",
    }

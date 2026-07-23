"""Spine-state derivation tests on fabricated frames (U5d).

All fixtures here are synthetic and PHI-free — no real record or real-data path
is touched (KTD-1). They exercise the organ-support ladder and the four
organ-failure flags against hand-constructed timelines with known answers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from clifforge.fit import spine_state
from clifforge.fit.spine_state import SpineStateConfig, Support

_T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _at(hours: int) -> datetime:
    return _T0 + timedelta(hours=hours)


def _tables() -> dict[str, pl.LazyFrame]:
    """One hospitalization H1 walking up and down the support ladder."""
    hospitalization = pl.LazyFrame(
        {
            "hospitalization_id": ["H1"],
            "patient_id": ["P1"],
            "admission_dttm": [_T0],
            "discharge_category": ["Expired"],
        }
    )
    respiratory_support = pl.LazyFrame(
        {
            "hospitalization_id": ["H1", "H1"],
            "recorded_dttm": [_at(0), _at(2)],
            "device_category": ["IMV", "Room Air"],
        }
    )
    medication_admin_continuous = pl.LazyFrame(
        {
            "hospitalization_id": ["H1"],
            "admin_dttm": [_at(1)],
            "med_category": ["norepinephrine"],
        }
    )
    crrt_therapy = pl.LazyFrame(
        {
            "hospitalization_id": ["H1"],
            "recorded_dttm": [_at(3)],
        }
    )
    patient_assessments = pl.LazyFrame(
        {
            "hospitalization_id": ["H1"],
            "recorded_dttm": [_at(0)],
            "assessment_category": ["RASS"],
            "numerical_value": [-4.0],
        }
    )
    return {
        "hospitalization": hospitalization,
        "respiratory_support": respiratory_support,
        "medication_admin_continuous": medication_admin_continuous,
        "crrt_therapy": crrt_therapy,
        "patient_assessments": patient_assessments,
    }


def test_support_ladder_levels_by_interval() -> None:
    timeline = spine_state.derive_state_timeline(_tables()).collect().sort("interval_idx")
    levels = dict(zip(timeline["interval_idx"], timeline["support_level"], strict=True))
    assert levels[0] == int(Support.IMV)  # device IMV
    assert levels[1] == int(Support.CIRC)  # norepinephrine infusion
    assert levels[2] == int(Support.NONE)  # room air
    assert levels[3] == int(Support.RENAL_MCS)  # CRRT


def test_organ_failure_flags() -> None:
    timeline = spine_state.derive_state_timeline(_tables()).collect().sort("interval_idx")
    rows = {r["interval_idx"]: r for r in timeline.iter_rows(named=True)}
    assert rows[0]["resp_flag"] is True  # IMV
    assert rows[0]["neuro_flag"] is True  # RASS -4
    assert rows[1]["cv_flag"] is True  # vasopressor
    assert rows[3]["renal_flag"] is True  # CRRT
    assert rows[2]["cv_flag"] is False  # room-air interval, nothing active


def test_outcome_expired() -> None:
    outcome = spine_state.outcome_by_hospitalization(_tables()["hospitalization"]).collect()
    assert outcome["outcome"].to_list() == ["expired"]


def test_horizon_cap_drops_far_intervals() -> None:
    tables = _tables()
    # An event 100 days out must be dropped by the 28-day horizon cap.
    late = pl.LazyFrame(
        {
            "hospitalization_id": ["H1"],
            "recorded_dttm": [_at(24 * 100)],
            "device_category": ["IMV"],
        }
    )
    tables["respiratory_support"] = pl.concat([tables["respiratory_support"], late], how="vertical")
    config = SpineStateConfig()
    timeline = spine_state.derive_state_timeline(tables, config).collect()
    assert timeline["interval_idx"].max() < config.horizon_intervals

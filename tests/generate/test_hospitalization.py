"""Unit tests for the Tier 1 hospitalization generator (U8, R8/R12/AE4).

Driven by a small hand-built pack plus directly-constructed ``SpineFrame``s (no
real data, no run_fit) so the generator's contract is checked in isolation: AE4
death/discharge consistency both directions, LOS from the spine horizon, tz-aware
UTC datetimes, zero orphans against a patient cohort, exact mCIDE categories, and
seed reproducibility.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import hospitalization
from clifforge.generate.tables.hospitalization import (
    DEATH_DISCHARGE_CATEGORY,
    HospitalizationRecord,
    hospitalization_frame,
    sample_hospitalization,
)
from clifforge.generate.tables.patient import patient_frame, sample_patient
from clifforge.reference import categories

_ADMIT_TYPE = {"ed": 0.8, "direct": 0.15, "elective": 0.05}
_DISCHARGE = {"Home": 0.6, "Skilled Nursing Facility (SNF)": 0.2, "Expired": 0.15, "Hospice": 0.05}


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "hospitalization": {
                "n_records": 1000,
                "fitted": True,
                "params": {
                    "admission_type_category_marginal": _ADMIT_TYPE,
                    "discharge_category_marginal": _DISCHARGE,
                },
            },
            "spine": {
                "n_records": 1000,
                "fitted": True,
                "params": {"state_model": {"grid_step_hours": grid_step_hours}},
            },
        },
    )


def _spine(hid: str, n: int, outcome: str, level: int = 1) -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[level] * n,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome=outcome,
    )


def test_sample_is_deterministic_under_fixed_seed() -> None:
    pack = _pack()
    sp = _spine("H0", 12, "alive")
    a = sample_hospitalization(sp, pack, np.random.default_rng(5))
    b = sample_hospitalization(sp, pack, np.random.default_rng(5))
    assert a == b


def test_ae4_expired_sets_death_and_expired_disposition() -> None:
    pack = _pack()
    admit = datetime(2021, 3, 1, tzinfo=UTC)
    rec = sample_hospitalization(
        _spine("H1", 20, "expired"), pack, np.random.default_rng(0), admit_dttm=admit
    )
    assert rec.discharge_category == DEATH_DISCHARGE_CATEGORY
    assert rec.death_dttm is not None
    assert rec.death_dttm == rec.discharge_dttm  # death consistent with discharge


def test_ae4_survivor_has_no_death_and_non_death_disposition() -> None:
    pack = _pack()
    for seed in range(50):
        rec = sample_hospitalization(_spine("H", 8, "alive"), pack, np.random.default_rng(seed))
        assert rec.death_dttm is None
        assert rec.discharge_category != DEATH_DISCHARGE_CATEGORY
        assert rec.discharge_category in set(categories("hospitalization", "discharge_category"))


def test_los_follows_spine_horizon_on_the_grid() -> None:
    admit = datetime(2020, 6, 1, tzinfo=UTC)
    # grid_step 2h * 15 intervals = 30h LOS.
    rec = sample_hospitalization(
        _spine("H", 15, "alive"),
        _pack(grid_step_hours=2.0),
        np.random.default_rng(1),
        admit_dttm=admit,
    )
    assert (rec.discharge_dttm - rec.admission_dttm).total_seconds() == 30 * 3600


def test_datetimes_are_tz_aware_utc() -> None:
    pack = _pack()
    frame = hospitalization_frame(
        [sample_hospitalization(_spine("H0", 5, "alive"), pack, np.random.default_rng(0))]
    )
    for col in ("admission_dttm", "discharge_dttm"):
        dtype = frame.schema[col]
        assert isinstance(dtype, pl.Datetime)
        assert dtype.time_zone == "UTC"


def test_admission_type_marginal_matches_pack() -> None:
    pack = _pack()
    rng = np.random.default_rng(17)
    cats = [
        sample_hospitalization(_spine("H", 4, "alive"), pack, rng).admission_type_category
        for _ in range(6000)
    ]
    for cat, prob in _ADMIT_TYPE.items():
        assert abs(cats.count(cat) / len(cats) - prob) < 0.03


def test_zero_orphans_against_patient_cohort() -> None:
    # One-to-many linking: 3 patients own 5 hospitalizations. Every
    # hospitalization's patient_id must resolve to a patient row (R8, zero
    # orphans). The patient pack needs demographic marginals too.
    pack = _pack()
    pack.tables["patient"] = {
        "params": {
            "race_category_marginal": {"White": 0.6, "Unknown": 0.4},
            "ethnicity_category_marginal": {"Non-Hispanic": 0.7, "Unknown": 0.3},
            "sex_category_marginal": {"Female": 0.5, "Male": 0.5},
        }
    }
    rng = np.random.default_rng(2)

    # patient P0 owns H0/H1, P1 owns H2/H3, P2 owns H4 — a genuine one-to-many map.
    ownership = {"H0": "P0", "H1": "P0", "H2": "P1", "H3": "P1", "H4": "P2"}
    patients = patient_frame(
        [sample_patient(pack, rng, patient_id=pid) for pid in ("P0", "P1", "P2")]
    )
    hospitalizations = hospitalization_frame(
        [
            sample_hospitalization(
                _spine(hid, 6, "alive"), pack, rng, hospitalization_id=hid, patient_id=pid
            )
            for hid, pid in ownership.items()
        ]
    )

    patient_ids = set(patients["patient_id"].to_list())
    hosp_patient_ids = set(hospitalizations["patient_id"].to_list())
    assert hosp_patient_ids <= patient_ids  # zero orphans
    assert hospitalizations["hospitalization_id"].n_unique() == 5
    assert patients["patient_id"].n_unique() == 3  # one-to-many confirmed


def test_frame_passes_gate_and_is_reproducible() -> None:
    pack = _pack()

    def cohort(rng: np.random.Generator) -> list[HospitalizationRecord]:
        outcomes = ["alive", "expired", "alive", "alive"]
        return [
            sample_hospitalization(
                _spine(f"H{i}", 6 + i, outcomes[i % 4], level=1),
                pack,
                rng,
                hospitalization_id=f"H{i}",
                patient_id=f"P{i}",
            )
            for i in range(40)
        ]

    a = hospitalization_frame(cohort(np.random.default_rng(3)))
    b = hospitalization_frame(cohort(np.random.default_rng(3)))
    assert a.equals(b)
    report = gate.validate(a, "hospitalization", run_secondary=False)
    assert report.pandera_passed


def test_survivor_marginal_all_death_raises() -> None:
    pack = ParamPack(
        manifest={},
        tables={
            "hospitalization": {
                "params": {
                    "admission_type_category_marginal": _ADMIT_TYPE,
                    "discharge_category_marginal": {"Expired": 1.0},
                }
            },
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
        },
    )
    try:
        sample_hospitalization(_spine("H", 4, "alive"), pack, np.random.default_rng(0))
    except ValueError as exc:
        assert "non-death" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when no survivor disposition exists")


def test_missing_hospitalization_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={})
    try:
        sample_hospitalization(_spine("H", 4, "alive"), empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "hospitalization" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no hospitalization block")


def test_module_exports() -> None:
    assert set(hospitalization.__all__) == {
        "DEATH_DISCHARGE_CATEGORY",
        "HospitalizationRecord",
        "hospitalization_frame",
        "sample_hospitalization",
    }


def test_age_at_admission_generated_only_when_pack_has_grid() -> None:
    from clifforge.generate.tables.hospitalization import _sample_age

    # No age grid in the pack -> no age column at all (byte-identical to before).
    pack = _pack()
    rec = sample_hospitalization(_spine("H0", 6, "alive"), pack, np.random.default_rng(0))
    assert rec.age_at_admission is None
    frame = hospitalization_frame([rec])
    assert "age_at_admission" not in frame.columns

    # With a grid, age is sampled within its bounds and appears in the frame.
    grid = [18.0, 40.0, 55.0, 61.0, 70.0, 85.0, 105.0]
    pack.tables["hospitalization"]["params"]["age_at_admission_quantiles"] = grid
    recs = [
        sample_hospitalization(_spine(f"H{i}", 6, "alive"), pack, np.random.default_rng(i))
        for i in range(200)
    ]
    ages = [r.age_at_admission for r in recs]
    assert all(a is not None and 18 <= a <= 105 for a in ages)
    frame = hospitalization_frame(recs)
    assert "age_at_admission" in frame.columns
    assert frame.schema["age_at_admission"] == pl.Int64
    # inverse-CDF recovers the grid's median within a couple of years
    assert abs(float(pl.Series(ages).median()) - 61.0) < 4

    # _sample_age is deterministic under a fixed rng
    assert _sample_age(grid, np.random.default_rng(3)) == _sample_age(
        grid, np.random.default_rng(3)
    )

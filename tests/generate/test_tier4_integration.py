"""Tier 4 integration: cross-table coherence flows through the shared spine (KTD-6).

Every Tier 4 generator reads only ``(spine, pack, rng)``, so their agreement is
mediated entirely by the latent spine. These tests assert that the couplings line
up across tables: invasive ventilation in respiratory_support coincides with
sedation in medication_admin_continuous and patient_assessments; severe hypoxemia
drives both proning and IMV; and cardiovascular failure ties vasopressor use to
low blood pressure without any table reading another.
"""

from __future__ import annotations

import numpy as np

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables.medication_admin_continuous import (
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.patient_assessments import sample_patient_assessments
from clifforge.generate.tables.position import sample_position
from clifforge.generate.tables.respiratory_support import sample_respiratory_support
from clifforge.generate.tables.vitals import sample_vitals

_SBP = {
    str(s): {"mean": m, "phi": 0.5, "sigma": 3.0}
    for s, m in {0: 125, 1: 123, 2: 118, 3: 112, 4: 90, 5: 88}.items()
}


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "vitals": {"params": {"sbp_ar1_by_state": _SBP}},
            "medication_admin_continuous": {
                "params": {
                    "infusion_hazards": {
                        "norepinephrine": {"stop_hazard": 0.0, "mean_run_intervals": 2.0},
                        "propofol": {"stop_hazard": 0.0, "mean_run_intervals": 2.0},
                    }
                }
            },
            "spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}},
        },
    )


def _spine(levels: list[int], *, cv: bool, resp: bool, hid: str) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[resp] * n,
        cv_flag=[cv] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_invasive_ventilation_coincides_with_sedation() -> None:
    pack = _pack()
    sp = _spine([3] * 12, cv=False, resp=True, hid="Hvent")
    rng = np.random.default_rng(0)
    resp_rows = sample_respiratory_support(sp, pack, rng)
    meds = sample_medication_admin_continuous(sp, pack, rng)
    assessments = sample_patient_assessments(sp, pack, rng)
    # IMV device present AND propofol sedation present AND RASS sedated (negative).
    assert any(r.device_category == "IMV" for r in resp_rows)
    assert any(m.med_category == "propofol" for m in meds)
    rass = [a.numerical_value for a in assessments if a.assessment_category == "RASS"]
    assert rass and float(np.mean(rass)) < 0


def test_severe_hypoxemia_drives_proning_and_imv_together() -> None:
    pack = _pack()
    sp = _spine([3] * 200, cv=False, resp=True, hid="Hards")
    rng = np.random.default_rng(1)
    resp_rows = sample_respiratory_support(sp, pack, rng)
    positions = sample_position(sp, pack, rng)
    assert all(r.device_category == "IMV" for r in resp_rows)  # intubated throughout
    prone = sum(p.position_category == "prone" for p in positions)
    assert prone / len(positions) > 0.4  # proning concentrated in the ARDS window


def test_vasopressor_ties_to_hypotension_via_spine() -> None:
    pack = _pack()
    rng = np.random.default_rng(2)
    shock_sbp: list[float] = []
    stable_sbp: list[float] = []
    shock_has_norepi = stable_has_norepi = 0
    for i in range(30):
        shock = _spine([4] * 24, cv=True, resp=False, hid=f"S{i}")
        stable = _spine([1] * 24, cv=False, resp=False, hid=f"T{i}")
        shock_sbp += [
            v.vital_value for v in sample_vitals(shock, pack, rng) if v.vital_category == "sbp"
        ]
        stable_sbp += [
            v.vital_value for v in sample_vitals(stable, pack, rng) if v.vital_category == "sbp"
        ]
        if any(
            m.med_category == "norepinephrine"
            for m in sample_medication_admin_continuous(shock, pack, rng)
        ):
            shock_has_norepi += 1
        if any(
            m.med_category == "norepinephrine"
            for m in sample_medication_admin_continuous(stable, pack, rng)
        ):
            stable_has_norepi += 1
    # Cardiovascular-failure stays get vasopressors AND lower blood pressure;
    # stable stays get neither — coherence purely through the shared spine flag.
    assert shock_has_norepi == 30 and stable_has_norepi == 0
    assert float(np.mean(shock_sbp)) < float(np.mean(stable_sbp)) - 15


def test_all_tier4_tables_share_hospitalization_id() -> None:
    pack = _pack()
    sp = _spine([2, 3, 4, 3, 2], cv=True, resp=True, hid="Hlink")
    rng = np.random.default_rng(3)
    tables = [
        sample_respiratory_support(sp, pack, rng),
        sample_medication_admin_continuous(sp, pack, rng),
        sample_patient_assessments(sp, pack, rng),
        sample_position(sp, pack, rng),
    ]
    for rows in tables:
        assert rows and all(r.hospitalization_id == "Hlink" for r in rows)

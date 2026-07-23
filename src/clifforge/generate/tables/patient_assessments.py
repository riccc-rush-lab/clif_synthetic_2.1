"""Tier 4 ``patient_assessments`` generator (U14; R12, KTD-6/KTD-7).

No fitted block exists, so bedside assessments are derived from the latent spine.
Two coupled scores are emitted at an assessment cadence within ICU windows:

* **RASS** (sedation-agitation, valid −5..+4) tracks sedation depth. Continuous
  sedation accompanies invasive ventilation, so at ``support_level >= 3`` the
  score is drawn from the sedated (negative) range; otherwise it sits near zero.
  Sedation presence is read from the spine, never from the U13 med table.
* **gcs_total** (Glasgow Coma Scale, valid 3..15) tracks the spine's neuro-failure
  flag: neuro failure draws a low (impaired) score, otherwise a near-normal one.

Scores stay inside their clinical ranges; the categories are exact mCIDE members.
Un-fitted numeric distributions use documented score ranges (R15), not invented
detail. Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame

__all__ = [
    "AssessmentRow",
    "patient_assessments_frame",
    "sample_patient_assessments",
]

_ICU_MIN_SUPPORT_LEVEL = 2
_SEDATION_MIN_SUPPORT_LEVEL = 3  # invasive ventilation implies continuous sedation
_ASSESSMENT_INTERVAL_HOURS = 4.0  # bedside scores charted a few times a shift

_RASS_CATEGORY = "RASS"
_GCS_CATEGORY = "gcs_total"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class AssessmentRow:
    """One charted assessment score."""

    hospitalization_id: str
    recorded_dttm: datetime
    assessment_category: str
    numerical_value: float


def _grid_step_hours(pack: ParamPack) -> float:
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        return 1.0
    return float(block["params"].get("state_model", {}).get("grid_step_hours", 1.0))


def _assessment_intervals(support_level: list[int], grid_step: float) -> list[int]:
    """ICU interval indices at which scores are charted (~every few hours)."""
    intervals: list[int] = []
    last: int | None = None
    for idx, level in enumerate(support_level):
        if level < _ICU_MIN_SUPPORT_LEVEL:
            continue
        if last is None or (idx - last) * grid_step >= _ASSESSMENT_INTERVAL_HOURS:
            intervals.append(idx)
            last = idx
    return intervals


def sample_patient_assessments(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[AssessmentRow]:
    """Emit one hospitalization's RASS/GCS assessments (R12, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = _grid_step_hours(pack)

    rows: list[AssessmentRow] = []
    for idx in _assessment_intervals(spine.support_level, grid_step):
        recorded = admit_dttm + timedelta(hours=idx * grid_step)

        sedated = spine.support_level[idx] >= _SEDATION_MIN_SUPPORT_LEVEL
        rass = int(rng.integers(-4, 0)) if sedated else int(rng.integers(-1, 2))
        rows.append(AssessmentRow(hid, recorded, _RASS_CATEGORY, float(rass)))

        neuro = spine.neuro_flag[idx]
        gcs = int(rng.integers(3, 9)) if neuro else int(rng.integers(13, 16))
        rows.append(AssessmentRow(hid, recorded, _GCS_CATEGORY, float(gcs)))

    return rows


def patient_assessments_frame(rows: list[AssessmentRow]) -> pl.DataFrame:
    """Stack assessments into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "assessment_name": [r.assessment_category for r in rows],
            "assessment_category": [r.assessment_category for r in rows],
            "numerical_value": [r.numerical_value for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": _UTC_DT,
            "assessment_name": pl.String,
            "assessment_category": pl.String,
            "numerical_value": pl.Float64,
        },
    )

"""Tier 1 ``hospitalization`` table generator (U8; R8, R12, AE4).

Each hospitalization is one encounter driven by a latent spine: its length of
stay is the spine's horizon (``n_intervals`` on the fitted grid), and its
death/discharge disposition is the spine's terminal outcome. This is the first
generator to read the spine (KTD-6): it consumes only ``spine.outcome`` and
``spine.n_intervals``, never another table's output.

**AE4 (death/discharge consistency).** CLIF 2.1.0 records the moment of death on
``patient.death_dttm``, not on ``hospitalization`` — the encounter's death signal
is ``discharge_category == "Expired"``. So:

* spine outcome ``expired`` -> ``discharge_category = "Expired"`` and the record
  exposes ``death_dttm = discharge_dttm`` (the value the U21 orchestrator writes
  back to the owning ``patient`` row, closing the cross-table couple, R12/AE4).
* spine outcome ``alive`` -> ``discharge_category`` drawn from the pack marginal
  **with the death category removed and renormalized**, and ``death_dttm = None``.

``patient_id`` / ``hospitalization_id`` are caller-assigned (the orchestrator owns
the id scheme and one-to-many linking, R8); the sampled content is reproducible
byte-for-byte under a fixed ``rng`` (R22). Optional CLIF columns U5 does not fit
(``age_at_admission``, geographic zip/census codes) are omitted, not fabricated
(R15; schema is permissive).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame

__all__ = [
    "DEATH_DISCHARGE_CATEGORY",
    "HospitalizationRecord",
    "hospitalization_frame",
    "sample_hospitalization",
]

#: The one mCIDE discharge_category that means the patient died in the encounter.
#: "Hospice" is a live disposition, not death, so only this value is excluded
#: from the survivor discharge distribution.
DEATH_DISCHARGE_CATEGORY = "Expired"

#: Default admission instant when the caller does not supply one. The orchestrator
#: spreads real admit times across a calendar; a fixed tz-aware epoch keeps a
#: standalone sample reproducible and tz-aware UTC (R7, R22).
_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)

_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class HospitalizationRecord:
    """One encounter linked to a patient, with AE4-consistent disposition.

    ``death_dttm`` is ``discharge_dttm`` for a death and ``None`` for a survivor;
    it is the value the orchestrator writes to the owning ``patient.death_dttm``
    (it is not a ``hospitalization`` column in CLIF 2.1.0).
    """

    patient_id: str
    hospitalization_id: str
    hospitalization_joined_id: str
    admission_dttm: datetime
    discharge_dttm: datetime
    admission_type_category: str
    admission_type_name: str
    discharge_category: str
    discharge_name: str
    death_dttm: datetime | None


def _hospitalization_params(pack: ParamPack) -> dict[str, dict[str, float]]:
    block = pack.tables.get("hospitalization")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'hospitalization' block to sample from")
    params: dict[str, dict[str, float]] = block["params"]
    return params


def _grid_step_hours(pack: ParamPack) -> float:
    """Hours per spine interval — needed to turn ``n_intervals`` into a real LOS."""
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        return 1.0
    state_model = block["params"].get("state_model", {})
    return float(state_model.get("grid_step_hours", 1.0))


def sample_hospitalization(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str = "H0",
    patient_id: str = "P0",
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> HospitalizationRecord:
    """Sample one hospitalization from its spine and the pack (R8, R12, AE4, R22).

    Length of stay is ``spine.n_intervals`` on the fitted grid; disposition is
    driven by ``spine.outcome`` for AE4 consistency. Draws ``admission_type`` from
    ``rng`` first, then (survivors only) ``discharge_category``.
    """
    params = _hospitalization_params(pack)
    los_hours = spine.n_intervals * _grid_step_hours(pack)
    discharge_dttm = admit_dttm + timedelta(hours=los_hours)

    admission_type = categorical(params["admission_type_category_marginal"], rng)

    if spine.outcome == "expired":
        discharge_category = DEATH_DISCHARGE_CATEGORY
        death_dttm: datetime | None = discharge_dttm
    else:
        survivor_marginal = {
            cat: prob
            for cat, prob in params["discharge_category_marginal"].items()
            if cat != DEATH_DISCHARGE_CATEGORY
        }
        if not survivor_marginal:
            raise ValueError(
                "hospitalization discharge_category marginal has no non-death category "
                "to draw a survivor disposition from"
            )
        discharge_category = categorical(survivor_marginal, rng)
        death_dttm = None

    return HospitalizationRecord(
        patient_id=patient_id,
        hospitalization_id=hospitalization_id,
        hospitalization_joined_id=hospitalization_id,
        admission_dttm=admit_dttm,
        discharge_dttm=discharge_dttm,
        admission_type_category=admission_type,
        admission_type_name=admission_type,
        discharge_category=discharge_category,
        discharge_name=discharge_category,
        death_dttm=death_dttm,
    )


def hospitalization_frame(records: list[HospitalizationRecord]) -> pl.DataFrame:
    """Stack sampled encounters into one conformant ``hospitalization`` frame."""
    return pl.DataFrame(
        {
            "patient_id": [r.patient_id for r in records],
            "hospitalization_id": [r.hospitalization_id for r in records],
            "hospitalization_joined_id": [r.hospitalization_joined_id for r in records],
            "admission_dttm": [r.admission_dttm for r in records],
            "discharge_dttm": [r.discharge_dttm for r in records],
            "admission_type_name": [r.admission_type_name for r in records],
            "admission_type_category": [r.admission_type_category for r in records],
            "discharge_name": [r.discharge_name for r in records],
            "discharge_category": [r.discharge_category for r in records],
        },
        schema={
            "patient_id": pl.String,
            "hospitalization_id": pl.String,
            "hospitalization_joined_id": pl.String,
            "admission_dttm": _UTC_DT,
            "discharge_dttm": _UTC_DT,
            "admission_type_name": pl.String,
            "admission_type_category": pl.String,
            "discharge_name": pl.String,
            "discharge_category": pl.String,
        },
    )

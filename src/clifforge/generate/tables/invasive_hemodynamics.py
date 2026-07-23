"""Tier 6 ``invasive_hemodynamics`` generator (U20; prior-driven, R5, R14, KTD-6).

Pulmonary-artery-catheter measurements are placed in patients with
cardiovascular failure, so measurement events are emitted at a charting cadence
during the spine's ``cv_flag`` windows. There is no fitted block, so cadence uses
a documented constant (R15 — prior-driven, marked in ``PROVENANCE.md``).

The vendored CLIF 2.1.0 dictionary defines this beta table with only
``measure_name`` / ``measure_category`` (no numeric value column), so only the
measurement *event* is emitted — inventing a value column would violate R15.
``measure_category`` values are exact mCIDE members (R5). The spine supplies only
the cv-failure signal (KTD-6); reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame

__all__ = [
    "HemodynamicRow",
    "invasive_hemodynamics_frame",
    "sample_invasive_hemodynamics",
]

_MEASURE_INTERVAL_HOURS = 6.0  # hemodynamics charted a few times a day
#: A standard PA-catheter measure set, weighted toward the routinely-charted ones.
_MEASURE_MARGINAL = {
    "cvp": 0.3,
    "pa_systolic": 0.15,
    "pa_diastolic": 0.15,
    "pa_mean": 0.15,
    "pcwp": 0.15,
    "cardiac_output_thermodilution": 0.1,
}

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class HemodynamicRow:
    """One invasive hemodynamic measurement event."""

    hospitalization_id: str
    recorded_dttm: datetime
    measure_category: str


def _measure_intervals(cv_flag: list[bool], grid_step: float) -> list[int]:
    """cv-failure interval indices at which to chart a hemodynamic measure."""
    intervals: list[int] = []
    last: int | None = None
    for idx, cv in enumerate(cv_flag):
        if not cv:
            continue
        if last is None or (idx - last) * grid_step >= _MEASURE_INTERVAL_HOURS:
            intervals.append(idx)
            last = idx
    return intervals


def sample_invasive_hemodynamics(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[HemodynamicRow]:
    """Emit PA-catheter measurement events during cv-failure windows (R5, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)

    rows: list[HemodynamicRow] = []
    for idx in _measure_intervals(spine.cv_flag, grid_step):
        rows.append(
            HemodynamicRow(
                hospitalization_id=hid,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                measure_category=categorical(_MEASURE_MARGINAL, rng),
            )
        )
    return rows


def invasive_hemodynamics_frame(rows: list[HemodynamicRow]) -> pl.DataFrame:
    """Stack hemodynamic measurement events into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "measure_name": [r.measure_category for r in rows],
            "measure_category": [r.measure_category for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "measure_name": pl.String,
            "measure_category": pl.String,
        },
    )

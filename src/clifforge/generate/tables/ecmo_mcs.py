"""Tier 6 ``ecmo_mcs`` generator (U20; prior-driven, R14, KTD-6).

ECMO / mechanical circulatory support is confined to the sickest patients, so it
is emitted only during the top of the organ-support ladder
(``support_level >= IMV+2`` = the CRRT/ECMO tier). There is no fitted block and no
consortium prior for device parameters, so device flow/sweep/rate use documented
adult VV-ECMO literature values (R15 — prior-driven, marked in ``PROVENANCE.md``,
not fitted). The spine supplies only acuity (KTD-6); reproducible under a fixed
``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["EcmoRow", "ecmo_mcs_frame", "sample_ecmo_mcs"]

#: ECMO/MCS is the top of the support ladder (5 = +CRRT/ECMO).
_ECMO_MIN_SUPPORT_LEVEL = 5
_DEVICE_CATEGORY = "VV ECMO"
_MCS_GROUP = "ECMO"
_DEVICE_METRIC = "sweep_speed"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class EcmoRow:
    """One ECMO/MCS charting row."""

    hospitalization_id: str
    recorded_dttm: datetime
    device_category: str
    mcs_group: str
    device_metric_name: str
    device_rate: float
    flow: float
    sweep: float


def sample_ecmo_mcs(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[EcmoRow]:
    """Emit ECMO/MCS rows during the highest-acuity (ECMO-tier) intervals (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)

    rows: list[EcmoRow] = []
    for idx, level in enumerate(spine.support_level):
        if level < _ECMO_MIN_SUPPORT_LEVEL:
            continue
        rows.append(
            EcmoRow(
                hospitalization_id=hid,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                device_category=_DEVICE_CATEGORY,
                mcs_group=_MCS_GROUP,
                device_metric_name=_DEVICE_METRIC,
                device_rate=round(float(rng.uniform(2500.0, 3500.0)), 0),  # pump RPM
                flow=round(float(rng.uniform(3.5, 5.0)), 2),  # L/min
                sweep=round(float(rng.uniform(2.0, 6.0)), 1),  # L/min
            )
        )
    return rows


def ecmo_mcs_frame(rows: list[EcmoRow]) -> pl.DataFrame:
    """Stack ECMO/MCS rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "device_name": [r.device_category for r in rows],
            "device_category": [r.device_category for r in rows],
            "mcs_group": [r.mcs_group for r in rows],
            "device_metric_name": [r.device_metric_name for r in rows],
            "device_rate": [r.device_rate for r in rows],
            "flow": [r.flow for r in rows],
            "sweep": [r.sweep for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": _UTC_DT,
            "device_name": pl.String,
            "device_category": pl.String,
            "mcs_group": pl.String,
            "device_metric_name": pl.String,
            "device_rate": pl.Float64,
            "flow": pl.Float64,
            "sweep": pl.Float64,
        },
    )

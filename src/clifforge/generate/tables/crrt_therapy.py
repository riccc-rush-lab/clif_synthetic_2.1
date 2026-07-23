"""Tier 5 ``crrt_therapy`` generator (U18; R9, R12, KTD-6).

Continuous renal replacement is a wide device-parameter table charted while a
patient is on CRRT. No fitted block exists, so CRRT sessions are driven by the
spine's **renal-failure flag** (KTD-6/R12): a wide row is emitted per charting
interval where the flag is set, and none otherwise. The mode is CVVHDF (the
common ICU modality); blood/dialysate/replacement-fluid rates and ultrafiltration
are documented in-bounds constants with jitter (un-fitted, like the adt
constants), kept inside the consortium outlier bounds (R9) — not invented
distributions (R15).

``device_id`` is synthesized per stay. Output is reproducible byte-for-byte under
a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["CrrtRow", "crrt_therapy_frame", "sample_crrt_therapy"]

_CRRT_MODE = "cvvhdf"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CrrtRow:
    """One CRRT charting row (device parameters at one time)."""

    hospitalization_id: str
    device_id: str
    recorded_dttm: datetime
    crrt_mode_category: str
    blood_flow_rate: float
    pre_filter_replacement_fluid_rate: float
    post_filter_replacement_fluid_rate: float
    dialysate_flow_rate: float
    ultrafiltration_out: float


def sample_crrt_therapy(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[CrrtRow]:
    """Emit one hospitalization's CRRT rows during renal-failure windows (R12, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    device_id = f"{hid}-CRRT"

    rows: list[CrrtRow] = []
    for idx, renal in enumerate(spine.renal_flag):
        if not renal:
            continue
        rows.append(
            CrrtRow(
                hospitalization_id=hid,
                device_id=device_id,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                crrt_mode_category=_CRRT_MODE,
                blood_flow_rate=round(float(rng.uniform(180.0, 240.0)), 1),
                pre_filter_replacement_fluid_rate=round(float(rng.uniform(400.0, 800.0)), 1),
                post_filter_replacement_fluid_rate=round(float(rng.uniform(400.0, 800.0)), 1),
                dialysate_flow_rate=round(float(rng.uniform(1500.0, 2500.0)), 1),
                ultrafiltration_out=round(float(rng.uniform(50.0, 250.0)), 1),
            )
        )
    return rows


def crrt_therapy_frame(rows: list[CrrtRow]) -> pl.DataFrame:
    """Stack CRRT rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "device_id": [r.device_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "crrt_mode_name": [r.crrt_mode_category for r in rows],
            "crrt_mode_category": [r.crrt_mode_category for r in rows],
            "blood_flow_rate": [r.blood_flow_rate for r in rows],
            "pre_filter_replacement_fluid_rate": [
                r.pre_filter_replacement_fluid_rate for r in rows
            ],
            "post_filter_replacement_fluid_rate": [
                r.post_filter_replacement_fluid_rate for r in rows
            ],
            "dialysate_flow_rate": [r.dialysate_flow_rate for r in rows],
            "ultrafiltration_out": [r.ultrafiltration_out for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "device_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "crrt_mode_name": pl.String,
            "crrt_mode_category": pl.String,
            "blood_flow_rate": pl.Float64,
            "pre_filter_replacement_fluid_rate": pl.Float64,
            "post_filter_replacement_fluid_rate": pl.Float64,
            "dialysate_flow_rate": pl.Float64,
            "ultrafiltration_out": pl.Float64,
        },
    )

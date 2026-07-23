"""Tier 2 ``adt`` (admission/discharge/transfer) generator (U9; R8).

The ``adt`` table records a hospitalization's location movements. There is no
fitted ``adt`` block in the parameter pack, so movements are derived **entirely
from the latent spine** (KTD-6): the per-interval acuity trajectory is
run-length-encoded into contiguous ``ward`` / ``icu`` location segments, placed
on the same ``admit_dttm`` + grid timeline the hospitalization generator uses so
the last ``out_dttm`` coincides with the encounter's discharge.

Acuity -> location heuristic: an interval whose support level is at or above
:data:`ICU_MIN_SUPPORT_LEVEL` (high-flow O2 / NIV and above) is ``icu``, else
``ward``. This is the plan's "ICU windows align with high-acuity spine segments"
made concrete; it is a heuristic, not a fitted mapping, because MIMIC ``adt``
location structure was not fitted by U5.

The resulting ICU segments are exposed via :func:`icu_windows` — the sole channel
by which later tiers (vitals, labs, …) restrict observations to ICU time, read
through the spine/orchestrator rather than by cross-reading this table.

Un-fitted fields use documented MIMIC-appropriate constants rather than invented
distributions (R15): ``hospital_type = "academic"`` (MIMIC is a single academic
center), ``location_type = "medical_icu"`` for ICU rows (null off-ICU). Movements
are a deterministic function of the spine, so they are reproducible under any
``rng`` (R22); ``rng`` is accepted for the uniform generator signature but unused.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["ICU_MIN_SUPPORT_LEVEL", "AdtMovement", "adt_frame", "icu_windows", "sample_adt"]


#: MIMIC-appropriate constants for fields with no fitted distribution (R15).
_HOSPITAL_TYPE = "academic"
_ICU_LOCATION_TYPE = "medical_icu"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class AdtMovement:
    """One contiguous location stay within a hospitalization."""

    hospitalization_id: str
    hospital_id: str
    hospital_type: str
    in_dttm: datetime
    out_dttm: datetime
    location_name: str
    location_category: str
    location_type: str | None


def _location_segments(support_level: list[int]) -> list[tuple[str, int]]:
    """Run-length-encode the acuity trajectory into ``(category, n_intervals)``."""
    segments: list[tuple[str, int]] = []
    for level in support_level:
        category = "icu" if level >= ICU_MIN_SUPPORT_LEVEL else "ward"
        if segments and segments[-1][0] == category:
            prev_cat, prev_n = segments[-1]
            segments[-1] = (prev_cat, prev_n + 1)
        else:
            segments.append((category, 1))
    return segments


def sample_adt(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    hospital_id: str = "HOSP0",
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[AdtMovement]:
    """Emit one hospitalization's ordered, contiguous location movements (R8, R22).

    Segments tile ``[admit_dttm, admit_dttm + n_intervals * grid_step]`` with no
    gaps or overlaps; the terminal ``out_dttm`` equals the hospitalization's
    discharge (both use the pack grid). ``hospitalization_id`` defaults to the
    spine's own id.
    """
    del rng  # deterministic from the spine; accepted for signature uniformity
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)

    movements: list[AdtMovement] = []
    cursor = admit_dttm
    for category, n_int in _location_segments(spine.support_level):
        out = cursor + timedelta(hours=n_int * grid_step)
        movements.append(
            AdtMovement(
                hospitalization_id=hid,
                hospital_id=hospital_id,
                hospital_type=_HOSPITAL_TYPE,
                in_dttm=cursor,
                out_dttm=out,
                location_name=category,
                location_category=category,
                location_type=_ICU_LOCATION_TYPE if category == "icu" else None,
            )
        )
        cursor = out
    return movements


def icu_windows(movements: list[AdtMovement]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Map ``hospitalization_id -> [(in_dttm, out_dttm), …]`` over ICU stays only.

    This is the channel later tiers use to keep observations inside ICU time
    (KTD-6). A hospitalization with no ICU segment is absent from the mapping.
    """
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for m in movements:
        if m.location_category == "icu":
            windows.setdefault(m.hospitalization_id, []).append((m.in_dttm, m.out_dttm))
    return windows


def adt_frame(movements: list[AdtMovement]) -> pl.DataFrame:
    """Stack movements into one conformant ``adt`` frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [m.hospitalization_id for m in movements],
            "hospital_id": [m.hospital_id for m in movements],
            "hospital_type": [m.hospital_type for m in movements],
            "in_dttm": [m.in_dttm for m in movements],
            "out_dttm": [m.out_dttm for m in movements],
            "location_name": [m.location_name for m in movements],
            "location_category": [m.location_category for m in movements],
            "location_type": [m.location_type for m in movements],
        },
        schema={
            "hospitalization_id": pl.String,
            "hospital_id": pl.String,
            "hospital_type": pl.String,
            "in_dttm": _UTC_DT,
            "out_dttm": _UTC_DT,
            "location_name": pl.String,
            "location_category": pl.String,
            "location_type": pl.String,
        },
    )

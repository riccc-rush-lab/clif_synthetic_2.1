"""Tier 4 ``medication_admin_continuous`` generator (U13; R11, AE3, KTD-6).

The pack fits only per-med **infusion hazards** (``stop_hazard`` and mean run
length), not start rates or doses, so continuous infusions are driven by the
latent spine with those hazards layered on: a **vasopressor** (norepinephrine)
runs during the spine's cardiovascular-failure windows, and a **sedative**
(propofol) runs during invasive ventilation (``support_level >= 3``). Both couple
to the same spine that drives U10 hypotension, so norepinephrine co-occurs with
low blood pressure without either table reading the other (KTD-6).

**R11 / AE3 — rate-encoded, stop = new zero-dose row, no boluses.** ``med_dose``
is an infusion *rate* (``med_dose_unit`` a rate unit); a stop is emitted as a
**new** row with ``med_dose = 0`` and ``mar_action_category = "stop"`` — prior
rows are never mutated and there are no bolus rows. Within an active window the
pack ``stop_hazard`` can end an infusion, which restarts if the coupling still
holds, producing realistic on/off cycling.

Doses are documented clinical rate ranges (un-fitted, like the adt constants),
not invented distributions (R15). ``med_order_id`` is synthesized per infusion.
Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import IMV_MIN_SUPPORT_LEVEL, UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = [
    "MedAdminRow",
    "medication_admin_continuous_frame",
    "sample_medication_admin_continuous",
]

_DOSE_UNIT = "mcg/kg/min"
_ROUTE = "iv"
#: (med_category, documented rate range) for the two spine-coupled infusions.
_VASOPRESSOR = "norepinephrine"
_SEDATIVE = "propofol"
_DOSE_RANGE: dict[str, tuple[float, float]] = {
    _VASOPRESSOR: (0.02, 0.5),
    _SEDATIVE: (5.0, 50.0),
}
_DEFAULT_STOP_HAZARD = 0.2

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class MedAdminRow:
    """One continuous-med administration event (a start, or a zero-dose stop)."""

    hospitalization_id: str
    med_order_id: str
    admin_dttm: datetime
    med_category: str
    med_route_category: str
    med_dose: float
    med_dose_unit: str
    mar_action_category: str


def _stop_hazard(pack: ParamPack, med: str) -> float:
    block = pack.tables.get("medication_admin_continuous")
    if block is None or "params" not in block:
        return _DEFAULT_STOP_HAZARD
    hazards = block["params"].get("infusion_hazards", {})
    return float(hazards.get(med, {}).get("stop_hazard", _DEFAULT_STOP_HAZARD))


def _infusion_rows(
    hid: str,
    med: str,
    active: list[bool],
    pack: ParamPack,
    rng: np.random.Generator,
    admit_dttm: datetime,
    grid_step: float,
    order_seq: Iterator[int],
) -> list[MedAdminRow]:
    """Emit start/stop rows for one med gated on its per-interval ``active`` mask."""
    lo, hi = _DOSE_RANGE[med]
    stop_hazard = _stop_hazard(pack, med)
    rows: list[MedAdminRow] = []
    running = False
    order_id = ""

    def row(dose: float, action: str, t: int) -> MedAdminRow:
        return MedAdminRow(
            hospitalization_id=hid,
            med_order_id=order_id,
            admin_dttm=admit_dttm + timedelta(hours=t * grid_step),
            med_category=med,
            med_route_category=_ROUTE,
            med_dose=round(dose, 4),
            med_dose_unit=_DOSE_UNIT,
            mar_action_category=action,
        )

    for t, on in enumerate(active):
        if on and not running:
            running = True
            order_id = f"{hid}-{med}-{next(order_seq)}"
            rows.append(row(float(rng.uniform(lo, hi)), "start", t))
        elif on and running and rng.random() < stop_hazard:
            rows.append(row(0.0, "stop", t))  # AE3: new zero-dose stop row
            running = False
        elif not on and running:
            rows.append(row(0.0, "stop", t))
            running = False
    if running:
        rows.append(row(0.0, "stop", len(active)))  # stop at discharge
    return rows


def sample_medication_admin_continuous(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[MedAdminRow]:
    """Emit one hospitalization's continuous-med rows (R11, AE3, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    order_seq = iter(range(10**6))

    vaso_active = list(spine.cv_flag)
    sed_active = [level >= IMV_MIN_SUPPORT_LEVEL for level in spine.support_level]

    rows = _infusion_rows(
        hid, _VASOPRESSOR, vaso_active, pack, rng, admit_dttm, grid_step, order_seq
    )
    rows += _infusion_rows(hid, _SEDATIVE, sed_active, pack, rng, admit_dttm, grid_step, order_seq)
    rows.sort(key=lambda r: (r.admin_dttm, r.med_category, r.mar_action_category))
    return rows


def medication_admin_continuous_frame(rows: list[MedAdminRow]) -> pl.DataFrame:
    """Stack med-admin events into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "med_order_id": [r.med_order_id for r in rows],
            "admin_dttm": [r.admin_dttm for r in rows],
            "med_name": [r.med_category for r in rows],
            "med_category": [r.med_category for r in rows],
            "med_route_name": [r.med_route_category for r in rows],
            "med_route_category": [r.med_route_category for r in rows],
            "med_dose": [r.med_dose for r in rows],
            "med_dose_unit": [r.med_dose_unit for r in rows],
            "mar_action_name": [r.mar_action_category for r in rows],
            "mar_action_category": [r.mar_action_category for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "med_order_id": pl.String,
            "admin_dttm": UTC_DATETIME,
            "med_name": pl.String,
            "med_category": pl.String,
            "med_route_name": pl.String,
            "med_route_category": pl.String,
            "med_dose": pl.Float64,
            "med_dose_unit": pl.String,
            "mar_action_name": pl.String,
            "mar_action_category": pl.String,
        },
    )

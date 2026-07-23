"""Shared generate-stage thresholds and the pack grid-step accessor.

Single source of truth for the two organ-support thresholds and the grid-step
helper that many sibling table generators consume. Defining them once keeps the
safety-relevant clinical thresholds from drifting across a dozen files (a
name-based grep for a threshold used to miss half its copies, since the same
value appeared under several private names).

The ordinal ``support_level`` spine ladder: 0 room-air, 1 low-flow O2,
2 high-flow/NIV, 3 IMV, 4 +vasopressor, 5 +CRRT/ECMO.
"""

from __future__ import annotations

import polars as pl

from clifforge.fit.param_pack import ParamPack

__all__ = [
    "ICU_MIN_SUPPORT_LEVEL",
    "IMV_MIN_SUPPORT_LEVEL",
    "UTC_DATETIME",
    "grid_step_hours",
]

#: The polars dtype for every tz-aware UTC datetime column (R7). Shared so a
#: generator's frame schema never drifts from the conformance gate's expectation.
UTC_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")

#: ``support_level >=`` this is ICU-level care (high-flow O2 / NIV and above).
#: Used to place adt ICU segments and to gate ICU-only observations.
ICU_MIN_SUPPORT_LEVEL = 2

#: ``support_level >=`` this is invasive mechanical ventilation. The same
#: threshold marks sedation presence (ventilated patients are sedated), enables
#: proning (intubated ARDS), and drives the IMV device — one fact, one constant.
IMV_MIN_SUPPORT_LEVEL = 3


def grid_step_hours(pack: ParamPack) -> float:
    """Hours per spine interval (from the pack's spine block; default 1.0).

    The AR(1)/hazard params are only valid at the grid the pack was fitted on, so
    every generator turns interval indices into real durations through this one
    accessor.
    """
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        return 1.0
    return float(block["params"].get("state_model", {}).get("grid_step_hours", 1.0))

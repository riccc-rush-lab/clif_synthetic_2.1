"""Tier 3 ``labs`` generator (U11; R9, R12, KTD-4, KTD-6).

Labs are drawn from a **hand-rolled Gaussian copula** (no SDV/copulas dependency,
R-stack constraint): the pack carries a positive-definite Spearman correlation
matrix over labs measured together (``lab_correlation``, indexed by ``lab_order``),
per-lab log-normal marginals fit on ``log1p`` values (``lab_marginals``), and
per-hospitalization presence rates (``lab_presence``).

Generation is **sample-then-mask**, never impute-then-sparsify, so no CLIF
missingness artifact is manufactured (KTD-4):

1. Draw the present-set once per hospitalization — each lab is present for the
   stay with its fitted ``lab_presence`` probability. This matches the fit
   definition (presence = fraction of hospitalizations with >=1 measurement).
2. At each order time in the stay's ICU windows, draw the **full** correlated
   45-vector from the copula (``z = L @ N(0, I)`` with ``L`` the Cholesky factor
   of the correlation), map each component through its log-normal marginal
   (``value = expm1(log_mean + log_sd * z)``), and emit a row **only** for labs
   in the present-set. Absent labs are native nulls (no row), never imputed.

Every value is clamped into the consortium outlier bounds (R9). Creatinine and
bun are shifted up in log space when the spine's renal-failure flag is set at the
order interval — a documented R12 clinical coupling (the copula itself is
acuity-agnostic, so this is an explicit coupling, not a fitted mechanism), kept
to classic renal markers rather than invented broadly (R15). The spine is the
only cross-table channel (KTD-6); this generator never reads another table.

Un-fitted columns (collect/result timestamps, order/specimen category, LOINC,
reference unit) are omitted rather than fabricated (R15; the schema is
permissive). Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.reference import bounds

__all__ = ["LabObservation", "labs_frame", "sample_labs"]

#: At or above this support level an interval is ICU-level care (matches adt/vitals).
_ICU_MIN_SUPPORT_LEVEL = 2

#: Target spacing between lab panels within ICU time (labs are ~daily in the ICU).
_LAB_PANEL_INTERVAL_HOURS = 24.0

#: Classic renal-failure markers shifted up when the spine renal flag is set, and
#: the additive shift in log1p space (~doubles creatinine) — a documented R12
#: clinical coupling, not a fitted quantity.
_RENAL_MARKERS = frozenset({"creatinine", "bun"})
_RENAL_LOG_SHIFT = 0.5

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class LabObservation:
    """One observed lab result in the long ``labs`` table."""

    hospitalization_id: str
    lab_order_dttm: datetime
    lab_name: str
    lab_category: str
    lab_value: str
    lab_value_numeric: float


def _labs_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("labs")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'labs' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _grid_step_hours(pack: ParamPack) -> float:
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        return 1.0
    return float(block["params"].get("state_model", {}).get("grid_step_hours", 1.0))


def _cholesky(correlation: list[list[float]]) -> npt.NDArray[np.float64]:
    """Cholesky factor of the copula correlation, jittered onto the PD cone.

    The fit projects to the nearest PD matrix, but round-tripping through JSON can
    leave it merely PSD; escalating diagonal jitter recovers a usable factor.
    """
    mat = np.asarray(correlation, dtype=float)
    eye = np.eye(mat.shape[0])
    for jitter in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        try:
            return np.linalg.cholesky(mat + jitter * eye)
        except np.linalg.LinAlgError:
            continue
    raise ValueError("lab correlation matrix is not positive semi-definite")


def _panel_intervals(support_level: list[int], grid_step: float) -> list[int]:
    """ICU interval indices at which to draw a lab panel (~daily within ICU)."""
    panels: list[int] = []
    last: int | None = None
    for idx, level in enumerate(support_level):
        if level < _ICU_MIN_SUPPORT_LEVEL:
            continue
        if last is None or (idx - last) * grid_step >= _LAB_PANEL_INTERVAL_HOURS:
            panels.append(idx)
            last = idx
    return panels


def _clamp(value: float, lab: str) -> float:
    try:
        lower, upper = bounds("labs", lab)
    except (LookupError, ValueError):
        # reference.bounds raises ReferenceDataError (a LookupError) when a lab has
        # no fitted outlier bounds — leave the value as drawn rather than crash.
        return value
    return min(max(value, lower), upper)


def sample_labs(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[LabObservation]:
    """Emit one hospitalization's observed labs as a long list of rows (R9, R22).

    Draws the stay's present-set from ``lab_presence`` once, then a full correlated
    copula vector per ICU order time, emitting rows only for present labs.
    ``hospitalization_id`` defaults to the spine's own id.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    params = _labs_params(pack)
    order: list[str] = params["lab_order"]
    marginals: dict[str, dict[str, float]] = params["lab_marginals"]
    presence: dict[str, float] = params["lab_presence"]
    chol = _cholesky(params["lab_correlation"])
    grid_step = _grid_step_hours(pack)
    n = len(order)

    # (1) present-set for the whole stay — one vector draw, matching the fit's
    #     per-hospitalization presence definition.
    presence_vec = np.array([presence.get(lab, 0.0) for lab in order], dtype=float)
    present_mask = rng.random(n) < presence_vec

    observations: list[LabObservation] = []
    for interval_idx in _panel_intervals(spine.support_level, grid_step):
        z = chol @ rng.standard_normal(n)  # (2) correlated latent draw per panel
        jitter = rng.random() * grid_step
        order_dttm = admit_dttm + timedelta(hours=interval_idx * grid_step + jitter)
        renal = spine.renal_flag[interval_idx]
        for i, lab in enumerate(order):
            if not present_mask[i]:
                continue
            marg = marginals.get(lab)
            if marg is None:
                continue
            log_val = marg["log_mean"] + marg["log_sd"] * float(z[i])
            if renal and lab in _RENAL_MARKERS:
                log_val += _RENAL_LOG_SHIFT  # R12 renal coupling
            value = _clamp(float(np.expm1(log_val)), lab)
            value = round(value, 4)
            observations.append(
                LabObservation(
                    hospitalization_id=hid,
                    lab_order_dttm=order_dttm,
                    lab_name=lab,
                    lab_category=lab,
                    lab_value=f"{value:g}",
                    lab_value_numeric=value,
                )
            )

    observations.sort(key=lambda o: (o.lab_order_dttm, o.lab_category))
    return observations


def labs_frame(observations: list[LabObservation]) -> pl.DataFrame:
    """Stack observed labs into one conformant long ``labs`` frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [o.hospitalization_id for o in observations],
            "lab_order_dttm": [o.lab_order_dttm for o in observations],
            "lab_name": [o.lab_name for o in observations],
            "lab_category": [o.lab_category for o in observations],
            "lab_value": [o.lab_value for o in observations],
            "lab_value_numeric": [o.lab_value_numeric for o in observations],
        },
        schema={
            "hospitalization_id": pl.String,
            "lab_order_dttm": _UTC_DT,
            "lab_name": pl.String,
            "lab_category": pl.String,
            "lab_value": pl.String,
            "lab_value_numeric": pl.Float64,
        },
    )

"""Tier 4 ``position`` generator (U15; R12, KTD-6/KTD-7).

No fitted block exists, so positioning is derived from the latent spine. Prone
positioning is an ARDS therapy for intubated, severely hypoxemic patients, so
prone episodes concentrate where the spine marks **severe hypoxemia** — the
respiratory-failure flag set *and* invasive ventilation (``support_level >= 3``),
which keeps proning consistent with IMV periods without reading the
respiratory_support table (KTD-6). Elsewhere the patient is overwhelmingly
``not_prone``.

One position per row (mutually exclusive ``position_category`` ∈ mCIDE). Prone
probabilities are documented care constants (R15), not fitted. Output is
reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, IMV_MIN_SUPPORT_LEVEL, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["PositionRow", "position_frame", "sample_position"]

_POSITION_INTERVAL_HOURS = 6.0  # positioning charted a few times a day

#: Documented prone probabilities per position check (un-fitted care constants).
_PRONE_PROB_SEVERE = 0.65
_PRONE_PROB_OTHERWISE = 0.03

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class PositionRow:
    """One position event (exactly one position per row)."""

    hospitalization_id: str
    recorded_dttm: datetime
    position_category: str


def _position_intervals(support_level: list[int], grid_step: float) -> list[int]:
    """ICU interval indices at which position is charted (~every few hours)."""
    intervals: list[int] = []
    last: int | None = None
    for idx, level in enumerate(support_level):
        if level < ICU_MIN_SUPPORT_LEVEL:
            continue
        if last is None or (idx - last) * grid_step >= _POSITION_INTERVAL_HOURS:
            intervals.append(idx)
            last = idx
    return intervals


def sample_position(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[PositionRow]:
    """Emit one hospitalization's position events (R12, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)

    rows: list[PositionRow] = []
    for idx in _position_intervals(spine.support_level, grid_step):
        severe = spine.resp_flag[idx] and spine.support_level[idx] >= IMV_MIN_SUPPORT_LEVEL
        prone_prob = _PRONE_PROB_SEVERE if severe else _PRONE_PROB_OTHERWISE
        category = "prone" if rng.random() < prone_prob else "not_prone"
        rows.append(
            PositionRow(
                hospitalization_id=hid,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                position_category=category,
            )
        )
    return rows


def position_frame(rows: list[PositionRow]) -> pl.DataFrame:
    """Stack position events into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "position_name": [r.position_category for r in rows],
            "position_category": [r.position_category for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": _UTC_DT,
            "position_name": pl.String,
            "position_category": pl.String,
        },
    )

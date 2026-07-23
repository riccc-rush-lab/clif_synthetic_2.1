"""Tier 6 ``therapy_details`` generator (U20; prior-driven, R14, KTD-6).

Rehab therapy detail rows accompany the mobilization ordered for a subset of ICU
stays. There is no fitted block and the vendored 2.1.0 dictionary leaves the
element columns free text (no mCIDE list), so documented PT/OT session elements
are used (R15 — prior-driven, marked in ``PROVENANCE.md``). ``session_start_dttm``
is a string column per the dictionary, emitted as an ISO-8601 UTC timestamp. The
spine supplies only the stay horizon (KTD-6); reproducible under a fixed ``rng``
(R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["TherapyDetailRow", "sample_therapy_details", "therapy_details_frame"]

_REHAB_PROB = 0.5
_SESSION_INTERVAL_HOURS = 24.0
#: Documented PT/OT session elements (element_category -> value).
_SESSION_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("mobility", "sat_at_edge_of_bed"),
    ("activity_tolerance", "moderate"),
)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class TherapyDetailRow:
    """One therapy-session detail element."""

    hospitalization_id: str
    session_start_dttm: str
    therapy_element_category: str
    therapy_element_value: str


def sample_therapy_details(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[TherapyDetailRow]:
    """Emit therapy-session detail rows for a subset of ICU stays (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    icu_intervals = [i for i, lvl in enumerate(spine.support_level) if lvl >= ICU_MIN_SUPPORT_LEVEL]
    if not icu_intervals or rng.random() >= _REHAB_PROB:
        return []

    stride = max(1, round(_SESSION_INTERVAL_HOURS / grid_step))
    rows: list[TherapyDetailRow] = []
    for idx in range(icu_intervals[0], icu_intervals[-1] + 1, stride):
        session_start = (admit_dttm + timedelta(hours=idx * grid_step)).isoformat()
        for category, value in _SESSION_ELEMENTS:
            rows.append(TherapyDetailRow(hid, session_start, category, value))
    return rows


def therapy_details_frame(rows: list[TherapyDetailRow]) -> pl.DataFrame:
    """Stack therapy detail rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "session_start_dttm": [r.session_start_dttm for r in rows],
            "therapy_element_name": [r.therapy_element_category for r in rows],
            "therapy_element_category": [r.therapy_element_category for r in rows],
            "therapy_element_value": [r.therapy_element_value for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "session_start_dttm": pl.String,
            "therapy_element_name": pl.String,
            "therapy_element_category": pl.String,
            "therapy_element_value": pl.String,
        },
    )

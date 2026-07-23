"""Tier 6 ``provider`` generator (U20; prior-driven, R14, KTD-6).

Every hospitalization has a care team, so each stay gets a small documented set
of provider assignments (an attending plus a bedside nurse) spanning admission to
discharge. There is no fitted block and the vendored 2.1.0 dictionary leaves
``provider_role_category`` free text (no mCIDE list), so documented role labels
are used (R15 — prior-driven, marked in ``PROVENANCE.md``). ``provider_id`` is
synthesized per role. The spine supplies only the stay horizon (KTD-6);
reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["ProviderRow", "provider_frame", "sample_provider"]

#: Documented care-team roles assigned for the whole stay.
_ROLES: tuple[str, ...] = ("Attending", "Nurse")

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)
_UTC_DT = pl.Datetime(time_unit="us", time_zone="UTC")


@dataclass(frozen=True)
class ProviderRow:
    """One provider assignment spanning the stay."""

    hospitalization_id: str
    provider_id: str
    start_dttm: datetime
    stop_dttm: datetime
    provider_role_category: str


def sample_provider(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[ProviderRow]:
    """Emit the stay's provider assignments (R22).

    ``rng`` is accepted for signature uniformity; provider roles are deterministic
    from the stay, so the result is trivially reproducible.
    """
    del rng
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)
    discharge = admit_dttm + timedelta(hours=los_hours)

    return [
        ProviderRow(
            hospitalization_id=hid,
            provider_id=f"{hid}-{role}",
            start_dttm=admit_dttm,
            stop_dttm=discharge,
            provider_role_category=role,
        )
        for role in _ROLES
    ]


def provider_frame(rows: list[ProviderRow]) -> pl.DataFrame:
    """Stack provider assignments into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "provider_id": [r.provider_id for r in rows],
            "start_dttm": [r.start_dttm for r in rows],
            "stop_dttm": [r.stop_dttm for r in rows],
            "provider_role_name": [r.provider_role_category for r in rows],
            "provider_role_category": [r.provider_role_category for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "provider_id": pl.String,
            "start_dttm": _UTC_DT,
            "stop_dttm": _UTC_DT,
            "provider_role_name": pl.String,
            "provider_role_category": pl.String,
        },
    )

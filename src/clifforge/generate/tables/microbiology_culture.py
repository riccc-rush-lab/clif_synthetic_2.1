"""Tier 5 ``microbiology_culture`` generator (U17; R5, KTD-6).

Cultures are **sparse** events (a handful per stay at most). No fitted block
exists, so a documented per-ICU-day culture rate (un-fitted, like the adt
constants) draws a small Poisson count per stay; each culture picks a fluid,
method, and organism group from documented marginals — heavily weighted toward
``no_growth``, matching real ICU microbiology. Timestamps respect the clinical
order ``order_dttm <= collect_dttm < result_dttm`` (result lands after a
turnaround). The specific ``organism_category`` (543 organisms) has no fitted
mapping, so only the coarse ``organism_group`` is populated; ``organism_category``
is left null (R15; the schema is permissive).

The spine supplies only the stay horizon (KTD-6). Categories are exact mCIDE
members (R5); output is reproducible byte-for-byte under a fixed ``rng`` (R22).
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
    "CultureEvent",
    "microbiology_culture_frame",
    "sample_microbiology_culture",
]

#: Expected cultures per ICU day (documented sparsity constant, un-fitted).
_CULTURES_PER_DAY = 0.4

_FLUID_MARGINAL = {
    "blood_buffy": 0.5,
    "respiratory_tract_lower": 0.2,
    "genito_urinary_tract": 0.2,
    "skin_unspecified": 0.1,
}
_METHOD_MARGINAL = {"culture": 0.8, "gram_stain": 0.2}
#: Organism groups, heavily weighted to no growth (realistic ICU yield).
_ORGANISM_GROUP_MARGINAL = {
    "no_growth": 0.65,
    "staphylococcus_coag_pos": 0.1,
    "escherichia": 0.1,
    "klebsiella": 0.08,
    "pseudomonas_wo_cepacia_maltophilia": 0.07,
}

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CultureEvent:
    """One microbiology culture (order/collect/result + categories)."""

    hospitalization_id: str
    order_dttm: datetime
    collect_dttm: datetime
    result_dttm: datetime
    fluid_category: str
    method_category: str
    organism_group: str


def sample_microbiology_culture(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[CultureEvent]:
    """Emit one hospitalization's sparse culture events (R5, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)
    if los_hours <= 0:
        return []

    n_cultures = int(rng.poisson(_CULTURES_PER_DAY * los_hours / 24.0))
    events: list[CultureEvent] = []
    for _ in range(n_cultures):
        order = admit_dttm + timedelta(hours=float(rng.random()) * los_hours)
        collect = order + timedelta(minutes=float(rng.uniform(5.0, 60.0)))
        result = collect + timedelta(hours=float(rng.uniform(24.0, 72.0)))
        events.append(
            CultureEvent(
                hospitalization_id=hid,
                order_dttm=order,
                collect_dttm=collect,
                result_dttm=result,
                fluid_category=categorical(_FLUID_MARGINAL, rng),
                method_category=categorical(_METHOD_MARGINAL, rng),
                organism_group=categorical(_ORGANISM_GROUP_MARGINAL, rng),
            )
        )
    events.sort(key=lambda e: e.order_dttm)
    return events


def microbiology_culture_frame(events: list[CultureEvent]) -> pl.DataFrame:
    """Stack culture events into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [e.hospitalization_id for e in events],
            "order_dttm": [e.order_dttm for e in events],
            "collect_dttm": [e.collect_dttm for e in events],
            "result_dttm": [e.result_dttm for e in events],
            "fluid_name": [e.fluid_category for e in events],
            "fluid_category": [e.fluid_category for e in events],
            "method_name": [e.method_category for e in events],
            "method_category": [e.method_category for e in events],
            "organism_group": [e.organism_group for e in events],
        },
        schema={
            "hospitalization_id": pl.String,
            "order_dttm": UTC_DATETIME,
            "collect_dttm": UTC_DATETIME,
            "result_dttm": UTC_DATETIME,
            "fluid_name": pl.String,
            "fluid_category": pl.String,
            "method_name": pl.String,
            "method_category": pl.String,
            "organism_group": pl.String,
        },
    )

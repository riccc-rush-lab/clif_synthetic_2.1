"""Reproducible population derivation: shape a fitted base pack to a target cohort.

The base pack is faithful to its source center's demographics. A *population*
derivation re-weights that pack to a target population — here a Chicago-
representative, age-shifted ICU cohort — so a re-fit can be re-layered from code
instead of interactive steps. This reads the real dataset (a fit-stage-adjacent
operation, KTD-1) only to derive **aggregate** quantities: shifted age quantiles
and the continuous-med category marginal; the demographic re-weighting is a set of
documented target constants.
"""

from __future__ import annotations

import copy
from pathlib import Path

import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.reference import categories

__all__ = ["CHICAGO_ETHNICITY_TARGET", "CHICAGO_RACE_TARGET", "derive_chicago_population"]

#: Chicago-representative race distribution (deliberate re-weighting away from the
#: source center's demographics), using exact mCIDE ``race_category`` members.
CHICAGO_RACE_TARGET: dict[str, float] = {
    "White": 0.42,
    "Black or African American": 0.30,
    "Other": 0.18,
    "Asian": 0.07,
    "American Indian or Alaska Native": 0.008,
    "Native Hawaiian or Other Pacific Islander": 0.002,
    "Unknown": 0.02,
}
#: Chicago-representative ethnicity distribution (mCIDE ``ethnicity_category``).
CHICAGO_ETHNICITY_TARGET: dict[str, float] = {
    "Hispanic": 0.29,
    "Non-Hispanic": 0.69,
    "Unknown": 0.02,
}

#: Continuous-med category string normalization to canonical forms, then a small set
#: of specific renames; non-infusion agents are dropped. Matches the fitted marginal
#: the generator's med path consumes.
_MED_RENAME: dict[str, str] = {
    "dextrose": "dextrose_other",
    "dextrose_in_water_d5w": "dextrose_5_water",
    "albumin_infusion": "albumin",
    "magnesium": "magnesium_sulfate",
    "aminocaproic": "aminocaproic_acid",
}
#: Canonical mCIDE ``med_category`` vocabulary; any med not mapping into it is
#: dropped, so the emitted marginal can never carry a non-conformant category.
_VALID_MED_CATEGORIES: frozenset[str] = frozenset(
    categories("medication_admin_continuous", "med_category")
)


def _normalize_med(raw: str) -> str | None:
    """Canonicalize a raw med_category string to an mCIDE member (or None to drop)."""
    norm = raw.strip().lower().replace(" ", "_")
    norm = _MED_RENAME.get(norm, norm)
    return norm if norm in _VALID_MED_CATEGORIES else None


def _proportions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def derive_chicago_population(
    base_pack: ParamPack,
    real_dir: str | Path,
    *,
    age_shift_years: float = 2.5,
    race_target: dict[str, float] | None = None,
    ethnicity_target: dict[str, float] | None = None,
) -> ParamPack:
    """Return a deep-copied pack re-weighted to the Chicago-representative cohort.

    Overrides the patient race/ethnicity marginals with the target constants,
    replaces ``age_at_admission_quantiles`` with the real age deciles shifted by
    ``age_shift_years``, and fits the continuous-med ``med_category_marginal`` /
    ``mar_action_category_marginal`` from the real data under the normalization
    above. The base pack is not mutated.
    """
    real = Path(real_dir)
    tables = copy.deepcopy(dict(base_pack.tables))

    # 1. Demographic re-weighting (documented target constants).
    patient = tables.setdefault("patient", {"params": {}})
    patient["params"]["race_category_marginal"] = dict(race_target or CHICAGO_RACE_TARGET)
    patient["params"]["ethnicity_category_marginal"] = dict(
        ethnicity_target or CHICAGO_ETHNICITY_TARGET
    )

    # 2. Age quantiles: real deciles shifted up.
    ages = (
        pl.scan_parquet(real / "clif_hospitalization.parquet")
        .select("age_at_admission")
        .drop_nulls()
        .collect()
        .to_series()
    )
    deciles = [float(ages.quantile(i / 10) or 0.0) + age_shift_years for i in range(11)]
    hosp = tables.setdefault("hospitalization", {"params": {}})
    hosp["params"]["age_at_admission_quantiles"] = deciles

    # 3. Continuous-med category + action marginals under normalization.
    med = (
        pl.scan_parquet(real / "clif_medication_admin_continuous.parquet")
        .select("med_category", "mar_action_category")
        .collect()
    )
    med_counts: dict[str, int] = {}
    for row in med.group_by("med_category").len().iter_rows(named=True):
        canon = _normalize_med(row["med_category"])
        if canon is not None:
            med_counts[canon] = med_counts.get(canon, 0) + int(row["len"])
    action_counts = {
        r["mar_action_category"]: int(r["len"])
        for r in med.group_by("mar_action_category").len().iter_rows(named=True)
        if r["mar_action_category"] is not None
    }
    med_block = tables.setdefault("medication_admin_continuous", {"params": {}})
    med_block["params"]["med_category_marginal"] = _proportions(med_counts)
    med_block["params"]["mar_action_category_marginal"] = _proportions(action_counts)

    return ParamPack(manifest=dict(base_pack.manifest), tables=tables)

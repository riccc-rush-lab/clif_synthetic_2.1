"""Expected-cohort profile — the headline realized stats for a generated dataset.

Shared by the CLI ``generate --preview`` dry-run and the Cohort Designer UI so
both report the same numbers from the same computation (no drift). Pure polars,
no UI dependency.
"""

from __future__ import annotations

from typing import Any

import polars as pl

__all__ = ["PREVIEW_SAMPLE", "cohort_profile", "format_profile"]

#: Encounters used for a quick preview sample — small enough to generate in a
#: second or two; the realized rates converge to these shapes at full size.
PREVIEW_SAMPLE = 350


def cohort_profile(dataset: Any) -> dict[str, float]:
    """Return headline realized stats from a ``GeneratedDataset`` (defensive).

    Keys: ``n``, ``mortality``, ``los_median_h``, ``imv``, ``icu``, ``vaso``,
    ``crrt`` (rates as fractions in [0, 1]). Missing optional columns are skipped
    rather than raising, so the profile works on any conformant dataset.
    """
    h = dataset.tables["hospitalization"]
    n = h.height
    out: dict[str, float] = {"n": float(n)}
    if "discharge_category" in h.columns:
        out["mortality"] = float((h["discharge_category"] == "Expired").mean() or 0.0)
    los = (h["discharge_dttm"] - h["admission_dttm"]).dt.total_seconds() / 3600.0
    out["los_median_h"] = float(los.median() or 0.0)

    peak = dataset.truth.group_by("hospitalization_id").agg(
        pl.col("support_level").max().alias("peak"),
        pl.col("cv_flag").cast(pl.Int8).max().alias("cv"),
    )
    out["imv"] = float((peak["peak"] >= 3).mean() or 0.0)
    out["icu"] = float((peak["peak"] >= 2).mean() or 0.0)
    out["vaso"] = float(peak["cv"].mean() or 0.0)
    crrt = dataset.tables.get("crrt_therapy")
    out["crrt"] = (
        float(crrt["hospitalization_id"].n_unique() / n) if crrt is not None and n else 0.0
    )
    return out


def format_profile(profile: dict[str, float]) -> str:
    """Render a cohort profile as an aligned, human-readable text block."""
    rows = [
        ("encounters (sample)", f"{int(profile['n'])}"),
        ("in-hospital mortality", f"{profile.get('mortality', 0.0) * 100:.1f}%"),
        ("hospital LOS (median)", f"{profile['los_median_h']:.0f} h"),
        ("invasive ventilation", f"{profile['imv'] * 100:.0f}%"),
        ("reached ICU", f"{profile['icu'] * 100:.0f}%"),
        ("vasopressors", f"{profile['vaso'] * 100:.0f}%"),
        ("CRRT", f"{profile['crrt'] * 100:.0f}%"),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label.ljust(width)}   {value}" for label, value in rows)

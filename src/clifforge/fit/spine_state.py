"""Latent ICU-acuity state derivation for the empirical-fidelity spine (U5, U6).

The spine is a semi-Markov process over a latent per-interval state. This module
operationalizes that state from real CLIF tables as an **organ-support severity
ladder** plus derived organ-failure flags (the modelling choice recorded for this
pack). The ladder is ordinal in the intensity of life support actually delivered:

    0  none / room air
    1  low-flow O2        (Nasal Cannula, Face Mask, Trach Collar)
    2  high-flow / NIV    (High Flow NC, NIPPV, CPAP)
    3  invasive vent      (IMV)
    4  circulatory support (vasopressor / inotrope infusion active)
    5  renal / mechanical-circulatory support (CRRT or ECMO/MCS active)

``support_level`` at an interval is the **max** of the components active in that
interval — a patient on norepinephrine breathing room air is level 4, a patient
on CRRT is level 5 — so the ladder degrades gracefully to whatever the record
actually contains rather than requiring a strict cumulative combination.

Four organ-failure **flags** are derived alongside the level and drive downstream
table coupling (which labs/meds/assessments a generated patient plausibly has):

    resp_flag  = respiratory support at high-flow/NIV/IMV intensity
    cv_flag    = vasopressor/inotrope infusion or MCS/ECMO active
    renal_flag = CRRT active
    neuro_flag = deep sedation (RASS <= -3) or low GCS (total <= 8)

This module operates purely on polars frames passed in by ``run_fit`` — it never
opens a real-data path itself, so KTD-1 (only ``run_fit`` imports a real path)
holds and the derivation is testable with fabricated frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import polars as pl

__all__ = [
    "Support",
    "SpineStateConfig",
    "DEVICE_SUPPORT_LEVEL",
    "VASOPRESSOR_INOTROPE",
    "MCS_ECMO_CV",
    "derive_state_timeline",
    "outcome_by_hospitalization",
]


class Support(IntEnum):
    """Ordinal organ-support severity ladder (see module docstring)."""

    NONE = 0
    LOW_O2 = 1
    HIGH_O2 = 2
    IMV = 3
    CIRC = 4
    RENAL_MCS = 5


#: respiratory_support ``device_category`` -> support level. ``Other`` and
#: unmapped categories contribute nothing (treated as no evidence, not level 0).
DEVICE_SUPPORT_LEVEL: dict[str, int] = {
    "Room Air": Support.NONE,
    "Nasal Cannula": Support.LOW_O2,
    "Face Mask": Support.LOW_O2,
    "Trach Collar": Support.LOW_O2,
    "High Flow NC": Support.HIGH_O2,
    "NIPPV": Support.HIGH_O2,
    "CPAP": Support.HIGH_O2,
    "IMV": Support.IMV,
}

#: device categories that count as active respiratory support for ``resp_flag``
#: (high-flow / non-invasive / invasive — low-flow O2 and room air excluded).
RESP_FLAG_DEVICES: frozenset[str] = frozenset({"High Flow NC", "NIPPV", "CPAP", "IMV"})

#: medication_admin_continuous ``med_category`` values that are vasopressors or
#: inotropes — an active infusion marks circulatory support (level 4, cv_flag).
VASOPRESSOR_INOTROPE: frozenset[str] = frozenset(
    {
        "norepinephrine",
        "epinephrine",
        "phenylephrine",
        "vasopressin",
        "dopamine",
        "angiotensin",
        "dobutamine",
        "milrinone",
        "isoproterenol",
    }
)

#: ECMO/MCS presence marks mechanical circulatory support (level 5, cv_flag).
MCS_ECMO_CV = "ecmo_mcs"

#: patient_assessments categories used for the neuro flag.
RASS_CATEGORY = "RASS"
GCS_TOTAL_CATEGORY = "gcs_total"


@dataclass(frozen=True)
class SpineStateConfig:
    """Reproducibility knobs for state derivation; serialized into the pack.

    Recorded verbatim in the parameter-pack manifest so a later fit or an
    auditor can reproduce the exact state labelling that the transitions,
    sojourns, and per-state physiology were fit against.
    """

    grid_step_hours: float = 1.0
    #: cap per-hospitalization horizon so a pathological LOS can't explode the
    #: interval grid; 28 days at hourly resolution.
    horizon_intervals: int = 24 * 28
    rass_deep_sedation_max: float = -3.0
    gcs_low_max: float = 8.0

    def as_manifest(self) -> dict[str, float | int | str]:
        return {
            "state_model": "organ_support_ladder_v1",
            "grid_step_hours": self.grid_step_hours,
            "horizon_intervals": self.horizon_intervals,
            "rass_deep_sedation_max": self.rass_deep_sedation_max,
            "gcs_low_max": self.gcs_low_max,
        }


def _interval_expr(dttm_col: str, admit_col: str, grid_step_hours: float) -> pl.Expr:
    """Interval index = floor((event_time - admission_time) / grid_step)."""
    delta_hours = (pl.col(dttm_col) - pl.col(admit_col)).dt.total_seconds() / 3600.0
    return (delta_hours / grid_step_hours).floor().cast(pl.Int64).alias("interval_idx")


def _admissions(hospitalization: pl.LazyFrame) -> pl.LazyFrame:
    return hospitalization.select(
        "hospitalization_id",
        pl.col("admission_dttm").alias("_admit"),
    )


def _support_events(
    tables: Mapping[str, pl.LazyFrame], admits: pl.LazyFrame, config: SpineStateConfig
) -> pl.LazyFrame:
    """Per-(hospitalization, interval) support level + flags from all sources.

    Each source table contributes rows carrying a candidate support level and
    the flags it implies; the caller aggregates (max level, any flag) per cell.
    """
    step = config.grid_step_hours
    parts: list[pl.LazyFrame] = []

    def assign(lf: pl.LazyFrame, dttm: str) -> pl.LazyFrame:
        return lf.join(admits, on="hospitalization_id", how="inner").with_columns(
            _interval_expr(dttm, "_admit", step)
        )

    if "respiratory_support" in tables:
        rs = assign(tables["respiratory_support"], "recorded_dttm").with_columns(
            pl.col("device_category")
            .replace_strict(DEVICE_SUPPORT_LEVEL, default=None)
            .cast(pl.Int64)
            .alias("_level"),
            pl.col("device_category").is_in(list(RESP_FLAG_DEVICES)).alias("_resp"),
        )
        parts.append(
            rs.select(
                "hospitalization_id",
                "interval_idx",
                pl.col("_level").alias("support_level"),
                pl.col("_resp").alias("resp_flag"),
                pl.lit(False).alias("cv_flag"),  # noqa: FBT003
                pl.lit(False).alias("renal_flag"),  # noqa: FBT003
                pl.lit(False).alias("neuro_flag"),  # noqa: FBT003
            )
        )

    if "medication_admin_continuous" in tables:
        mac = assign(tables["medication_admin_continuous"], "admin_dttm").filter(
            pl.col("med_category").is_in(list(VASOPRESSOR_INOTROPE))
        )
        parts.append(
            mac.select(
                "hospitalization_id",
                "interval_idx",
                pl.lit(int(Support.CIRC)).alias("support_level"),
                pl.lit(False).alias("resp_flag"),  # noqa: FBT003
                pl.lit(True).alias("cv_flag"),  # noqa: FBT003
                pl.lit(False).alias("renal_flag"),  # noqa: FBT003
                pl.lit(False).alias("neuro_flag"),  # noqa: FBT003
            )
        )

    if "crrt_therapy" in tables:
        crrt = assign(tables["crrt_therapy"], "recorded_dttm")
        parts.append(
            crrt.select(
                "hospitalization_id",
                "interval_idx",
                pl.lit(int(Support.RENAL_MCS)).alias("support_level"),
                pl.lit(False).alias("resp_flag"),  # noqa: FBT003
                pl.lit(False).alias("cv_flag"),  # noqa: FBT003
                pl.lit(True).alias("renal_flag"),  # noqa: FBT003
                pl.lit(False).alias("neuro_flag"),  # noqa: FBT003
            )
        )

    if "ecmo_mcs" in tables:
        ecmo = assign(tables["ecmo_mcs"], "recorded_dttm")
        parts.append(
            ecmo.select(
                "hospitalization_id",
                "interval_idx",
                pl.lit(int(Support.RENAL_MCS)).alias("support_level"),
                pl.lit(False).alias("resp_flag"),  # noqa: FBT003
                pl.lit(True).alias("cv_flag"),  # noqa: FBT003
                pl.lit(False).alias("renal_flag"),  # noqa: FBT003
                pl.lit(False).alias("neuro_flag"),  # noqa: FBT003
            )
        )

    if "patient_assessments" in tables:
        pa = assign(tables["patient_assessments"], "recorded_dttm")
        neuro = pa.filter(
            (
                (pl.col("assessment_category") == RASS_CATEGORY)
                & (pl.col("numerical_value") <= config.rass_deep_sedation_max)
            )
            | (
                (pl.col("assessment_category") == GCS_TOTAL_CATEGORY)
                & (pl.col("numerical_value") <= config.gcs_low_max)
            )
        )
        parts.append(
            neuro.select(
                "hospitalization_id",
                "interval_idx",
                pl.lit(None, dtype=pl.Int64).alias("support_level"),
                pl.lit(False).alias("resp_flag"),  # noqa: FBT003
                pl.lit(False).alias("cv_flag"),  # noqa: FBT003
                pl.lit(False).alias("renal_flag"),  # noqa: FBT003
                pl.lit(True).alias("neuro_flag"),  # noqa: FBT003
            )
        )

    if not parts:
        return pl.LazyFrame(
            schema={
                "hospitalization_id": pl.String,
                "interval_idx": pl.Int64,
                "support_level": pl.Int64,
                "resp_flag": pl.Boolean,
                "cv_flag": pl.Boolean,
                "renal_flag": pl.Boolean,
                "neuro_flag": pl.Boolean,
            }
        )
    return pl.concat(parts, how="vertical_relaxed")


def derive_state_timeline(
    tables: Mapping[str, pl.LazyFrame], config: SpineStateConfig | None = None
) -> pl.LazyFrame:
    """Return the observed per-interval spine state for each hospitalization.

    Output columns: ``hospitalization_id``, ``interval_idx`` (0-based, capped at
    ``horizon_intervals``), ``support_level`` (0-5, room-air baseline where a
    hospitalization has intervals but no support evidence), and the four boolean
    flags. Only intervals with at least one observation are emitted (a sparse,
    piecewise-constant timeline); the estimators forward-fill / run-length encode
    as each needs.
    """
    config = config or SpineStateConfig()
    if "hospitalization" not in tables:
        raise ValueError("derive_state_timeline requires a 'hospitalization' table")

    admits = _admissions(tables["hospitalization"])
    events = _support_events(tables, admits, config)

    return (
        events.filter(
            (pl.col("interval_idx") >= 0) & (pl.col("interval_idx") < config.horizon_intervals)
        )
        .group_by("hospitalization_id", "interval_idx")
        .agg(
            pl.col("support_level").max().fill_null(int(Support.NONE)).alias("support_level"),
            pl.col("resp_flag").any().alias("resp_flag"),
            pl.col("cv_flag").any().alias("cv_flag"),
            pl.col("renal_flag").any().alias("renal_flag"),
            pl.col("neuro_flag").any().alias("neuro_flag"),
        )
        .sort("hospitalization_id", "interval_idx")
    )


def outcome_by_hospitalization(hospitalization: pl.LazyFrame) -> pl.LazyFrame:
    """Map each hospitalization to a terminal outcome (``expired`` / ``alive``)."""
    return hospitalization.select(
        "hospitalization_id",
        pl.when(pl.col("discharge_category") == "Expired")
        .then(pl.lit("expired"))
        .otherwise(pl.lit("alive"))
        .alias("outcome"),
    )

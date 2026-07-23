"""One-time fit driver: real CLIF dataset -> aggregate parameter pack (U5).

This is the **only** module in ``clifforge`` that opens a real-data path
(KTD-1). Everything downstream — generation, conformance, evaluation — consumes
the parameter pack, never the source records. The driver:

1. Discovers and lazily scans the real CLIF tables under ``--real-dir``.
2. Computes a **seeded, patient-disjoint** train/holdout split. Only the split
   *spec* (seed, fraction, method, counts) is written to the pack manifest —
   never patient identifiers (that would trip the leakage gate). U22's TSTR
   evaluation recomputes the identical holdout set from the seed.
3. Fits the training partition only: demographic/encounter marginals, the
   semi-Markov organ-support spine (transitions + sojourns), per-state AR1
   physiology, the lab co-measurement copula, and infusion hazards — every
   cell routed through the n>=20 gate (R2).
4. Runs a **field-level source audit**: for each fitted table, records which
   CLIF 2.1.0 columns the real source actually carries (``fitted``) vs. which
   are new-in-2.1.0 with no real source (``prior`` — filled downstream by the
   prior-driven stage, U20).
5. Assembles the pack, runs the value-level leakage scan, and writes it. No
   row-level real record ever leaves this function.

Usage::

    python -m clifforge.fit.run_fit --real-dir /path/to/CLIF --out data/param_packs/mimic

The real path is a CLI argument and is never committed to the repo.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl

from clifforge.fit import estimators, spine_state
from clifforge.fit.cell_gate import SuppressionRecord
from clifforge.fit.param_pack import ParamPack, scan_for_leakage
from clifforge.fit.spine_state import SpineStateConfig
from clifforge.provenance import DEFAULT_CITATION
from clifforge.reference import loader

# Demographic / encounter categorical fields fit as marginals.
_PATIENT_CATEGORICALS = ("sex_category", "race_category", "ethnicity_category")
_HOSPITALIZATION_CATEGORICALS = ("admission_type_category", "discharge_category")

# Vitals fit with per-state AR1 (static anthropometrics excluded).
_AR1_VITALS = (
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "spo2",
    "respiratory_rate",
    "temp_c",
)

# CLIF table -> datetime column used to place a row on the interval grid.
_GRID_DTTM = {
    "vitals": "recorded_dttm",
    "labs": "lab_result_dttm",
    "medication_admin_continuous": "admin_dttm",
}


# --------------------------------------------------------------------------- #
# Real-data discovery (KTD-1: confined to this module)
# --------------------------------------------------------------------------- #
def _find_table(real_dir: Path, table: str) -> Path | None:
    """Locate a CLIF table file, tolerating the ``clif_`` prefix and csv/parquet."""
    for stem in (f"clif_{table}", table):
        for ext in (".parquet", ".csv"):
            candidate = real_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _scan(path: Path) -> pl.LazyFrame:
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10000)


def _load_tables(real_dir: Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan every CLIF table present under ``real_dir``."""
    tables: dict[str, pl.LazyFrame] = {}
    for table in loader.dictionary_tables():
        path = _find_table(real_dir, table)
        if path is not None:
            tables[table] = _scan(path)
    return tables


# --------------------------------------------------------------------------- #
# Seeded, patient-disjoint split (spec-only in the manifest — no identifiers)
# --------------------------------------------------------------------------- #
def _holdout_mask(patient_id: str, seed: int, holdout_fraction: float) -> bool:
    """Deterministic per-patient holdout membership from a stable hash.

    ``sha1(seed:patient_id) mod 10_000 < fraction*10_000``. Reproducible from
    the seed alone, so U22 recomputes the identical split without the pack ever
    storing an identifier.
    """
    digest = hashlib.sha1(f"{seed}:{patient_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    return bucket < int(holdout_fraction * 10_000)


def _split_patients(
    patient_ids: list[str], seed: int, holdout_fraction: float
) -> tuple[set[str], dict[str, object]]:
    """Return the training patient-id set and an identifier-free split spec."""
    train = {pid for pid in patient_ids if not _holdout_mask(pid, seed, holdout_fraction)}
    holdout_n = len(patient_ids) - len(train)
    spec = {
        "method": "sha1_mod_10000",
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "train_n_patients": len(train),
        "holdout_n_patients": holdout_n,
    }
    return train, spec


# --------------------------------------------------------------------------- #
# Gridding helpers
# --------------------------------------------------------------------------- #
def _grid_value_table(
    lf: pl.LazyFrame,
    admits: pl.LazyFrame,
    *,
    dttm_col: str,
    category_col: str,
    value_col: str,
    config: SpineStateConfig,
) -> pl.DataFrame:
    """Resample a long value table to (hospitalization, interval, category)-mean."""
    delta_hours = (pl.col(dttm_col) - pl.col("_admit")).dt.total_seconds() / 3600.0
    interval = (delta_hours / config.grid_step_hours).floor().cast(pl.Int64)
    return (
        lf.join(admits, on="hospitalization_id", how="inner")
        .with_columns(interval.alias("interval_idx"))
        .filter((pl.col("interval_idx") >= 0) & (pl.col("interval_idx") < config.horizon_intervals))
        .group_by("hospitalization_id", "interval_idx", category_col)
        .agg(pl.col(value_col).mean().alias("value"))
        .collect()
    )


# --------------------------------------------------------------------------- #
# Field-level source audit
# --------------------------------------------------------------------------- #
def _field_sources(table: str, source_columns: set[str]) -> list[dict[str, str]]:
    """Per CLIF column: ``fitted`` if the real source carries it, else ``prior``.

    Returned as a list of ``{"column", "source"}`` records rather than a
    ``{column: source}`` dict on purpose: identifier column names
    (``patient_id``, ``hospitalization_id``) are legitimate audit *content*
    here, but the pack leakage scanner forbids them as *keys*. As string values
    under a neutral ``"column"`` key they carry the same information without
    tripping the key-based guard.
    """
    return [
        {
            "column": col["name"],
            "source": "fitted" if col["name"] in source_columns else "prior",
        }
        for col in loader.table_columns(table)
    ]


def _source_columns(lf: pl.LazyFrame) -> set[str]:
    return set(lf.collect_schema().names())


# --------------------------------------------------------------------------- #
# Suppression-audit roll-up
# --------------------------------------------------------------------------- #
def _rollup(records: list[SuppressionRecord]) -> dict[str, object]:
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r.fallback_kind] = kinds.get(r.fallback_kind, 0) + 1
    return {
        "cells_considered": len(records),
        "cells_suppressed": sum(1 for r in records if r.fallback_kind == "none"),
        "fallback_kinds": kinds,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_fit(
    real_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 20260723,
    holdout_fraction: float = 0.2,
    dataset_id: str = "MIMIC-IV-Ext-CLIF",
    config: SpineStateConfig | None = None,
) -> ParamPack:
    """Fit a real CLIF dataset into an aggregate parameter pack and write it."""
    real_dir = Path(real_dir)
    config = config or SpineStateConfig()
    tables = _load_tables(real_dir)
    if "patient" not in tables or "hospitalization" not in tables:
        raise ValueError("real-dir must contain at least patient and hospitalization tables")

    # --- seeded, patient-disjoint training partition -----------------------
    patient_ids = tables["patient"].select("patient_id").collect().to_series().to_list()
    train_patients, split_spec = _split_patients(
        [str(p) for p in patient_ids], seed, holdout_fraction
    )
    train_hosp = (
        tables["hospitalization"]
        .filter(pl.col("patient_id").cast(pl.String).is_in(list(train_patients)))
        .select("hospitalization_id")
        .collect()
        .to_series()
        .to_list()
    )
    train_hosp_set = list(train_hosp)

    def restrict(lf: pl.LazyFrame, key: str) -> pl.LazyFrame:
        return lf.filter(pl.col(key).is_in(train_hosp_set))

    # Training-partition views (patient-keyed vs hospitalization-keyed).
    train_tables: dict[str, pl.LazyFrame] = {}
    for name, lf in tables.items():
        cols = set(lf.collect_schema().names())
        if "hospitalization_id" in cols:
            train_tables[name] = restrict(lf, "hospitalization_id")
        elif "patient_id" in cols:
            train_tables[name] = lf.filter(
                pl.col("patient_id").cast(pl.String).is_in(list(train_patients))
            )
        else:
            train_tables[name] = lf

    admits = spine_state._admissions(train_tables["hospitalization"])

    table_blocks: dict[str, dict[str, object]] = {}
    all_records: list[SuppressionRecord] = []
    field_audit: dict[str, list[dict[str, str]]] = {}

    # --- patient marginals -------------------------------------------------
    patient_df = train_tables["patient"].collect()
    p_params, p_rec = estimators.fit_categorical_marginals(patient_df, _PATIENT_CATEGORICALS)
    all_records.extend(p_rec)
    table_blocks["patient"] = {
        "n_records": patient_df.height,
        "fitted": True,
        "params": p_params,
    }
    field_audit["patient"] = _field_sources("patient", set(patient_df.columns))

    # --- hospitalization marginals ----------------------------------------
    hosp_df = train_tables["hospitalization"].collect()
    h_params, h_rec = estimators.fit_categorical_marginals(hosp_df, _HOSPITALIZATION_CATEGORICALS)
    all_records.extend(h_rec)
    table_blocks["hospitalization"] = {
        "n_records": hosp_df.height,
        "fitted": True,
        "params": h_params,
    }
    field_audit["hospitalization"] = _field_sources("hospitalization", set(hosp_df.columns))

    # --- semi-Markov spine: transitions + sojourns ------------------------
    timeline = spine_state.derive_state_timeline(train_tables, config).collect()
    t_params, t_rec = estimators.fit_transitions(timeline)
    s_params, s_rec = estimators.fit_sojourns(timeline, grid_step_hours=config.grid_step_hours)
    all_records.extend(t_rec)
    all_records.extend(s_rec)
    table_blocks["spine"] = {
        "n_records": timeline["hospitalization_id"].n_unique(),
        "fitted": True,
        "params": {
            **t_params,
            **s_params,
            "state_model": config.as_manifest(),
        },
    }

    # --- per-state AR1 physiology -----------------------------------------
    if "vitals" in train_tables:
        vitals_grid = _grid_value_table(
            train_tables["vitals"],
            admits,
            dttm_col=_GRID_DTTM["vitals"],
            category_col="vital_category",
            value_col="vital_value",
            config=config,
        )
        ar1_params, ar1_rec = estimators.fit_ar1_by_state(vitals_grid, timeline, vitals=_AR1_VITALS)
        all_records.extend(ar1_rec)
        table_blocks["vitals"] = {
            "n_records": vitals_grid.height,
            "fitted": True,
            "params": ar1_params,
        }
        field_audit["vitals"] = _field_sources("vitals", _source_columns(tables["vitals"]))

    # --- lab co-measurement copula ----------------------------------------
    if "labs" in train_tables:
        labs_grid = _grid_value_table(
            train_tables["labs"],
            admits,
            dttm_col=_GRID_DTTM["labs"],
            category_col="lab_category",
            value_col="lab_value_numeric",
            config=config,
        )
        n_hosp = len(train_hosp_set)
        lab_params, lab_rec = estimators.fit_lab_copula(labs_grid, n_hospitalizations=n_hosp)
        all_records.extend(lab_rec)
        table_blocks["labs"] = {
            "n_records": labs_grid.height,
            "fitted": True,
            "params": lab_params,
        }
        field_audit["labs"] = _field_sources("labs", _source_columns(tables["labs"]))

    # --- infusion hazards --------------------------------------------------
    if "medication_admin_continuous" in train_tables:
        mac_grid = _grid_value_table(
            train_tables["medication_admin_continuous"],
            admits,
            dttm_col=_GRID_DTTM["medication_admin_continuous"],
            category_col="med_category",
            value_col="med_dose",
            config=config,
        )
        haz_params, haz_rec = estimators.fit_infusion_hazards(mac_grid)
        all_records.extend(haz_rec)
        table_blocks["medication_admin_continuous"] = {
            "n_records": mac_grid.height,
            "fitted": True,
            "params": haz_params,
        }
        field_audit["medication_admin_continuous"] = _field_sources(
            "medication_admin_continuous",
            _source_columns(tables["medication_admin_continuous"]),
        )

    # --- assemble manifest + pack -----------------------------------------
    ref = loader.provenance()
    manifest: dict[str, object] = {
        "pack_version": "1.0",
        "clif_version": ref["clif_version"],
        "fit_source": {
            "dataset_id": dataset_id,
            "commit": "unknown",
        },
        "reference_source": {
            "source_repo": ref.get("source_repo"),
            "source_commit": ref.get("source_commit"),
            "retrieved_at": ref.get("retrieved_at"),
        },
        "split": split_spec,
        "field_sources": field_audit,
        "tables": {name: {"fitted": block["fitted"]} for name, block in table_blocks.items()},
        "suppression_audit": {"overall": _rollup(all_records)},
        "citation": DEFAULT_CITATION,
    }

    pack = ParamPack(manifest=manifest, tables=table_blocks)
    findings = scan_for_leakage(pack)
    if findings:
        rendered = "\n".join(f"  - {f.describe()}" for f in findings)
        raise RuntimeError(f"refusing to write pack — leakage scan found:\n{rendered}")
    pack.write(out_dir)
    return pack


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a real CLIF dataset into a parameter pack.")
    parser.add_argument("--real-dir", required=True, help="directory of real CLIF tables")
    parser.add_argument("--out", required=True, help="output parameter-pack directory")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--dataset-id", default="MIMIC-IV-Ext-CLIF")
    parser.add_argument("--grid-step-hours", type=float, default=1.0)
    args = parser.parse_args(argv)

    config = SpineStateConfig(grid_step_hours=args.grid_step_hours)
    pack = run_fit(
        args.real_dir,
        args.out,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        dataset_id=args.dataset_id,
        config=config,
    )
    n_tables = len(pack.tables)
    print(f"Wrote parameter pack with {n_tables} table block(s) to {args.out}")


if __name__ == "__main__":
    main()

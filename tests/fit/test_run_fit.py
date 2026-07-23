"""End-to-end fit-driver test on a fabricated CLIF directory (U5d).

Writes a small, entirely synthetic CLIF corpus to a temp dir, runs the fit
driver against it, and asserts the resulting pack loads, carries an
identifier-free split spec + field-source audit, and passes the leakage scan.
No real data path is used — the "real dir" here is a fabricated fixture (KTD-1
is about the *module boundary*, satisfied because run_fit is the only importer
of a data path; the path it reads in this test is synthetic).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from clifforge.fit import run_fit
from clifforge.fit.param_pack import ParamPack, scan_for_leakage
from clifforge.fit.spine_state import SpineStateConfig

_T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _write_corpus(real_dir: Path, n_hosp: int = 80) -> None:
    real_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    patients = pl.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(n_hosp)],
            "sex_category": [["Male", "Female"][i % 2] for i in range(n_hosp)],
            "race_category": [["White", "Black or African American"][i % 2] for i in range(n_hosp)],
            "ethnicity_category": ["Non-Hispanic"] * n_hosp,
        }
    )
    hospitalizations = pl.DataFrame(
        {
            "hospitalization_id": [f"H{i}" for i in range(n_hosp)],
            "patient_id": [f"P{i}" for i in range(n_hosp)],
            "admission_dttm": [_T0] * n_hosp,
            "discharge_category": [["Home", "Expired"][i % 5 == 0] for i in range(n_hosp)],
            "admission_type_category": ["Direct Admit"] * n_hosp,
        }
    )

    resp_rows, vit_rows, lab_rows, mac_rows = [], [], [], []
    for i in range(n_hosp):
        hid = f"H{i}"
        for interval, device in enumerate(["Room Air", "High Flow NC", "IMV", "Room Air"]):
            resp_rows.append(
                {
                    "hospitalization_id": hid,
                    "recorded_dttm": _T0 + timedelta(hours=interval),
                    "device_category": device,
                }
            )
        x = 82.0
        for interval in range(6):
            x = 82.0 + 0.6 * (x - 82.0) + rng.normal(0, 3.0)
            vit_rows.append(
                {
                    "hospitalization_id": hid,
                    "recorded_dttm": _T0 + timedelta(hours=interval),
                    "vital_category": "heart_rate",
                    "vital_value": float(x),
                }
            )
            base = rng.normal(0, 1)
            for lab, level in (("creatinine", 1.0 + 0.5 * base), ("lactate", 0.8 + 0.5 * base)):
                lab_rows.append(
                    {
                        "hospitalization_id": hid,
                        "lab_result_dttm": _T0 + timedelta(hours=interval),
                        "lab_category": lab,
                        "lab_value_numeric": float(np.expm1(level + rng.normal(0, 0.2))),
                    }
                )
        for interval in (1, 2):
            mac_rows.append(
                {
                    "hospitalization_id": hid,
                    "admin_dttm": _T0 + timedelta(hours=interval),
                    "med_category": "norepinephrine",
                    "med_dose": float(rng.uniform(0.02, 0.4)),
                }
            )

    patients.write_parquet(real_dir / "clif_patient.parquet")
    hospitalizations.write_parquet(real_dir / "clif_hospitalization.parquet")
    pl.DataFrame(resp_rows).write_parquet(real_dir / "clif_respiratory_support.parquet")
    pl.DataFrame(vit_rows).write_parquet(real_dir / "clif_vitals.parquet")
    pl.DataFrame(lab_rows).write_parquet(real_dir / "clif_labs.parquet")
    pl.DataFrame(mac_rows).write_parquet(real_dir / "clif_medication_admin_continuous.parquet")


def test_run_fit_writes_loadable_clean_pack(tmp_path: Path) -> None:
    real_dir = tmp_path / "CLIF"
    out_dir = tmp_path / "pack"
    _write_corpus(real_dir)

    run_fit.run_fit(real_dir, out_dir, seed=1234, holdout_fraction=0.2)

    pack = ParamPack.load(out_dir)
    assert pack.manifest["pack_version"] == "1.0"
    assert pack.manifest["clif_version"] == "2.1.0"
    # Pack loads clean through the leakage scanner.
    assert scan_for_leakage(pack) == []


def test_split_spec_is_identifier_free(tmp_path: Path) -> None:
    real_dir = tmp_path / "CLIF"
    out_dir = tmp_path / "pack"
    _write_corpus(real_dir)
    run_fit.run_fit(real_dir, out_dir, seed=99, holdout_fraction=0.25)

    pack = ParamPack.load(out_dir)
    split = pack.manifest["split"]
    assert split["seed"] == 99
    assert split["method"] == "sha1_mod_10000"
    assert split["train_n_patients"] + split["holdout_n_patients"] == 80
    assert split["holdout_n_patients"] > 0
    # The spec carries only counts/seed — never a patient identifier.
    assert "patient_id" not in split
    assert "holdout_ids" not in split


def test_field_source_audit_marks_present_columns_fitted(tmp_path: Path) -> None:
    real_dir = tmp_path / "CLIF"
    out_dir = tmp_path / "pack"
    _write_corpus(real_dir)
    run_fit.run_fit(real_dir, out_dir)

    pack = ParamPack.load(out_dir)
    patient_fields = {
        rec["column"]: rec["source"] for rec in pack.manifest["field_sources"]["patient"]
    }
    assert patient_fields["sex_category"] == "fitted"


def test_split_is_reproducible(tmp_path: Path) -> None:
    # Same seed -> identical holdout membership (U22 recomputes without ids).
    ids = [f"P{i}" for i in range(200)]
    train_a, spec_a = run_fit._split_patients(ids, seed=5, holdout_fraction=0.3)
    train_b, spec_b = run_fit._split_patients(ids, seed=5, holdout_fraction=0.3)
    assert train_a == train_b
    assert spec_a == spec_b
    # Different seed -> different split.
    train_c, _ = run_fit._split_patients(ids, seed=6, holdout_fraction=0.3)
    assert train_a != train_c


def test_spine_block_present(tmp_path: Path) -> None:
    real_dir = tmp_path / "CLIF"
    out_dir = tmp_path / "pack"
    _write_corpus(real_dir)
    run_fit.run_fit(real_dir, out_dir, config=SpineStateConfig(grid_step_hours=1.0))

    pack = ParamPack.load(out_dir)
    assert "spine" in pack.tables
    matrix = pack.tables["spine"]["params"]["support_level_transition_matrix"]
    # Every emitted transition row is stochastic with a zero diagonal.
    for from_level, row in matrix.items():
        assert from_level not in row
        assert abs(sum(row.values()) - 1.0) < 1e-9

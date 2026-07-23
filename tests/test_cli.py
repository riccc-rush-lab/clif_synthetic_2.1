"""Tests for the CLI + U21 orchestrator (R22, R23, R25, AE6).

The scaffold parser tests (U1) plus the end-to-end pipeline: a fully
self-contained synthetic parameter pack (no real data) drives spine -> 19 tables
-> gate -> parquet, so CI exercises seeded determinism / byte-identical output
(AE6), CLIF ``--out`` naming (R23), nonzero exit on any validation failure (R25),
and the ``fit`` subcommand wiring.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from clifforge.cli import build_parser, main
from clifforge.conformance.gate import ConformanceError
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import GeneratedDataset, generate_dataset, write_dataset


# --- U1 scaffold parser tests ------------------------------------------------ #
def test_parser_program_name() -> None:
    assert build_parser().prog == "clif-forge"


def test_no_command_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "generate" in out and "fit" in out


def test_generate_requires_flags() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate"])  # missing --n-patients / --out


def test_generate_parses_flags() -> None:
    args = build_parser().parse_args(
        ["generate", "--n-patients", "100", "--seed", "42", "--out", "./out"]
    )
    assert args.command == "generate"
    assert args.n_patients == 100
    assert args.seed == 42
    assert args.pack == "data/param_packs/mimic"  # default pack location


def test_rng_fixture_is_seed_reproducible(rng: np.random.Generator, seed: int) -> None:
    first = rng.integers(0, 1_000_000, size=5)
    fresh = np.random.Generator(np.random.PCG64(seed))
    assert (first == fresh.integers(0, 1_000_000, size=5)).all()


# --- synthetic pack fixture -------------------------------------------------- #
def _synthetic_pack() -> ParamPack:
    """A minimal but complete pack: every fitted block the generators read."""

    def six(mean: float) -> dict[str, dict[str, float]]:
        return {str(s): {"mean": mean - 3 * s, "phi": 0.4, "sigma": 3.0} for s in range(6)}

    return ParamPack(
        manifest={
            "clif_version": "2.1.0",
            "pack_version": "1.0",
            "fit_source": {"dataset_id": "synthetic-test", "commit": "none"},
            "suppression_audit": [],
        },
        tables={
            "patient": {
                "params": {
                    "race_category_marginal": {"White": 0.6, "Unknown": 0.4},
                    "ethnicity_category_marginal": {"Non-Hispanic": 0.7, "Unknown": 0.3},
                    "sex_category_marginal": {"Female": 0.5, "Male": 0.5},
                }
            },
            "hospitalization": {
                "params": {
                    "admission_type_category_marginal": {"ed": 0.8, "direct": 0.2},
                    "discharge_category_marginal": {"Home": 0.7, "Expired": 0.3},
                }
            },
            "vitals": {
                "params": {
                    f"{v}_ar1_by_state": six(base)
                    for v, base in {
                        "heart_rate": 90,
                        "sbp": 130,
                        "dbp": 75,
                        "map": 90,
                        "respiratory_rate": 20,
                        "spo2": 100,
                        "temp_c": 38,
                    }.items()
                }
            },
            "labs": {
                "params": {
                    "lab_order": ["creatinine", "bun", "sodium"],
                    "lab_correlation": [[1.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 1.0]],
                    "lab_marginals": {
                        "creatinine": {"log_mean": 0.79, "log_sd": 0.4},
                        "bun": {"log_mean": 2.8, "log_sd": 0.5},
                        "sodium": {"log_mean": 4.95, "log_sd": 0.03},
                    },
                    "lab_presence": {"creatinine": 0.8, "bun": 0.6, "sodium": 0.9},
                }
            },
            "medication_admin_continuous": {
                "params": {
                    "infusion_hazards": {
                        "norepinephrine": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                        "propofol": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                    }
                }
            },
            "spine": {
                "params": {
                    "state_model": {"grid_step_hours": 1.0, "horizon_intervals": 72},
                    "support_level_states": [0, 1, 2, 3, 4, 5],
                    "support_level_start_dist": {
                        "0": 0.3,
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.15,
                        "4": 0.1,
                        "5": 0.05,
                    },
                    "support_level_transition_matrix": {
                        "0": {"1": 0.2, "3": 0.3, "discharge": 0.5},
                        "1": {"2": 0.3, "0": 0.2, "discharge": 0.5},
                        "2": {"3": 0.4, "1": 0.2, "discharge": 0.4},
                        "3": {"4": 0.3, "2": 0.3, "discharge": 0.4},
                        "4": {"5": 0.3, "3": 0.3, "discharge": 0.4},
                        "5": {"4": 0.4, "discharge": 0.6},
                    },
                    "support_level_sojourn": {
                        str(s): {"family": "empirical_mean", "params": [2.0], "mean_hours": 2.0}
                        for s in range(6)
                    },
                    "flag_prevalence_by_level": {
                        "0": {
                            "resp_flag": 0.0,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.0,
                        },
                        "1": {
                            "resp_flag": 0.1,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.1,
                        },
                        "2": {
                            "resp_flag": 0.4,
                            "cv_flag": 0.1,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.2,
                        },
                        "3": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.2,
                            "renal_flag": 0.1,
                            "neuro_flag": 0.3,
                        },
                        "4": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.8,
                            "renal_flag": 0.3,
                            "neuro_flag": 0.3,
                        },
                        "5": {
                            "resp_flag": 0.8,
                            "cv_flag": 0.9,
                            "renal_flag": 0.8,
                            "neuro_flag": 0.4,
                        },
                    },
                    "outcome_marginal": {"alive": 0.8, "expired": 0.2},
                    "expired_rate_by_peak_level": {
                        str(s): {"expired_rate": 0.05 + 0.09 * s} for s in range(6)
                    },
                }
            },
        },
    )


@pytest.fixture
def pack() -> ParamPack:
    return _synthetic_pack()


# --- orchestrator ------------------------------------------------------------ #
def test_generate_dataset_produces_all_tables(pack: ParamPack) -> None:
    ds = generate_dataset(pack, n_patients=12, seed=1)
    assert isinstance(ds, GeneratedDataset)
    assert len(ds.tables) == 19
    assert "patient" in ds.tables and "provider" in ds.tables
    assert ds.truth.height > 0


def test_generate_dataset_is_deterministic(pack: ParamPack) -> None:
    a = generate_dataset(pack, n_patients=10, seed=7)
    b = generate_dataset(pack, n_patients=10, seed=7)
    for name in a.tables:
        assert a.tables[name].equals(b.tables[name]), f"{name} differs across identical seeds"
    assert a.truth.equals(b.truth)


def test_ae4_death_propagates_to_patient(pack: ParamPack) -> None:
    ds = generate_dataset(pack, n_patients=60, seed=3)
    deaths = ds.tables["patient"].filter(pl.col("death_dttm").is_not_null()).height
    expired = ds.tables["hospitalization"].filter(pl.col("discharge_category") == "Expired").height
    assert deaths == expired  # AE4: every expired encounter marks its patient row


def test_zero_orphans_across_all_tables(pack: ParamPack) -> None:
    ds = generate_dataset(pack, n_patients=40, seed=5)
    hosp_ids = set(ds.tables["hospitalization"]["hospitalization_id"].to_list())
    patient_ids = set(ds.tables["patient"]["patient_id"].to_list())
    assert set(ds.tables["hospitalization"]["patient_id"].to_list()) <= patient_ids
    for name, frame in ds.tables.items():
        if name != "hospitalization" and "hospitalization_id" in frame.columns:
            assert set(frame["hospitalization_id"].to_list()) <= hosp_ids, f"orphan in {name}"


def test_n_patients_must_be_positive(pack: ParamPack) -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_dataset(pack, n_patients=0, seed=1)


def test_conformance_failure_is_detected(pack: ParamPack) -> None:
    from clifforge.conformance import gate

    ds = generate_dataset(pack, n_patients=5, seed=1)
    corrupt = ds.tables["patient"].with_columns(pl.lit("NotAnMcideRace").alias("race_category"))
    with pytest.raises(ConformanceError):
        gate.validate(corrupt, "patient", run_secondary=False)


def test_write_dataset_can_skip_truth(pack: ParamPack, tmp_path) -> None:
    ds = generate_dataset(pack, n_patients=4, seed=1)
    written = write_dataset(ds, tmp_path, write_truth=False)
    assert not (tmp_path / "clif_truth.parquet").exists()
    assert len(written) == 19
    assert all(p.suffix == ".parquet" for p in written)


# --- CLI end-to-end ---------------------------------------------------------- #
def test_cli_generate_writes_clif_layout(pack: ParamPack, tmp_path) -> None:
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out = tmp_path / "out"
    code = main(
        [
            "generate",
            "--n-patients",
            "8",
            "--seed",
            "42",
            "--out",
            str(out),
            "--pack",
            str(pack_dir),
        ]
    )
    assert code == 0
    assert (out / "clif_patient.parquet").exists()
    assert (out / "clif_hospitalization.parquet").exists()
    assert (out / "clif_truth.parquet").exists()
    written = {p.name for p in out.glob("clif_*.parquet")}
    assert "clif_vitals.parquet" in written and "clif_provider.parquet" in written


def test_cli_ae6_two_runs_byte_identical(pack: ParamPack, tmp_path) -> None:
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    args = ["generate", "--n-patients", "10", "--seed", "99", "--pack", str(pack_dir), "--out"]
    assert main([*args, str(out_a)]) == 0
    assert main([*args, str(out_b)]) == 0
    for pa in sorted(out_a.glob("clif_*.parquet")):
        assert (out_b / pa.name).read_bytes() == pa.read_bytes(), f"{pa.name} not byte-identical"


def test_cli_generate_nonzero_on_conformance_failure(monkeypatch, tmp_path, pack) -> None:
    # Corrupt one table's assembler so the real conformance gate rejects it; the
    # CLI must surface that as a nonzero exit (R25) rather than writing bad data.
    import clifforge.generate.orchestrator as orch

    real_patient_frame = orch.patient_frame

    def corrupt_patient_frame(records):
        return real_patient_frame(records).with_columns(
            pl.lit("NotAnMcideRace").alias("race_category")
        )

    monkeypatch.setattr(orch, "patient_frame", corrupt_patient_frame)
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    code = main(
        ["generate", "--n-patients", "5", "--out", str(tmp_path / "o"), "--pack", str(pack_dir)]
    )
    assert code == 1  # R25: any validation failure -> nonzero exit
    assert not (tmp_path / "o" / "clif_patient.parquet").exists()  # nothing written on failure


def test_cli_generate_nonzero_on_missing_pack(tmp_path) -> None:
    code = main(
        [
            "generate",
            "--n-patients",
            "3",
            "--out",
            str(tmp_path / "o"),
            "--pack",
            str(tmp_path / "does_not_exist"),
        ]
    )
    assert code == 1


def test_cli_fit_invokes_run_fit(monkeypatch, tmp_path) -> None:
    called = {}

    def _fake_run_fit(real_dir, out_dir, **_k):
        called["real_dir"] = str(real_dir)
        called["out_dir"] = str(out_dir)

    monkeypatch.setattr("clifforge.fit.run_fit.run_fit", _fake_run_fit)
    code = main(["fit", "--real-dir", str(tmp_path / "real"), "--out", str(tmp_path / "pack")])
    assert code == 0
    assert called["real_dir"].endswith("real") and called["out_dir"].endswith("pack")

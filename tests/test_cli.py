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


def test_seed_actually_changes_output(pack: ParamPack) -> None:
    # Guards against the seed being ignored (e.g. a hardcoded SeedSequence(42)) —
    # the same-seed determinism tests would still pass in that regression.
    a = generate_dataset(pack, n_patients=10, seed=1)
    b = generate_dataset(pack, n_patients=10, seed=2)
    assert not a.truth.equals(b.truth)
    assert not a.tables["hospitalization"].equals(b.tables["hospitalization"])


def test_first_encounters_stable_across_n_patients(pack: ParamPack) -> None:
    # SeedSequence.spawn(n) assigns child i a stable key regardless of n, so the
    # first k encounters must be identical whether we ask for k or more — the
    # property that would make generation safely resumable/extendable.
    small = generate_dataset(pack, n_patients=5, seed=9)
    large = generate_dataset(pack, n_patients=20, seed=9)
    first5 = {f"P{i}" for i in range(5)}
    small_p = small.tables["patient"].sort("patient_id")
    large_p = large.tables["patient"].filter(pl.col("patient_id").is_in(first5)).sort("patient_id")
    assert small_p.equals(large_p)


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
    # code_status is the one table keyed on patient_id, not hospitalization_id.
    code_status = ds.tables["code_status"]
    assert "patient_id" in code_status.columns
    assert set(code_status["patient_id"].to_list()) <= patient_ids, "orphan in code_status"


def test_high_acuity_tables_are_actually_populated(pack: ParamPack) -> None:
    # ecmo/crrt/hemodynamics fire only at rare high-acuity states; assert they
    # are non-empty at n large enough to reach those states, so their row-build
    # logic is exercised rather than passing vacuously on empty frames.
    ds = generate_dataset(pack, n_patients=200, seed=11)
    for name in ("ecmo_mcs", "crrt_therapy", "invasive_hemodynamics", "transfusion"):
        assert ds.tables[name].height > 0, f"{name} never populated — coupling logic untested"


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


def _null_id_wrapper(frame_fn, id_col):
    """Wrap a frame assembler so it nulls a required id column (a real gate failure)."""

    def corrupt(records):
        return frame_fn(records).with_columns(pl.lit(None, dtype=pl.String).alias(id_col))

    return corrupt


@pytest.mark.parametrize(
    ("table", "id_col"),
    [
        ("patient", "patient_id"),  # assembled directly by the orchestrator
        ("provider", "provider_id"),  # assembled via the table registry (last entry)
    ],
)
def test_cli_generate_nonzero_on_conformance_failure(
    monkeypatch, tmp_path, pack, table, id_col
) -> None:
    # Corrupt a table's assembler (null a required id column) so the real gate
    # rejects it; the CLI must surface that as a nonzero exit (R25), not bad data.
    # Parametrized over an early and a late table so a dropped gate call for a
    # downstream table can't hide behind patient's own failure.
    import clifforge.generate.orchestrator as orch

    if table == "patient":
        monkeypatch.setattr(orch, "patient_frame", _null_id_wrapper(orch.patient_frame, id_col))
    else:
        # _TABLE_REGISTRY captures frame functions at import, so patching the
        # module attribute would not reach it — patch the registry entry itself.
        patched = tuple(
            (
                name,
                sample_fn,
                _null_id_wrapper(frame_fn, id_col) if name == table else frame_fn,
                key,
            )
            for name, sample_fn, frame_fn, key in orch._TABLE_REGISTRY
        )
        monkeypatch.setattr(orch, "_TABLE_REGISTRY", patched)

    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out = tmp_path / "o"
    code = main(["generate", "--n-patients", "5", "--out", str(out), "--pack", str(pack_dir)])
    assert code == 1  # R25: any validation failure -> nonzero exit
    assert not out.exists()  # gate precedes write: no partial output for any table


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


def test_cli_fit_nonzero_on_run_fit_error(monkeypatch, tmp_path) -> None:
    def _boom(*_a, **_k):
        raise FileNotFoundError("real-dir not found")

    monkeypatch.setattr("clifforge.fit.run_fit.run_fit", _boom)
    code = main(["fit", "--real-dir", str(tmp_path / "real"), "--out", str(tmp_path / "pack")])
    assert code == 1  # fit's error path mirrors generate's R25-style clean exit


def test_cli_generate_clean_exit_when_out_is_a_file(pack: ParamPack, tmp_path) -> None:
    # --out pointing at an existing regular file makes mkdir raise FileExistsError;
    # the CLI must report it cleanly (nonzero), not crash with a traceback.
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out_file = tmp_path / "out_is_a_file"
    out_file.write_text("i am not a directory")
    code = main(["generate", "--n-patients", "3", "--out", str(out_file), "--pack", str(pack_dir)])
    assert code == 1

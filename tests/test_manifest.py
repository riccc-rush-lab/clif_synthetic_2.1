"""Tests for the dataset generation manifest (U5)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clifforge.manifest import datasets_are_distinct, read_manifest, write_manifest


def _dataset(tmp: Path, name: str, values: list[int]) -> Path:
    d = tmp / name
    d.mkdir()
    pl.DataFrame({"hospitalization_id": ["H0", "H1"]}).write_parquet(
        d / "clif_hospitalization.parquet"
    )
    pl.DataFrame({"vital_value": values}).write_parquet(d / "clif_vitals.parquet")
    return d


def test_write_manifest_records_version_spec_seed_and_hashes(tmp_path: Path) -> None:
    d = _dataset(tmp_path, "ds", [1, 2, 3])
    m = write_manifest(d, spec={"name": "v", "rates": {"imv": 0.5}}, seed=7)
    assert (d / "manifest.json").exists()
    assert m["generator_version"] and m["seed"] == 7 and m["spec"]["name"] == "v"
    assert m["tables"]["clif_vitals"]["rows"] == 3
    assert len(m["tables"]["clif_vitals"]["sha256"]) == 64
    assert read_manifest(d) == m


def test_identical_content_is_not_distinct(tmp_path: Path) -> None:
    a = write_manifest(_dataset(tmp_path, "a", [1, 2, 3]), spec="master", seed=1)
    b = write_manifest(_dataset(tmp_path, "b", [1, 2, 3]), spec="master", seed=1)
    assert not datasets_are_distinct(a, b)  # same content -> same hashes
    assert not datasets_are_distinct(a, a)


def test_different_content_is_distinct(tmp_path: Path) -> None:
    a = write_manifest(_dataset(tmp_path, "a", [1, 2, 3]), spec={"name": "a"}, seed=1)
    b = write_manifest(_dataset(tmp_path, "b", [9, 9, 9]), spec={"name": "b"}, seed=2)
    assert datasets_are_distinct(a, b)

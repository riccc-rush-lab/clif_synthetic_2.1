"""Tests for ``clifforge.fit.param_pack`` (U4).

Covers: round-trip write/read, pack_version compatibility gating, the
row-level-key schema check, and the value-level leakage scanner that
enforces R1/R2 beyond a key-only check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from clifforge.fit.param_pack import (
    PACK_FORMAT_VERSION,
    IncompatiblePackVersionError,
    LeakageError,
    ParamPack,
    read_pack,
    scan_for_leakage,
    write_pack,
)


def test_round_trip_write_read_preserves_values(
    tmp_path: Path,
    sample_manifest: dict[str, Any],
    sample_table_blocks: dict[str, dict[str, Any]],
) -> None:
    pack_dir = tmp_path / "pack"
    write_pack(pack_dir, sample_manifest, sample_table_blocks)

    loaded = read_pack(pack_dir)

    assert loaded.manifest == sample_manifest
    assert loaded.tables == sample_table_blocks


def test_write_creates_expected_directory_layout(
    tmp_path: Path,
    sample_manifest: dict[str, Any],
    sample_table_blocks: dict[str, dict[str, Any]],
) -> None:
    pack_dir = tmp_path / "pack"
    write_pack(pack_dir, sample_manifest, sample_table_blocks)

    assert (pack_dir / "manifest.json").is_file()
    for table in sample_table_blocks:
        assert (pack_dir / "tables" / f"{table}.json").is_file()


def test_incompatible_pack_version_raises_on_load(
    tmp_path: Path,
    sample_manifest: dict[str, Any],
    sample_table_blocks: dict[str, dict[str, Any]],
) -> None:
    bad_manifest = dict(sample_manifest)
    major = int(PACK_FORMAT_VERSION.split(".")[0])
    bad_manifest["pack_version"] = f"{major + 1}.0"

    pack_dir = tmp_path / "pack"
    write_pack(pack_dir, bad_manifest, sample_table_blocks)

    with pytest.raises(IncompatiblePackVersionError):
        read_pack(pack_dir)


def test_missing_manifest_key_raises_on_write(
    tmp_path: Path, sample_table_blocks: dict[str, dict[str, Any]]
) -> None:
    incomplete_manifest = {"pack_version": "1.0", "clif_version": "2.1.0"}
    with pytest.raises(ValueError, match="missing required key"):
        write_pack(tmp_path / "pack", incomplete_manifest, sample_table_blocks)


def test_manifest_lists_no_row_level_fields(
    tmp_path: Path,
    sample_manifest: dict[str, Any],
    sample_table_blocks: dict[str, dict[str, Any]],
) -> None:
    """Grep-style assertion: no per-record identifier key appears anywhere
    in the written JSON files (R1)."""
    pack_dir = tmp_path / "pack"
    write_pack(pack_dir, sample_manifest, sample_table_blocks)

    forbidden_substrings = ["patient_id", "mrn", "ssn", "subject_id", "row_id"]
    manifest_text = (pack_dir / "manifest.json").read_text(encoding="utf-8")
    for needle in forbidden_substrings:
        assert needle not in manifest_text

    for table_file in (pack_dir / "tables").glob("*.json"):
        text = table_file.read_text(encoding="utf-8")
        for needle in forbidden_substrings:
            assert needle not in text


def test_clean_pack_passes_scan_for_leakage(
    sample_manifest: dict[str, Any], sample_table_blocks: dict[str, dict[str, Any]]
) -> None:
    pack = ParamPack(manifest=sample_manifest, tables=sample_table_blocks)
    assert scan_for_leakage(pack) == []


def test_scan_for_leakage_rejects_high_cardinality_verbatim_array(
    sample_manifest: dict[str, Any],
) -> None:
    """A legally-named key (``quantile_bin_edges``) hiding a near-full-length
    verbatim value array must still be rejected (R1/R2 value-level guard)."""
    n_records = 500
    verbatim_leak = [float(i) + 0.123 for i in range(n_records)]  # all-distinct, len == n
    leaky_blocks = {
        "labs": {
            "n_records": n_records,
            "fitted": True,
            "params": {
                # legal key name, illegal content: a full verbatim column
                "creatinine_quantile_bin_edges": verbatim_leak,
            },
        }
    }
    pack = ParamPack(manifest=sample_manifest, tables=leaky_blocks)

    findings = scan_for_leakage(pack)
    assert len(findings) == 1
    assert findings[0].table == "labs"
    assert findings[0].length == n_records


def test_write_pack_rejects_fabricated_row_level_field(
    tmp_path: Path, sample_manifest: dict[str, Any]
) -> None:
    """A fabricated pack embedding a row-level field is rejected before
    anything is written to disk (verification requirement)."""
    fabricated_blocks = {
        "patient": {
            "n_records": 100,
            "fitted": True,
            "params": {"patient_id": ["P0001", "P0002", "P0003"]},
        }
    }

    pack_dir = tmp_path / "pack"
    with pytest.raises(LeakageError):
        write_pack(pack_dir, sample_manifest, fabricated_blocks)

    # Nothing should have been written.
    assert not pack_dir.exists()


def test_scan_for_leakage_flags_forbidden_key_regardless_of_value(
    sample_manifest: dict[str, Any],
) -> None:
    blocks = {"patient": {"n_records": 100, "fitted": True, "params": {"mrn": [1, 2, 3]}}}
    pack = ParamPack(manifest=sample_manifest, tables=blocks)

    findings = scan_for_leakage(pack)
    assert any("mrn" in f.reason for f in findings)


def test_prior_driven_table_with_zero_n_records_skips_value_scan(
    sample_manifest: dict[str, Any],
) -> None:
    """A prior-driven table (n_records=0) has no fitted record count for an
    array's length to "approach" — the value-level scan is a no-op there."""
    blocks = {
        "ecmo_mcs": {
            "n_records": 0,
            "fitted": False,
            "params": {"literature_rate_bins": list(range(100))},
        }
    }
    pack = ParamPack(manifest=sample_manifest, tables=blocks)
    assert scan_for_leakage(pack) == []


def test_json_files_are_actually_valid_json(
    tmp_path: Path,
    sample_manifest: dict[str, Any],
    sample_table_blocks: dict[str, dict[str, Any]],
) -> None:
    pack_dir = tmp_path / "pack"
    write_pack(pack_dir, sample_manifest, sample_table_blocks)

    assert json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8")) == sample_manifest

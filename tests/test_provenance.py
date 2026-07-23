"""Tests for ``clifforge.provenance`` (U4): the PROVENANCE.md renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clifforge.provenance import DEFAULT_CITATION, write_provenance


def test_write_provenance_marks_fitted_and_prior_driven_tables(tmp_path: Path) -> None:
    manifest: dict[str, Any] = {
        "pack_version": "1.0",
        "clif_version": "2.1.0",
        "fit_source": {"dataset_id": "MIMIC-IV-Ext-CLIF-sample", "commit": "abc1234"},
        "reference_source": {
            "mcide_url": "https://github.com/Common-Longitudinal-ICU-data-Format/CLIF",
            "mcide_commit": "deadbeef",
            "retrieved_date": "2026-07-01",
        },
        "tables": {
            "patient": {"fitted": True, "source": "MIMIC-IV-Ext-CLIF v1.1.0"},
            "ecmo_mcs": {"fitted": False, "source": "consortium clinical rules"},
        },
        "suppression_audit": {"patient": {"cells_suppressed": 1}},
    }

    out_path = tmp_path / "PROVENANCE.md"
    write_provenance(out_path, manifest)

    text = out_path.read_text(encoding="utf-8")
    assert "patient" in text
    assert "fitted" in text
    assert "ecmo_mcs" in text
    assert "prior-driven" in text
    assert "abc1234" in text
    assert "deadbeef" in text


def test_write_provenance_uses_default_citation_when_absent(tmp_path: Path) -> None:
    manifest: dict[str, Any] = {
        "pack_version": "1.0",
        "clif_version": "2.1.0",
        "fit_source": {"dataset_id": "MIMIC-IV-Ext-CLIF-sample", "commit": "abc1234"},
        "tables": {},
        "suppression_audit": {},
    }

    out_path = tmp_path / "PROVENANCE.md"
    write_provenance(out_path, manifest)

    text = out_path.read_text(encoding="utf-8")
    assert DEFAULT_CITATION in text


def test_write_provenance_uses_manifest_citation_when_present(tmp_path: Path) -> None:
    manifest: dict[str, Any] = {
        "pack_version": "1.0",
        "clif_version": "2.1.0",
        "fit_source": {"dataset_id": "x", "commit": "y"},
        "tables": {},
        "suppression_audit": {},
        "citation": "Custom CLIF-MIMIC citation line.",
    }

    out_path = tmp_path / "PROVENANCE.md"
    write_provenance(out_path, manifest)

    text = out_path.read_text(encoding="utf-8")
    assert "Custom CLIF-MIMIC citation line." in text

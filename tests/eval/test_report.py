"""Tests for evaluation report assembly (U25, R26).

The report must carry all four evaluation surfaces, must not silently omit or
fake the comparative ones when no reference is supplied, and must state what the
reference actually was — so synthetic-vs-synthetic self-consistency can never
read as real-data fidelity.
"""

from __future__ import annotations

import pytest

from clifforge.eval.report import build_report
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset

_SECTIONS = (
    "## 1. Dataset",
    "## 2. Validation",
    "## 3. Fidelity",
    "## 4. Privacy",
    "## 5. Utility",
)


@pytest.fixture
def demo(pack: ParamPack) -> dict:
    return generate_dataset(pack, n_patients=80, seed=42).tables


@pytest.fixture
def reference(pack: ParamPack) -> dict:
    return generate_dataset(pack, n_patients=80, seed=43).tables


def test_all_sections_present_without_reference(demo: dict) -> None:
    md = build_report(demo, run_secondary=False)
    for section in _SECTIONS:
        assert section in md
    # comparative sections are explicitly not computed, not silently dropped
    assert md.count("Not computed") >= 3


def test_all_sections_present_with_reference(demo: dict, reference: dict) -> None:
    md = build_report(
        demo, reference=reference, reference_label="a second draw", run_secondary=False
    )
    for section in _SECTIONS:
        assert section in md
    assert "Not computed" not in md
    assert "Quality" in md  # fidelity table rendered
    assert "DCR (median)" in md  # privacy table rendered
    assert "TSTR AUC" in md  # utility table rendered


def test_reference_label_is_surfaced(demo: dict, reference: dict) -> None:
    # The label is what stops a synthetic-vs-synthetic comparison from reading as
    # real-data fidelity, so it must appear verbatim in the report.
    label = "a second independent synthetic draw (self-consistency, NOT real-data fidelity)"
    md = build_report(demo, reference=reference, reference_label=label, run_secondary=False)
    assert label in md


def test_validation_section_reports_pass_for_conformant_data(demo: dict) -> None:
    md = build_report(demo, run_secondary=False)
    assert "All tables pass the primary (pandera) conformance gate." in md


def test_dataset_section_counts_tables_and_rows(demo: dict) -> None:
    md = build_report(demo, run_secondary=False)
    assert f"{len(demo)} tables" in md
    assert "| `patient` |" in md


def test_report_states_data_is_synthetic(demo: dict) -> None:
    md = build_report(demo, run_secondary=False)
    assert "fully synthetic" in md

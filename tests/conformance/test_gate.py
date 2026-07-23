"""Two-gate conformance harness tests (U3): fail-loud on each violation class.

Covers R5/R7/R9 enforcement, the KTD-5 LazyFrame-collect gotcha, and the R16/R17
policy that clifpy is an advisory secondary gate that never blocks. The
``ConformanceError`` raised here is what the CLI turns into a nonzero exit
(R25, AE5).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from clifforge.conformance import ConformanceError, GateReport, validate


def _patient() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "sex_category": ["Male", "Female"],
            "race_category": ["White", "Black or African American"],
            "death_dttm": [datetime(2021, 1, 1, tzinfo=UTC), None],
        }
    )


def _resp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hospitalization_id": ["h1"],
            "device_id": ["d1"],
            "device_category": ["IMV"],
            "fio2_set": [0.4],
        }
    )


def test_valid_frame_passes_primary_gate() -> None:
    report = validate(_patient(), "patient")
    assert isinstance(report, GateReport)
    assert report.pandera_passed is True
    assert report.n_rows == 2


def test_bad_category_wrong_case_raises() -> None:
    bad = _patient().with_columns(pl.lit("male").alias("sex_category"))
    with pytest.raises(ConformanceError) as exc:
        validate(bad, "patient")
    assert exc.value.table == "patient"
    # failure_cases carries the offending column for CLI rendering.
    assert exc.value.failure_cases is not None
    assert "sex_category" in exc.value.failure_cases.get_column("column").to_list()


def test_out_of_bounds_numeric_raises_ae5() -> None:
    # fio2_set bound is (0.21, 1.0); 1.5 must fail in_range (AE5, R25).
    bad = _resp().with_columns(pl.lit(1.5).alias("fio2_set"))
    with pytest.raises(ConformanceError):
        validate(bad, "respiratory_support")


def test_naive_datetime_raises_tz() -> None:
    naive = _patient().with_columns(pl.col("death_dttm").dt.replace_time_zone(None))
    with pytest.raises(ConformanceError):
        validate(naive, "patient")


def test_lazyframe_is_collected_and_value_checks_fire() -> None:
    # A bad category inside a LazyFrame must still fail — the gate .collect()s
    # first so value-level isin actually runs (KTD-5). A LazyFrame that skipped
    # value checks would pass here; it must not.
    bad_lazy = _patient().with_columns(pl.lit("male").alias("sex_category")).lazy()
    with pytest.raises(ConformanceError):
        validate(bad_lazy, "patient")


def test_valid_lazyframe_passes() -> None:
    report = validate(_patient().lazy(), "patient")
    assert report.pandera_passed is True


def test_clifpy_secondary_never_blocks_a_pandera_valid_frame() -> None:
    # clifpy may disagree (CLIF-version parity), but it is advisory — a frame
    # that passes pandera returns a report, never raises, whatever clifpy says.
    report = validate(_patient(), "patient")
    assert report.clifpy_status in {"passed", "failed", "skipped"}
    assert report.clifpy_note  # always carries a note


def test_clifpy_skipped_with_note_where_no_validator() -> None:
    # invasive_hemodynamics is a Concept-tier table clifpy has no class for.
    frame = pl.DataFrame({"hospitalization_id": ["h1"]})
    report = validate(frame, "invasive_hemodynamics")
    assert report.clifpy_status == "skipped"
    assert "no clifpy validator" in report.clifpy_note


def test_secondary_gate_can_be_disabled() -> None:
    report = validate(_patient(), "patient", run_secondary=False)
    assert report.clifpy_status == "skipped"
    assert "disabled" in report.clifpy_note


def test_unknown_table_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        validate(_patient(), "not_a_clif_table")

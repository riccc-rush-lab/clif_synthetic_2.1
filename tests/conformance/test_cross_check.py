"""synthetic_clif schema-agreement cross-check tests (U3, R18).

The live cross-check is dev-only and skipped when ``synthetic_clif`` is not
installed. The column-agreement *logic* is unit-tested unconditionally so the
comparison itself is verified without the optional dependency.
"""

from __future__ import annotations

from clifforge import schemas
from clifforge.conformance import cross_check


def test_cross_check_skips_cleanly_without_synthetic_clif() -> None:
    result = cross_check.cross_check_table("patient")
    if cross_check.synthetic_clif_available():
        # When present, it must actually compare (not skip on the import gate).
        assert result.skipped is False or "does not emit" in result.note
    else:
        assert result.skipped is True
        assert "synthetic_clif not installed" in result.note
        # A skip is treated as agreement so CI stays green until the dep lands.
        assert result.agrees is True


def test_column_agreement_flags_missing_required_column() -> None:
    # patient_id is a required schema column; omit it -> disagreement.
    schema = schemas.get_schema("patient")
    external = set(schema.columns) - {"patient_id"}
    result = cross_check.column_agreement("patient", external)
    assert result.agrees is False
    assert "patient_id" in result.missing_required


def test_column_agreement_passes_when_required_columns_present() -> None:
    schema = schemas.get_schema("patient")
    external = set(schema.columns)
    result = cross_check.column_agreement("patient", external)
    assert result.agrees is True
    assert result.missing_required == ()


def test_column_agreement_tolerates_extra_external_columns() -> None:
    # strict=False: columns synthetic_clif emits but we don't model are reported,
    # not fatal.
    schema = schemas.get_schema("patient")
    external = set(schema.columns) | {"some_future_2_1_1_column"}
    result = cross_check.column_agreement("patient", external)
    assert result.agrees is True
    assert "some_future_2_1_1_column" in result.external_only

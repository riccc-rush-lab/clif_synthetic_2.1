"""Schema-level tests for the generated ``patient`` pandera schema (U3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandera.polars as pa
import polars as pl
import pytest

from clifforge import schemas


def _valid() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "sex_category": ["Male", "Female"],
            "race_category": ["White", "Black or African American"],
            "death_dttm": [datetime(2021, 1, 1, tzinfo=UTC), None],
        }
    )


def test_valid_frame_passes() -> None:
    schema = schemas.get_schema("patient")
    validated = schema.validate(_valid())
    assert validated.height == 2


def test_wrong_case_category_fails_isin() -> None:
    schema = schemas.get_schema("patient")
    bad = _valid().with_columns(pl.lit("male").alias("sex_category"))  # wrong case
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(bad)


def test_naive_datetime_fails_tz() -> None:
    schema = schemas.get_schema("patient")
    naive = _valid().with_columns(pl.col("death_dttm").dt.replace_time_zone(None))
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(naive)


def test_missing_required_id_fails() -> None:
    schema = schemas.get_schema("patient")
    no_id = _valid().drop("patient_id")
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(no_id)


def test_extra_source_columns_allowed() -> None:
    # strict=False — an extra source/*_name column must not be rejected.
    schema = schemas.get_schema("patient")
    extra = _valid().with_columns(pl.lit("self-reported").alias("race_source"))
    assert schema.validate(extra).height == 2

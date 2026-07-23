"""Schema-level tests for the generated ``respiratory_support`` schema (U3).

Exercises the outlier ``in_range`` bounds (R9) and mCIDE ``isin`` membership
(R5) on a numeric-heavy table.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandera.polars as pa
import polars as pl
import pytest

from clifforge import schemas


def _valid() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hospitalization_id": ["h1", "h2"],
            "device_id": ["d1", "d2"],
            "recorded_dttm": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
            ],
            "device_category": ["IMV", "NIPPV"],
            "fio2_set": [0.4, 0.6],
            "peep_set": [5.0, 8.0],
        }
    )


def test_valid_frame_passes() -> None:
    schema = schemas.get_schema("respiratory_support")
    assert schema.validate(_valid()).height == 2


def test_fio2_above_bound_fails_in_range() -> None:
    schema = schemas.get_schema("respiratory_support")
    # fio2_set bound is (0.21, 1.0); 1.5 is out of range (AE5).
    bad = _valid().with_columns(pl.lit(1.5).alias("fio2_set"))
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(bad)


def test_fio2_below_bound_fails_in_range() -> None:
    schema = schemas.get_schema("respiratory_support")
    bad = _valid().with_columns(pl.lit(0.05).alias("fio2_set"))
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(bad)


def test_bad_device_category_fails_isin() -> None:
    schema = schemas.get_schema("respiratory_support")
    bad = _valid().with_columns(pl.lit("imv").alias("device_category"))  # wrong case
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(bad)

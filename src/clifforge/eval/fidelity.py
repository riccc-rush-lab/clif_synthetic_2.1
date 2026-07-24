"""Fidelity metrics — SDMetrics column-shape and column-pair similarity (U24; R24).

How closely does each synthetic table's *distribution* match the real one? Two
families, mirroring SDMetrics' quality report:

* **Column shapes** (marginals): ``KSComplement`` for numeric columns,
  ``TVComplement`` for categorical ones.
* **Column pair trends** (joint structure): ``CorrelationSimilarity`` for
  numeric-numeric pairs, ``ContingencySimilarity`` for categorical-categorical.

Each metric is already normalized to ``[0, 1]`` (1 = identical distribution), so a
table's **quality score** is the mean of its column-shape and column-pair
components — a shape-only score when a table has too few columns to form pairs.

Identifier columns (``*_id``) are excluded: they are arbitrary surrogate keys, so
"distribution similarity" over them is meaningless and would dilute the score.
Datetime columns are likewise excluded — their absolute placement is a calendar
choice of the orchestrator, not fitted structure. Columns that are constant or
all-null in either frame are skipped rather than allowed to error.

Polars throughout; ``.to_pandas()`` happens only at the SDMetrics boundary.
Metrics are deterministic given the inputs.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

__all__ = ["TableFidelity", "fidelity_report", "table_fidelity"]

#: Bound the pair work so a wide table can't blow up combinatorially. Pairs are
#: taken in sorted column order, so the selection stays deterministic.
_MAX_PAIRS_PER_KIND = 20

#: A string column is treated as a *category* only when its vocabulary is bounded.
#: Effectively-unique text (surrogate codes like ``product_code``, stringified
#: values, string timestamps) is free text, not a categorical distribution:
#: scoring it with TVComplement/ContingencySimilarity measures how unlikely two
#: draws are to reuse the same arbitrary string, not distributional fidelity, and
#: drags a table's score down for no real defect.
_MAX_DISTINCT_RATIO = 0.2
_RATIO_CHECK_MIN_DISTINCT = 20


@dataclass(frozen=True)
class TableFidelity:
    """Distribution-similarity scores for one table (all in ``[0, 1]``, 1 = best)."""

    table: str
    quality_score: float
    column_shape_score: float
    column_pair_score: float | None  # None when the table has no usable pairs
    n_columns: int
    n_pairs: int


def _usable(real: pl.DataFrame, synth: pl.DataFrame, col: str) -> bool:
    """A column is usable when both sides have at least one non-null value."""
    return real[col].null_count() < real.height and synth[col].null_count() < synth.height


def _is_bounded_vocabulary(frame: pl.DataFrame, col: str) -> bool:
    """True when a string column reads as a category rather than free text/an id."""
    if frame.height == 0:
        return False
    distinct = frame[col].n_unique()
    if distinct < _RATIO_CHECK_MIN_DISTINCT:
        return True  # a small vocabulary is a category by definition
    return (distinct / frame.height) <= _MAX_DISTINCT_RATIO


def _column_kinds(real: pl.DataFrame, synth: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Shared comparable columns split into (numeric, categorical)."""
    numeric: list[str] = []
    categorical: list[str] = []
    for col in sorted(set(real.columns) & set(synth.columns)):
        if col.endswith("_id"):
            continue  # surrogate keys carry no distribution
        dtype = real.schema[col]
        if dtype != synth.schema[col]:
            continue
        if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
            continue  # calendar placement is an orchestrator choice, not fitted
        if not _usable(real, synth, col):
            continue
        if dtype.is_numeric():
            numeric.append(col)
        elif dtype == pl.String and (
            _is_bounded_vocabulary(real, col) and _is_bounded_vocabulary(synth, col)
        ):
            categorical.append(col)
    return numeric, categorical


def _safe(compute: Any, *args: Any) -> float | None:
    """Run a metric, returning ``None`` when it cannot be computed for this column."""
    try:
        value = float(compute(*args))
    except Exception:  # noqa: BLE001 - a degenerate column must skip, not abort the report
        return None
    return None if value != value else value  # drop NaN


def table_fidelity(table: str, real: pl.DataFrame, synth: pl.DataFrame) -> TableFidelity | None:
    """Column-shape + column-pair fidelity for one table (R24).

    Returns ``None`` when the table is empty on either side or has no comparable
    columns, so an unpopulated table cannot masquerade as a perfect score.
    """
    from sdmetrics.column_pairs import (  # type: ignore[import-untyped]
        ContingencySimilarity,
        CorrelationSimilarity,
    )
    from sdmetrics.single_column import (  # type: ignore[import-untyped]
        KSComplement,
        TVComplement,
    )

    if real.height == 0 or synth.height == 0:
        return None
    numeric, categorical = _column_kinds(real, synth)
    if not numeric and not categorical:
        return None

    shapes: list[float] = []
    for col in numeric:
        score = _safe(KSComplement.compute, real[col].to_pandas(), synth[col].to_pandas())
        if score is not None:
            shapes.append(score)
    for col in categorical:
        score = _safe(TVComplement.compute, real[col].to_pandas(), synth[col].to_pandas())
        if score is not None:
            shapes.append(score)
    if not shapes:
        return None

    pairs: list[float] = []
    for metric, columns in (
        (CorrelationSimilarity, numeric),
        (ContingencySimilarity, categorical),
    ):
        for left, right in itertools.islice(
            itertools.combinations(columns, 2), _MAX_PAIRS_PER_KIND
        ):
            score = _safe(
                metric.compute,
                real.select(left, right).to_pandas(),
                synth.select(left, right).to_pandas(),
            )
            if score is not None:
                pairs.append(score)

    shape_score = sum(shapes) / len(shapes)
    pair_score = sum(pairs) / len(pairs) if pairs else None
    quality = shape_score if pair_score is None else (shape_score + pair_score) / 2.0

    return TableFidelity(
        table=table,
        quality_score=quality,
        column_shape_score=shape_score,
        column_pair_score=pair_score,
        n_columns=len(shapes),
        n_pairs=len(pairs),
    )


def fidelity_report(
    synthetic: Mapping[str, pl.DataFrame],
    real: Mapping[str, pl.DataFrame],
) -> dict[str, TableFidelity]:
    """Per-table fidelity across every table present on both sides (R24)."""
    report: dict[str, TableFidelity] = {}
    for table in sorted(set(synthetic) & set(real)):
        result = table_fidelity(table, real[table], synthetic[table])
        if result is not None:
            report[table] = result
    return report

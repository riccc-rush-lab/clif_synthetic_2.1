"""Two-gate CLIF 2.1.0 conformance harness (U3).

``validate(df, table)`` runs the **primary** pandera gate then the **secondary**
clifpy gate:

- **pandera (primary, hard gate)** — the per-table :mod:`clifforge.schemas`
  ``SCHEMA`` validates the *eager* frame with ``lazy=True`` so every mCIDE-
  membership (R5), tz-aware-datetime (R7), and outlier-bounds (R9) violation is
  collected before failing. Any violation raises :class:`ConformanceError`,
  which the CLI turns into a nonzero exit (R25, AE5).
- **clifpy (secondary, advisory)** — run per-table only where a clifpy validator
  exists, and *never* allowed to block: clifpy validates primarily against CLIF
  2.0 with partial 2.1 Beta coverage, so a clifpy disagreement is recorded as a
  note, not raised (R16, R17). Where clifpy has no class for a table it is
  skipped-with-note; pandera alone gates that table and the gap is recorded.

pandera's value-level checks (``isin``, ``in_range``) only fire on a materialized
``pl.DataFrame`` — on a bare ``LazyFrame`` it silently runs schema-level checks
only. So the gate ``.collect()``s any ``LazyFrame`` before validating (KTD-5),
guaranteeing membership and bounds actually run.
"""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import pandera.polars as pa
import polars as pl

from clifforge import schemas

__all__ = ["ConformanceError", "GateReport", "validate"]


class ConformanceError(Exception):
    """A table failed the primary (pandera) conformance gate.

    Carries the pandera ``failure_cases`` frame (one row per violated check) so
    a caller/CLI can render exactly which columns and checks failed.
    """

    def __init__(self, table: str, failure_cases: pl.DataFrame | None, message: str) -> None:
        self.table = table
        self.failure_cases = failure_cases
        super().__init__(message)


@dataclass(frozen=True)
class GateReport:
    """Outcome of both gates for one table (returned only when pandera passes)."""

    table: str
    n_rows: int
    pandera_passed: bool
    clifpy_status: str  # "passed" | "failed" | "skipped"
    clifpy_note: str
    clifpy_error_count: int = 0


def validate(
    df: pl.DataFrame | pl.LazyFrame, table: str, *, run_secondary: bool = True
) -> GateReport:
    """Run the two-gate conformance harness for ``table`` over ``df``.

    ``df`` may be a ``LazyFrame`` — it is ``.collect()``ed first so value-level
    checks fire (KTD-5). Raises :class:`ConformanceError` on any pandera
    violation; the clifpy secondary gate is advisory and never raises.
    """
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"validate expects a polars DataFrame/LazyFrame, got {type(frame).__name__}"
        )

    schema = schemas.get_schema(table)
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = _failure_cases(exc)
        raise ConformanceError(table, cases, _format_failure(table, cases)) from exc

    if run_secondary:
        status, note, count = _run_clifpy(frame, table)
    else:
        status, note, count = "skipped", "clifpy secondary gate disabled by caller", 0

    return GateReport(
        table=table,
        n_rows=frame.height,
        pandera_passed=True,
        clifpy_status=status,
        clifpy_note=note,
        clifpy_error_count=count,
    )


def _failure_cases(exc: pa.errors.SchemaErrors) -> pl.DataFrame | None:
    cases = getattr(exc, "failure_cases", None)
    if isinstance(cases, pl.DataFrame):
        return cases
    if cases is None:
        return None
    # pandera may hand back a pandas frame depending on backend — normalize.
    try:
        return pl.from_pandas(cases)
    except (TypeError, ValueError):
        return None


def _format_failure(table: str, cases: pl.DataFrame | None) -> str:
    if cases is None or cases.is_empty():
        return f"Table {table!r} failed pandera conformance."
    checks = cases.get_column("check").to_list() if "check" in cases.columns else []
    cols = (
        cases.get_column("column").drop_nulls().unique().to_list()
        if "column" in cases.columns
        else []
    )
    detail = ", ".join(dict.fromkeys(str(c) for c in checks))
    where = ", ".join(str(c) for c in cols)
    return (
        f"Table {table!r} failed pandera conformance: "
        f"{cases.height} failing case(s) across [{where}] — checks: {detail}"
    )


def _clifpy_class(table: str) -> type | None:
    """Return the clifpy table class for ``table``, or ``None`` if unavailable.

    clifpy uses PascalCase per-table classes (``patient`` -> ``Patient``,
    ``respiratory_support`` -> ``RespiratorySupport``). A table with no matching
    class (Concept-tier tables clifpy does not model) yields ``None`` -> skip.
    """
    try:
        from clifpy import tables as clifpy_tables
    except ImportError:
        return None
    class_name = "".join(part.capitalize() for part in table.split("_"))
    cls = getattr(clifpy_tables, class_name, None)
    return cls if isinstance(cls, type) else None


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Silence clifpy's emoji log/print chatter during secondary validation."""
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous_disable)


def _run_clifpy(frame: pl.DataFrame, table: str) -> tuple[str, str, int]:
    """Advisory secondary gate — records clifpy's verdict, never blocks (R17).

    Returns ``(status, note, error_count)`` where status is one of ``passed``
    (clifpy validates clean), ``failed`` (clifpy reports errors — recorded, not
    raised, honoring the CLIF-version-parity gap), or ``skipped`` (no clifpy
    class for the table, or clifpy raised/absent).
    """
    cls = _clifpy_class(table)
    if cls is None:
        return (
            "skipped",
            f"no clifpy validator for {table!r} (pandera alone gates this table)",
            0,
        )
    try:
        with tempfile.TemporaryDirectory() as tmp, _quiet():
            model = cls(data=frame.to_pandas(), output_directory=tmp, clif_version="2.1")
            model.validate()
            errors = list(getattr(model, "errors", []) or [])
            valid = bool(model.isvalid())
    except Exception as exc:  # advisory gate: any clifpy failure is a skip-with-note
        return (
            "skipped",
            f"clifpy validation unavailable for {table!r}: {type(exc).__name__}: {exc}",
            0,
        )
    if valid and not errors:
        return "passed", f"clifpy secondary gate passed for {table!r}", 0
    return (
        "failed",
        f"clifpy reported {len(errors)} advisory issue(s) for {table!r} "
        "(recorded, not blocking — pandera is the gate; R17 CLIF-version-parity gap)",
        len(errors),
    )

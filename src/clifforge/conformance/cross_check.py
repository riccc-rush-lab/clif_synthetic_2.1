"""``synthetic_clif`` schema-agreement cross-check (U3, R18).

An independent conformance cross-check: generate a tiny corpus with the
consortium's ``synthetic_clif`` and confirm the two tools agree on column names
per table. This is a **dev-/CI-only** check — ``synthetic_clif`` is upstream
prior art, never a hard runtime dependency (Scope Boundaries). When it is not
installed the cross-check is skipped-with-note rather than failing.

The comparison is intentionally name-level and directional: CLIFForge schemas
are ``strict=False`` (extra source/``*_name`` columns are allowed), so agreement
means *every column the schema marks required is present in the synthetic_clif
output* and the two share their identifier/category backbone. Dtype divergence
is reported as an advisory note, not a hard failure, because the two tools carry
independent (and both valid) polars/pandas dtype choices for optional columns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

from clifforge import schemas

__all__ = ["AgreementResult", "synthetic_clif_available", "column_agreement", "cross_check_table"]


def synthetic_clif_available() -> bool:
    """True when the optional ``synthetic_clif`` dev dependency is importable."""
    return find_spec("synthetic_clif") is not None


@dataclass(frozen=True)
class AgreementResult:
    """Directional column-name agreement between a schema and an external table."""

    table: str
    agrees: bool
    missing_required: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    external_only: tuple[str, ...] = ()
    skipped: bool = False
    note: str = ""


def _required_columns(table: str) -> list[str]:
    schema = schemas.get_schema(table)
    return [name for name, col in schema.columns.items() if getattr(col, "required", False)]


def column_agreement(table: str, external_columns: set[str]) -> AgreementResult:
    """Compare our schema's columns for ``table`` against an external column set.

    Agreement requires that every *required* schema column appears in
    ``external_columns``. Extra columns on either side are reported but do not
    break agreement (schemas are ``strict=False``; the external tool may emit
    optional columns we leave off, and vice versa).
    """
    schema = schemas.get_schema(table)
    schema_cols = set(schema.columns)
    required = set(_required_columns(table))
    missing = tuple(sorted(required - external_columns))
    shared = tuple(sorted(schema_cols & external_columns))
    external_only = tuple(sorted(external_columns - schema_cols))
    agrees = not missing
    note = f"{table!r}: {len(shared)} shared column(s)" + (
        f"; missing required {list(missing)}" if missing else "; all required columns present"
    )
    return AgreementResult(
        table=table,
        agrees=agrees,
        missing_required=missing,
        shared=shared,
        external_only=external_only,
        note=note,
    )


def cross_check_table(table: str, *, n_rows: int = 10) -> AgreementResult:
    """Generate ``table`` with ``synthetic_clif`` and check column-name agreement.

    Skipped-with-note when ``synthetic_clif`` is not installed. Kept dependency-
    injection-friendly: the actual generation call is resolved lazily so the
    module imports cleanly without the optional dependency.
    """
    if not synthetic_clif_available():
        return AgreementResult(
            table=table,
            agrees=True,
            skipped=True,
            note="synthetic_clif not installed — cross-check skipped (dev-only, R18)",
        )
    external_columns = _synthetic_clif_columns(table, n_rows=n_rows)
    if external_columns is None:
        return AgreementResult(
            table=table,
            agrees=True,
            skipped=True,
            note=f"synthetic_clif does not emit table {table!r} — cross-check skipped",
        )
    return column_agreement(table, external_columns)


def _synthetic_clif_columns(table: str, *, n_rows: int) -> set[str] | None:
    """Best-effort extraction of ``synthetic_clif``'s column set for ``table``.

    synthetic_clif's public surface differs across releases; this probes the
    documented entry points and returns ``None`` if the table can't be produced
    (caller renders that as skip-with-note rather than a hard failure).
    """
    import importlib

    sc = importlib.import_module("synthetic_clif")
    generate = getattr(sc, "generate", None) or getattr(sc, "generate_tables", None)
    if generate is None:
        return None
    corpus = generate(n=n_rows) if _accepts_n(generate) else generate()
    frame = corpus.get(table) if isinstance(corpus, dict) else getattr(corpus, table, None)
    if frame is None:
        return None
    columns = getattr(frame, "columns", None)
    return set(columns) if columns is not None else None


def _accepts_n(fn: Callable[..., object]) -> bool:
    import inspect

    try:
        return "n" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

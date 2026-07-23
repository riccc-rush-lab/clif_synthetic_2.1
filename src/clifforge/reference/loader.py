"""Read-only accessors for the vendored CLIF 2.1.0 reference data (U2).

The vendored ``data/manifest.json`` is self-describing: it maps every
``(table, field)`` to the relative path of the mCIDE category CSV that defines
the field's permissible values, and every table to its outlier-threshold CSV.
This module exposes that data as plain Python so schema construction (U3) and
sampling never parse filenames or guess at spellings.

Design contract (R4, R9): a *missing* table/field is a programming error, not an
empty result — every accessor RAISES ``ReferenceDataError`` rather than returning
an empty list or silently degrading. A schema built against a field the vendor
never captured must fail loudly.
"""

from __future__ import annotations

import csv
import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

_DATA_ROOT = Path(__file__).parent / "data"
_MANIFEST_PATH = _DATA_ROOT / "manifest.json"
_DICTIONARY_PATH = _DATA_ROOT / "dictionary.json"


class ReferenceDataError(LookupError):
    """A requested table, field, or reference file is absent from the vendor set."""


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        raise ReferenceDataError(
            f"Reference manifest not found at {_MANIFEST_PATH}. "
            "Run `uv run python scripts/vendor_reference.py` to vendor CLIF 2.1.0 data."
        )
    with _MANIFEST_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@lru_cache(maxsize=1)
def _dictionary() -> dict[str, Any]:
    if not _DICTIONARY_PATH.exists():
        raise ReferenceDataError(
            f"Data dictionary not found at {_DICTIONARY_PATH}. "
            "Run `uv run python scripts/vendor_dictionary.py` to vendor the CLIF 2.1.0 dictionary."
        )
    with _DICTIONARY_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@cache
def _read_first_column(rel_path: str) -> tuple[str, ...]:
    """Return the first-column values (the category values) of a vendored CSV.

    Skips the header row and blank lines. Values are returned verbatim (CLIF
    mCIDE categories are case-sensitive and matched exactly downstream).
    """
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    values: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value:
                values.append(value)
    return tuple(values)


def provenance() -> dict[str, str]:
    """Return the source provenance recorded at vendor time (version, commit, ...)."""
    m = _manifest()
    return {
        "clif_version": m["clif_version"],
        "source_repo": m["source_repo"],
        "source_commit": m["source_commit"],
        "source_ref": m["source_ref"],
        "retrieved_at": m["retrieved_at"],
    }


def tables() -> list[str]:
    """All tables that have at least one vendored mCIDE category field."""
    return sorted(_manifest()["mcide"].keys())


def dictionary_tables() -> list[str]:
    """All CLIF 2.1.0 tables defined in the vendored data dictionary."""
    return sorted(_dictionary()["tables"].keys())


def table_columns(table: str) -> list[dict[str, str]]:
    """Return the dictionary column list for ``table`` as ``[{name, dtype}, ...]``.

    ``dtype`` is the CLIF data-dictionary type (VARCHAR / DATETIME / DOUBLE /
    INT / ...), or ``UNKNOWN`` for Concept-tier columns the dictionary documents
    without a data type. Raises if the table is not in the dictionary.
    """
    tables_map = _dictionary()["tables"]
    if table not in tables_map:
        raise ReferenceDataError(
            f"No dictionary entry for table {table!r}. "
            f"Known tables: {', '.join(sorted(tables_map))}"
        )
    return [dict(c) for c in tables_map[table]["columns"]]


def table_maturity(table: str) -> str:
    """Return ``'beta'`` or ``'concept'`` for a dictionary table."""
    tables_map = _dictionary()["tables"]
    if table not in tables_map:
        raise ReferenceDataError(f"No dictionary entry for table {table!r}.")
    maturity = tables_map[table].get("maturity")
    if maturity is None:
        raise ReferenceDataError(f"Table {table!r} has no recorded maturity tier.")
    return str(maturity)


def dictionary_provenance() -> dict[str, str]:
    """Source provenance for the vendored data dictionary (repo, commit, ...)."""
    d = _dictionary()
    return {
        "clif_version": d["clif_version"],
        "source_repo": d["source_repo"],
        "source_commit": d["source_commit"],
        "source_path": d["source_path"],
        "retrieved_at": d["retrieved_at"],
    }


def mcide_fields(table: str) -> list[str]:
    """All mCIDE category fields vendored for ``table`` (raises if table unknown)."""
    mcide = _manifest()["mcide"]
    if table not in mcide:
        raise ReferenceDataError(
            f"No mCIDE reference data for table {table!r}. Known tables: {', '.join(sorted(mcide))}"
        )
    return sorted(mcide[table].keys())


def categories(table: str, field: str) -> list[str]:
    """Return the exact, case-sensitive permissible values for ``table.field``.

    Raises ``ReferenceDataError`` if the table or field is not in the vendor set —
    never returns an empty list for a missing field.
    """
    mcide = _manifest()["mcide"]
    if table not in mcide:
        raise ReferenceDataError(
            f"No mCIDE reference data for table {table!r}. Known tables: {', '.join(sorted(mcide))}"
        )
    fields = mcide[table]
    if field not in fields:
        raise ReferenceDataError(
            f"No mCIDE field {field!r} for table {table!r}. "
            f"Known fields: {', '.join(sorted(fields))}"
        )
    return list(_read_first_column(fields[field]))


def bounds(table: str, field: str) -> tuple[float, float]:
    """Return ``(lower_limit, upper_limit)`` plausibility bounds for ``table.field``.

    ``field`` is the key used within the table's outlier CSV — the category or
    variable name (e.g. ``"heart_rate"`` for vitals, ``"fio2_set"`` for
    respiratory_support). Raises ``ReferenceDataError`` if the table has no
    outlier file or the field is not listed in it.
    """
    outliers = _manifest()["outliers"]
    if table not in outliers:
        raise ReferenceDataError(
            f"No outlier-threshold reference data for table {table!r}. "
            f"Known tables: {', '.join(sorted(outliers))}"
        )
    rel_path = outliers[table]
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ReferenceDataError(f"Outlier file has no header: {path}")
        # Simple two-sided numeric bounds only. Some upstream files (ecmo_mcs)
        # use a multi-column ranged-text schema that does not reduce to a single
        # (lower, upper) pair — reject those loudly rather than fabricate inf.
        if "lower_limit" not in fieldnames or "upper_limit" not in fieldnames:
            raise ReferenceDataError(
                f"Outlier file for table {table!r} has a non-standard schema "
                f"(columns: {', '.join(fieldnames)}); it does not expose "
                "lower_limit/upper_limit numeric bounds."
            )
        key_col = fieldnames[0]
        for row in reader:
            if (row.get(key_col) or "").strip() == field:
                lower = _parse_limit(row.get("lower_limit"), default=float("-inf"))
                upper = _parse_limit(row.get("upper_limit"), default=float("inf"))
                return (lower, upper)
    raise ReferenceDataError(f"No outlier bounds for {field!r} in table {table!r} ({rel_path}).")


def outlier_keys(table: str) -> list[str]:
    """List the keys (categories / variable names) bounded for ``table``."""
    outliers = _manifest()["outliers"]
    if table not in outliers:
        raise ReferenceDataError(
            f"No outlier-threshold reference data for table {table!r}. "
            f"Known tables: {', '.join(sorted(outliers))}"
        )
    path = _DATA_ROOT / outliers[table]
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ReferenceDataError(f"Outlier file has no header: {path}")
        key_col = fieldnames[0]
        return [
            (row.get(key_col) or "").strip() for row in reader if (row.get(key_col) or "").strip()
        ]


def _parse_limit(raw: str | None, *, default: float) -> float:
    """Parse an outlier limit cell to float; blank/absent falls back to ``default``.

    ``default`` carries the sign of the unbounded side (``-inf`` for a missing
    lower limit, ``+inf`` for a missing upper limit).
    """
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ReferenceDataError(f"Non-numeric outlier limit {raw!r}") from exc

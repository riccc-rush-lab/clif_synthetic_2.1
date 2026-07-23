"""Versioned, inspectable parameter-pack read/write (KTD-2).

A parameter pack is a plain DIRECTORY, never a pickle:

    <pack_dir>/
        manifest.json          # pack_version, clif_version, fit_source, ...
        tables/<table>.json    # one aggregate parameter block per table

Every value in the pack must be an aggregate statistic (R1): a fitted count,
a parametric-family parameter, or a coarse quantile-bin edge list backed by
>= 20 real observations (R2) — never a raw per-record value. Two mechanized
guards enforce this before a pack is ever written or published:

1. A key/schema check rejects blocks that use a known row-level identifier
   key (``patient_id``, ``mrn``, ...) anywhere, however deeply nested.
2. ``scan_for_leakage`` is a VALUE-level scanner: it rejects any numeric
   array whose length or distinct-value count approaches the table's
   declared ``n_records`` — this catches real values smuggled into an
   otherwise legally-named block (e.g. an "empirical quantile" array that is
   secretly the full raw column, or per-stratum residuals).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pack format version. Compatibility is checked on the MAJOR component only:
# a "1.x" pack loads under any "1.y" reader, but a "2.0" pack is rejected by
# a reader pinned to "1.x" (and vice versa).
PACK_FORMAT_VERSION = "1.0"

# Manifest keys every pack must carry (R4, KTD-2). ``fit_source`` records a
# dataset id + commit only — never a patient/record identifier.
_REQUIRED_MANIFEST_KEYS = frozenset(
    {"pack_version", "clif_version", "fit_source", "suppression_audit"}
)

# Key names that would indicate a row-level / real-record field leaked into
# an aggregate pack (R1). Matched case-insensitively against every key in the
# manifest and every table block, at any nesting depth.
FORBIDDEN_ROW_LEVEL_KEYS = frozenset(
    {
        "patient_id",
        "hospitalization_id",
        "encounter_id",
        "subject_id",
        "mrn",
        "ssn",
        "record_id",
        "row_id",
        "raw_row",
        "raw_rows",
        "raw_record",
        "raw_records",
        "date_of_birth",
        "dob",
    }
)

# --- Leakage-scanner thresholds -------------------------------------------
#
# Rationale (R1/R2 mechanized enforcement): a legally-emitted continuous
# marginal is either (a) a parametric family — a handful of scalar
# parameters — or (b) a coarse quantile-bin edge list, and R2 requires every
# bin to be backed by >= 20 real observations. So a legitimate bin-edge
# array fit on ``n_records`` observations has at most ``n_records / 20``
# entries, which is a small fraction of ``n_records`` for any table large
# enough to be fitted at all. A verbatim/leaked array (a real column, a
# per-record residual, a copula tail sample, ...) instead scales with
# ``n_records`` directly — its length or distinct-value count sits close to
# ``n_records``. We flag an array once BOTH:
#   * it is long enough to plausibly be a leak rather than noise
#     (>= DEFAULT_LEAKAGE_MIN_LENGTH, default 20 — the same cell-count floor
#     as R2, so we never flag arrays smaller than the smallest legal cell),
#     and
#   * its length OR its distinct-value count is >= DEFAULT_LEAKAGE_FRACTION
#     (default 5%) of the table's declared ``n_records``.
# 5% is generous headroom over the ~5% (1/20) a maximally-fine legal
# quantile-bin array could reach, while still catching arrays that are an
# order of magnitude smaller than a full verbatim column.
DEFAULT_LEAKAGE_MIN_LENGTH = 20
DEFAULT_LEAKAGE_FRACTION = 0.05

_Number = int | float
_JSONPath = tuple[str, ...]


class IncompatiblePackVersionError(Exception):
    """Raised when a pack's ``pack_version`` major component is unsupported."""


class LeakageError(Exception):
    """Raised when a pack fails the row-level-key or value-level leakage scan."""


@dataclass(frozen=True)
class LeakageFinding:
    """One suspicious location found by :func:`scan_for_leakage`."""

    table: str
    path: _JSONPath
    reason: str
    length: int | None = None
    distinct_count: int | None = None
    n_records: int | None = None

    def describe(self) -> str:
        """Human-readable one-line rendering, used in error messages."""
        location = f"{self.table}.{'.'.join(self.path)}" if self.path else self.table
        if self.length is None:
            return f"{location}: {self.reason}"
        return (
            f"{location}: {self.reason} "
            f"(length={self.length}, distinct={self.distinct_count}, "
            f"n_records={self.n_records})"
        )


@dataclass
class ParamPack:
    """In-memory representation of a parameter pack directory.

    ``manifest`` and ``tables`` are plain JSON-compatible dicts so the whole
    pack is always human-inspectable (KTD-2) — no pickled Python objects.
    """

    manifest: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def write(self, pack_dir: str | Path) -> None:
        """Validate then write this pack to ``pack_dir`` (creating it)."""
        _validate_manifest(self.manifest)
        assert_no_leakage(self)

        pack_dir = Path(pack_dir)
        pack_dir.mkdir(parents=True, exist_ok=True)
        (pack_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        tables_dir = pack_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, block in self.tables.items():
            (tables_dir / f"{name}.json").write_text(
                json.dumps(block, indent=2, sort_keys=True), encoding="utf-8"
            )

    @classmethod
    def load(cls, pack_dir: str | Path) -> ParamPack:
        """Read a pack from ``pack_dir``, validating version compatibility."""
        pack_dir = Path(pack_dir)
        manifest_path = pack_dir / "manifest.json"
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest)
        _check_version_compatible(str(manifest["pack_version"]))

        tables: dict[str, dict[str, Any]] = {}
        tables_dir = pack_dir / "tables"
        if tables_dir.is_dir():
            for table_file in sorted(tables_dir.glob("*.json")):
                tables[table_file.stem] = json.loads(table_file.read_text(encoding="utf-8"))

        return cls(manifest=manifest, tables=tables)


def write_pack(
    pack_dir: str | Path,
    manifest: Mapping[str, Any],
    table_blocks: Mapping[str, Mapping[str, Any]],
) -> None:
    """Convenience wrapper: build a :class:`ParamPack` and write it."""
    pack = ParamPack(
        manifest=dict(manifest),
        tables={name: dict(block) for name, block in table_blocks.items()},
    )
    pack.write(pack_dir)


def read_pack(pack_dir: str | Path) -> ParamPack:
    """Convenience wrapper around :meth:`ParamPack.load`."""
    return ParamPack.load(pack_dir)


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = _REQUIRED_MANIFEST_KEYS - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing required key(s): {sorted(missing)}")


def _check_version_compatible(pack_version: str) -> None:
    """Same-major-version compatibility check (KTD-2)."""
    reader_major = PACK_FORMAT_VERSION.split(".")[0]
    pack_major = pack_version.split(".")[0]
    if reader_major != pack_major:
        raise IncompatiblePackVersionError(
            f"pack_version {pack_version!r} is incompatible with reader "
            f"major version {PACK_FORMAT_VERSION!r} (expected major "
            f"{reader_major!r})"
        )


def _is_numeric_array(value: object) -> bool:
    """True if ``value`` is a non-empty list of int/float (bools excluded)."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)


def _iter_leaves(node: object, path: _JSONPath) -> Iterator[tuple[_JSONPath, object]]:
    """Depth-first walk over a JSON-like structure yielding (path, leaf).

    A numeric array is itself treated as a leaf (it is not descended into),
    since its elements are the values under scrutiny for leakage.
    """
    if isinstance(node, dict):
        for key, val in node.items():
            yield from _iter_leaves(val, (*path, str(key)))
    elif isinstance(node, list) and not _is_numeric_array(node):
        for i, item in enumerate(node):
            yield from _iter_leaves(item, (*path, str(i)))
    else:
        yield path, node


def _scan_forbidden_keys(table: str, node: object, path: _JSONPath) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    if isinstance(node, dict):
        for key, val in node.items():
            if str(key).strip().lower() in FORBIDDEN_ROW_LEVEL_KEYS:
                findings.append(
                    LeakageFinding(
                        table=table,
                        path=(*path, str(key)),
                        reason=f"forbidden row-level key {key!r}",
                    )
                )
            findings.extend(_scan_forbidden_keys(table, val, (*path, str(key))))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            findings.extend(_scan_forbidden_keys(table, item, (*path, str(i))))
    return findings


def scan_for_leakage(
    pack: ParamPack,
    *,
    fraction_threshold: float = DEFAULT_LEAKAGE_FRACTION,
    min_flag_length: int = DEFAULT_LEAKAGE_MIN_LENGTH,
) -> list[LeakageFinding]:
    """Scan a pack for row-level leakage; return structured findings (R1/R2).

    Two independent checks, both run over the manifest and every table
    block:

    * key/schema: any key matching :data:`FORBIDDEN_ROW_LEVEL_KEYS` at any
      nesting depth is flagged, regardless of its value.
    * value-level: any numeric array whose length or distinct-value count
      is both >= ``min_flag_length`` and >= ``fraction_threshold`` of the
      table's declared ``n_records`` is flagged (see module docstring for
      the threshold rationale). Tables with no declared/positive
      ``n_records`` (prior-driven tables) are skipped by this check — there
      is no real record count for their arrays to "approach".

    Returns an empty list for a clean pack; never raises. Use
    :func:`assert_no_leakage` to raise :class:`LeakageError` on findings.
    """
    findings: list[LeakageFinding] = []

    findings.extend(_scan_forbidden_keys("manifest", pack.manifest, ()))

    for table, block in pack.tables.items():
        findings.extend(_scan_forbidden_keys(table, block, ()))

        n_records_raw = block.get("n_records", 0)
        n_records = int(n_records_raw) if isinstance(n_records_raw, int | float) else 0
        if n_records <= 0:
            continue

        for path, leaf in _iter_leaves(block, ()):
            if not _is_numeric_array(leaf):
                continue
            arr: Sequence[_Number] = leaf  # type: ignore[assignment]
            length = len(arr)
            if length < min_flag_length:
                continue
            distinct = len({round(float(v), 12) for v in arr})
            threshold = fraction_threshold * n_records
            if length >= threshold or distinct >= threshold:
                findings.append(
                    LeakageFinding(
                        table=table,
                        path=path,
                        reason="numeric array length/distinct-count approaches n_records",
                        length=length,
                        distinct_count=distinct,
                        n_records=n_records,
                    )
                )

    return findings


def assert_no_leakage(
    pack: ParamPack,
    *,
    fraction_threshold: float = DEFAULT_LEAKAGE_FRACTION,
    min_flag_length: int = DEFAULT_LEAKAGE_MIN_LENGTH,
) -> None:
    """Raise :class:`LeakageError` if :func:`scan_for_leakage` finds anything."""
    findings = scan_for_leakage(
        pack, fraction_threshold=fraction_threshold, min_flag_length=min_flag_length
    )
    if findings:
        rendered = "\n".join(f"  - {f.describe()}" for f in findings)
        raise LeakageError(f"leakage scan rejected pack:\n{rendered}")

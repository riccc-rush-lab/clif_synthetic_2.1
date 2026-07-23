"""Vendor the CLIF 2.1.0 data dictionary as a structured schema map (U3).

The mCIDE CSVs (vendored in U2) define only the *category* fields and numeric
outlier bounds. The complete per-table column list + data types lives in the
CLIF website's Quarto data dictionary (`data-dictionary-2.1.0.qmd`). This script
fetches that file at a pinned commit, parses each ``## Table`` section into a
``{table -> {maturity, columns:[{name, dtype}]}}`` map, and writes it to
``src/clifforge/reference/data/dictionary.json`` so ``scripts/gen_schemas.py``
can build complete pandera schemas fully offline (R24).

Two dictionary quirks are handled:
- Tables appear as GitHub pipe tables *or* Pandoc grid tables; both encode a row
  as ``| cell | cell | ...``. Concept-tier tables sometimes omit data types (a
  ``Description`` column instead of ``Data Type``) — their columns are recorded
  with an ``unknown`` dtype (the generator defaults them to string).
- ``medication_admin_intermittent`` documents itself as "the same schema as
  medication_admin_continuous"; it is recorded as an alias and resolved here.

Re-run with::

    uv run python scripts/vendor_dictionary.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

REPO = "clif-consortium/website"
# Commit that last touched data-dictionary-2.1.0.qmd (2025-07-16).
COMMIT = "888353c0521ac2ce2c380cf2ec94cedaabd36bfe"
QMD_PATH = "data-dictionary/data-dictionary-2.1.0.qmd"
CLIF_VERSION = "2.1.0"
RETRIEVED_AT = "2026-07-23"

DATA_ROOT = Path(__file__).resolve().parent.parent / "src" / "clifforge" / "reference" / "data"

KNOWN_DTYPES = {
    "VARCHAR",
    "DATETIME",
    "DOUBLE",
    "INT",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "BOOL",
    "NUMERIC",
    "BIGINT",
}

# Dictionary section header -> mCIDE/CLIF canonical table id (only where they differ).
TABLE_RENAME = {
    "microbiology_non_culture": "microbiology_nonculture",
    "procedures": "patient_procedures",
    "sensitivity": "microbiology_susceptibility",
}


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"


def _fetch_text(path: str) -> str:
    with urllib.request.urlopen(_raw_url(path), timeout=30) as resp:  # noqa: S310 (pinned host)
        return resp.read().decode("utf-8")


def _norm_table(header: str) -> str:
    # Drop a `{#anchor}` suffix and bold markers, then snake-case.
    header = re.sub(r"\{#.*?\}", "", header)
    header = header.replace("*", "").strip()
    tid = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")
    return TABLE_RENAME.get(tid, tid)


def _row_cells(line: str) -> list[str]:
    return [c.strip().strip("*` ") for c in line.strip().split("|")[1:-1]]


def _is_separator(cells: list[str]) -> bool:
    return all(set(c) <= set("-:= ") for c in cells) if cells else True


def _parse(text: str) -> dict[str, dict[str, Any]]:
    lines = text.splitlines()
    maturity: str | None = None
    tables: dict[str, dict[str, Any]] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        top = re.match(r"^#\s+\*\*(.+?)\*\*", line)
        if top:
            label = top.group(1).lower()
            if "beta" in label:
                maturity = "beta"
            elif "concept" in label:
                maturity = "concept"
            i += 1
            continue
        sec = re.match(r"^##\s+(.+?)\s*$", line)
        if sec:
            table = _norm_table(sec.group(1))
            cols, alias = _parse_section(lines, i + 1)
            entry: dict[str, Any] = {"maturity": maturity, "columns": cols}
            if alias:
                entry["alias_of"] = alias
            tables[table] = entry
        i += 1
    _resolve_aliases(tables)
    return tables


def _parse_section(lines: list[str], start: int) -> tuple[list[dict[str, str]], str | None]:
    """Parse the first schema table (and any alias note) in a section body."""
    alias: str | None = None
    # Scan the section for an alias note and the first schema-table header.
    j = start
    while j < len(lines) and not lines[j].startswith("## "):
        body = lines[j]
        if alias is None and "same schema as" in body.lower():
            ref = re.search(r"`([a-z_]+)`|\(#([a-z-]+)\)", body)
            if ref:
                alias = (ref.group(1) or ref.group(2) or "").replace("-", "_")
        cells = _row_cells(lines[j]) if lines[j].strip().startswith("|") else []
        if len(cells) >= 2 and cells[0].lower() in {"variable name", "column name"}:
            has_dtypes = cells[1].lower() in {"data type", "datatype"}
            cols = _collect_rows(lines, j + 1, has_dtypes)
            return cols, alias
        j += 1
    return [], alias


def _collect_rows(lines: list[str], start: int, has_dtypes: bool) -> list[dict[str, str]]:
    cols: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("+"):  # grid-table separator
            continue
        if not stripped.startswith("|"):
            break  # blank line or prose ends the table
        cells = _row_cells(line)
        if not cells or _is_separator(cells):
            continue
        name = cells[0]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            continue  # header echo, example row, or malformed
        if name in seen:
            continue
        if has_dtypes:
            dtype = cells[1].upper() if len(cells) > 1 else ""
            if dtype not in KNOWN_DTYPES:
                continue
        else:
            dtype = "UNKNOWN"
        seen.add(name)
        cols.append({"name": name, "dtype": dtype})
    return cols


def _resolve_aliases(tables: dict[str, dict[str, Any]]) -> None:
    for entry in tables.values():
        target = entry.get("alias_of")
        if target and not entry["columns"] and target in tables:
            entry["columns"] = [dict(c) for c in tables[target]["columns"]]


def main() -> None:
    text = _fetch_text(QMD_PATH)
    tables = _parse(text)
    non_empty = {t: v for t, v in tables.items() if v["columns"]}

    payload = {
        "clif_version": CLIF_VERSION,
        "source_repo": f"https://github.com/{REPO}",
        "source_commit": COMMIT,
        "source_path": QMD_PATH,
        "retrieved_at": RETRIEVED_AT,
        "note": (
            "Parsed from the CLIF 2.1.0 Quarto data dictionary. Column data types "
            "recorded as UNKNOWN come from Concept-tier tables the dictionary "
            "documents without a Data Type column; the schema generator defaults "
            "them to string."
        ),
        "tables": dict(sorted(tables.items())),
    }
    out = DATA_ROOT / "dictionary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_cols = sum(len(v["columns"]) for v in tables.values())
    print(
        f"Wrote {out.name}: {len(tables)} tables "
        f"({len(non_empty)} with columns), {total_cols} columns total."
    )
    for t, v in sorted(tables.items()):
        flag = "" if v["columns"] else "  <-- EMPTY"
        print(f"  {v['maturity'] or '?':7} {t:30} {len(v['columns']):2} cols{flag}")


if __name__ == "__main__":
    main()

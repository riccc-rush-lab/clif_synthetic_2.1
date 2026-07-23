"""Render a per-pack ``PROVENANCE.md`` from a parameter-pack manifest (R4, R4a).

``PROVENANCE.md`` is the technical/methods-level disclosure that keeps
CLIFForge's public "synthetic generator" framing honest (R4a): it names,
per table, whether the emitted parameters were fitted from real CLIF-MIMIC
or are prior-driven (consortium rules / literature rates), records the
mCIDE/outlier reference source (URL + spec-repo commit + retrieval date),
and carries the CLIF-MIMIC citation obligation for the methods
documentation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Default citation used when the manifest does not supply its own
# ``citation`` string. This intentionally cites the two facts that are
# stably verifiable rather than inventing a specific "CLIF-MIMIC" author
# list: the MIMIC-IV dataset paper, and the CLIF specification repository
# whose pinned commit (recorded separately under ``reference_source``)
# is the authoritative source for the exact CLIF-MIMIC derivation used.
DEFAULT_CITATION = (
    "Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely "
    "accessible electronic health record dataset. Sci Data 10, 1 (2023). "
    "https://doi.org/10.1038/s41597-022-01899-x — derived per the Common "
    "Longitudinal ICU data Format (CLIF) specification, "
    "github.com/Common-Longitudinal-ICU-data-Format/CLIF (see "
    "`reference_source` above for the exact pinned commit and retrieval "
    "date used by this pack)."
)


def write_provenance(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Render ``manifest`` to a ``PROVENANCE.md`` file at ``path``."""
    lines: list[str] = ["# PROVENANCE", ""]

    lines.append(f"- **Pack version:** {manifest.get('pack_version', 'unknown')}")
    lines.append(f"- **CLIF version:** {manifest.get('clif_version', 'unknown')}")

    fit_source: Mapping[str, Any] = manifest.get("fit_source", {})
    dataset_id = fit_source.get("dataset_id", "unknown")
    commit = fit_source.get("commit", "unknown")
    lines.append(f"- **Fit source:** {dataset_id} @ commit `{commit}`")
    lines.append("")

    reference_source: Mapping[str, Any] = manifest.get("reference_source", {})
    if reference_source:
        lines.append("## Reference data (mCIDE / outlier) source")
        lines.append("")
        for key in ("mcide_url", "mcide_commit", "outlier_url", "retrieved_date"):
            if key in reference_source:
                lines.append(f"- **{key}:** {reference_source[key]}")
        lines.append("")

    lines.append("## Tables: fitted vs prior-driven")
    lines.append("")
    lines.append("| table | status | source |")
    lines.append("|---|---|---|")
    tables: Mapping[str, Any] = manifest.get("tables", {})
    for name in sorted(tables):
        info = tables[name]
        status = "fitted" if info.get("fitted") else "prior-driven"
        source = info.get("source", "")
        lines.append(f"| {name} | {status} | {source} |")
    lines.append("")

    lines.append("## Suppression audit (n >= 20 cell gate)")
    lines.append("")
    suppression_audit: Mapping[str, Any] = manifest.get("suppression_audit", {})
    if suppression_audit:
        lines.append("| table | detail |")
        lines.append("|---|---|")
        for name in sorted(suppression_audit):
            lines.append(f"| {name} | {suppression_audit[name]} |")
    else:
        lines.append("_No suppression events recorded._")
    lines.append("")

    lines.append("## Citation")
    lines.append("")
    lines.append(str(manifest.get("citation", DEFAULT_CITATION)))
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")

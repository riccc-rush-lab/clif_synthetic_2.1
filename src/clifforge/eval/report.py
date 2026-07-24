"""Evaluation report assembly (U25; R26).

Rolls the four evaluation surfaces into one Markdown report:

1. **Dataset** — tables and row counts.
2. **Validation** — both conformance gates (pandera primary, clifpy secondary)
   per table. Always computable: it needs only the synthetic dataset.
3. **Fidelity** (U24) — SDMetrics column-shape / column-pair quality per table.
4. **Privacy** (U23) — DCR, NN-distance ratio, identifiability.
5. **Utility** (U22) — TSTR vs TRTR AUC and the utility gap.

Sections 3-5 are *comparative*: they need a reference dataset. When none is
supplied they are still rendered, but explicitly marked as not computed rather
than silently omitted or filled with a placebo number. ``reference_label``
records **what** the comparison was against, so a report can never imply
real-data fidelity when the reference was in fact another synthetic draw — the
distinction that keeps the report honest for a public artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import polars as pl

from clifforge.conformance import gate

__all__ = ["build_report"]

_NOT_COMPUTED = (
    "_Not computed — this section requires a reference dataset. "
    "Re-run with a credentialed real CLIF reference to populate it._"
)


def _dataset_section(tables: Mapping[str, pl.DataFrame]) -> str:
    rows = "\n".join(
        f"| `{name}` | {tables[name].height:,} | {len(tables[name].columns)} |"
        for name in sorted(tables)
    )
    total = sum(t.height for t in tables.values())
    return (
        "## 1. Dataset\n\n"
        f"{len(tables)} tables, {total:,} rows total.\n\n"
        "| Table | Rows | Columns |\n|---|---:|---:|\n"
        f"{rows}\n"
    )


def _validation_section(tables: Mapping[str, pl.DataFrame], *, run_secondary: bool) -> str:
    lines = []
    all_passed = True
    for name in sorted(tables):
        try:
            report = gate.validate(tables[name], name, run_secondary=run_secondary)
        except gate.ConformanceError as exc:
            all_passed = False
            lines.append(f"| `{name}` | FAIL | - | {str(exc)[:80]} |")
            continue
        lines.append(f"| `{name}` | pass | {report.clifpy_status} | {report.clifpy_note[:80]} |")
    verdict = (
        "All tables pass the primary (pandera) conformance gate."
        if all_passed
        else "**One or more tables FAILED the primary conformance gate.**"
    )
    return (
        "## 2. Validation\n\n"
        f"{verdict}\n\n"
        "| Table | Primary (pandera) | Secondary (clifpy) | Note |\n|---|---|---|---|\n"
        + "\n".join(lines)
        + "\n"
    )


def _fidelity_section(
    synthetic: Mapping[str, pl.DataFrame], reference: Mapping[str, pl.DataFrame] | None
) -> str:
    if reference is None:
        return f"## 3. Fidelity\n\n{_NOT_COMPUTED}\n"
    from clifforge.eval.fidelity import fidelity_report

    scored = fidelity_report(synthetic, reference)
    if not scored:
        return "## 3. Fidelity\n\n_No comparable tables._\n"
    rows = []
    for name, r in sorted(scored.items(), key=lambda kv: kv[1].quality_score):
        pair = f"{r.column_pair_score:.3f}" if r.column_pair_score is not None else "-"
        rows.append(
            f"| `{name}` | {r.quality_score:.3f} | {r.column_shape_score:.3f} | "
            f"{pair} | {r.n_columns} | {r.n_pairs} |"
        )
    mean_q = sum(r.quality_score for r in scored.values()) / len(scored)
    return (
        "## 3. Fidelity\n\n"
        f"SDMetrics column-shape and column-pair similarity (1.0 = identical distribution). "
        f"Mean quality across {len(scored)} tables: **{mean_q:.3f}**.\n\n"
        "| Table | Quality | Column shapes | Column pairs | Cols | Pairs |\n"
        "|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n"
    )


def _privacy_section(
    synthetic: Mapping[str, pl.DataFrame], reference: Mapping[str, pl.DataFrame] | None
) -> str:
    if reference is None:
        return f"## 4. Privacy\n\n{_NOT_COMPUTED}\n"
    from clifforge.eval.privacy import privacy_metrics

    p = privacy_metrics(synthetic, reference)
    memorization = "none detected" if p.dcr_p5 > 0 else "**possible memorization**"
    return (
        "## 4. Privacy\n\n"
        "| Metric | Value | Reading |\n|---|---:|---|\n"
        f"| DCR (median) | {p.dcr_median:.3f} | distance to the closest reference record |\n"
        f"| DCR (5th pct) | {p.dcr_p5:.3f} | closest-match tail — {memorization} |\n"
        f"| NN-distance ratio (median) | {p.nndr_median:.3f} | near 1.0 = not "
        "singling out one record |\n"
        f"| Identifiability | {p.identifiability:.3f} | fraction of reference records closer to a "
        "synthetic than to another reference record |\n"
    )


def _utility_section(
    synthetic: Mapping[str, pl.DataFrame],
    reference: Mapping[str, pl.DataFrame] | None,
    seed: int,
) -> str:
    if reference is None:
        return f"## 5. Utility (TSTR)\n\n{_NOT_COMPUTED}\n"
    from clifforge.eval.tstr import run_tstr

    try:
        t = run_tstr(synthetic, reference, seed=seed)
    except ValueError as exc:
        return f"## 5. Utility (TSTR)\n\n_Not computed: {exc}_\n"
    return (
        "## 5. Utility (TSTR)\n\n"
        "In-hospital mortality, LightGBM, both models scored on the same reference test split.\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| TSTR AUC (trained on synthetic) | {t.tstr_auc:.3f} |\n"
        f"| TRTR AUC (trained on reference) | {t.trtr_auc:.3f} |\n"
        f"| Utility gap (TRTR - TSTR) | {t.auc_gap:+.3f} |\n"
        f"| Features / test rows | {t.n_features} / {t.n_test_real} |\n"
    )


def build_report(
    synthetic: Mapping[str, pl.DataFrame],
    *,
    reference: Mapping[str, pl.DataFrame] | None = None,
    reference_label: str | None = None,
    title: str = "CLIFForge evaluation report",
    generated_at: str | None = None,
    seed: int = 0,
    run_secondary: bool = True,
) -> str:
    """Assemble the four evaluation surfaces into one Markdown report (R26).

    ``reference`` drives the comparative sections; ``reference_label`` states what
    it was, so synthetic-vs-synthetic self-consistency can never read as
    real-data fidelity.
    """
    stamp = generated_at or datetime.now().astimezone().strftime("%Y-%m-%d")
    if reference is None:
        ref_line = (
            "**Reference:** none supplied — comparative sections (fidelity, privacy, "
            "utility) are not computed."
        )
    else:
        ref_line = f"**Reference:** {reference_label or 'unlabelled reference dataset'}"

    return "\n".join(
        [
            f"# {title}",
            "",
            f"Generated {stamp}. All data is fully synthetic — sampled from an aggregate "
            "parameter pack, never copied from a real record.",
            "",
            ref_line,
            "",
            _dataset_section(synthetic),
            _validation_section(synthetic, run_secondary=run_secondary),
            _fidelity_section(synthetic, reference),
            _privacy_section(synthetic, reference),
            _utility_section(synthetic, reference, seed),
        ]
    )

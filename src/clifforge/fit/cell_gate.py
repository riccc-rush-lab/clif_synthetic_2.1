"""The n >= 20 cell-count gate (R2).

``suppress`` is the single choke point every fit-stage estimator (U5) routes
through: a fitted cell (a stratum, a bin, a transition pair, ...) is only
emitted into the parameter pack if it is backed by at least ``min_n`` real
observations. Sub-threshold cells are never emitted verbatim — they either
fall back to a caller-supplied prior, fall back to a coarser aggregate cell
(if one is available and itself meets the gate), or are dropped entirely.
Every decision is recorded in an audit trail so ``PROVENANCE.md`` can report
exactly how many cells were suppressed and to what fallback (R4).

Pure and deterministic: no randomness, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar

CellKeyT = TypeVar("CellKeyT", bound=Hashable)
ParamT = TypeVar("ParamT")

FallbackKind = Literal["prior", "coarser_aggregate", "none"]


@dataclass(frozen=True)
class SuppressionRecord:
    """One suppression decision, for the pack's ``suppression_audit``."""

    cell: Hashable
    n: int
    fallback_kind: FallbackKind
    fallback_source: Hashable | None = None


def suppress(
    counts: Mapping[CellKeyT, int],
    params: Mapping[CellKeyT, ParamT],
    *,
    min_n: int = 20,
    prior: Mapping[CellKeyT, ParamT] | None = None,
    coarsen: Callable[[CellKeyT], CellKeyT] | None = None,
) -> tuple[dict[CellKeyT, ParamT], list[SuppressionRecord]]:
    """Drop sub-threshold cells, recording the fallback used for each (R2).

    Args:
        counts: real observation count per cell key.
        params: fitted parameter value per cell key (same key space as
            ``counts``; a cell missing from ``counts`` is treated as n=0).
        min_n: the minimum real-observation count a cell must have to be
            emitted as-fitted (default 20, per R2).
        prior: an optional prior parameter value per cell key, used as the
            fallback when a coarser aggregate is unavailable or itself
            sub-threshold.
        coarsen: an optional function mapping a cell key to its coarser
            aggregate key (e.g. a fine stratum -> its parent stratum). If
            the coarser cell exists in both ``counts`` and ``params`` and
            itself meets ``min_n``, its value is used as the fallback in
            preference to ``prior``.

    Returns:
        A tuple of (surviving params — one entry per cell that is either
        fitted-as-is or has a fallback, i.e. every sub-threshold cell with
        no available fallback is simply absent — and the full audit list,
        which includes an entry for every sub-threshold cell, including
        those with no fallback recorded as ``fallback_kind="none"``).
    """
    prior = prior or {}
    surviving: dict[CellKeyT, ParamT] = {}
    audit: list[SuppressionRecord] = []

    for cell, value in params.items():
        n = counts.get(cell, 0)
        if n >= min_n:
            surviving[cell] = value
            continue

        if coarsen is not None:
            coarser_cell = coarsen(cell)
            coarser_n = counts.get(coarser_cell, 0)
            if coarser_cell in params and coarser_n >= min_n:
                surviving[cell] = params[coarser_cell]
                audit.append(
                    SuppressionRecord(
                        cell=cell,
                        n=n,
                        fallback_kind="coarser_aggregate",
                        fallback_source=coarser_cell,
                    )
                )
                continue

        if cell in prior:
            surviving[cell] = prior[cell]
            audit.append(
                SuppressionRecord(cell=cell, n=n, fallback_kind="prior", fallback_source=cell)
            )
            continue

        # No fallback available: the cell is not emitted at all (R2).
        audit.append(
            SuppressionRecord(cell=cell, n=n, fallback_kind="none", fallback_source=None)
        )

    return surviving, audit

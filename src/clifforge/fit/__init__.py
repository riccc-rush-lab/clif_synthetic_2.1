"""Fit-stage package: parameter-pack I/O, cell-count gating, and provenance.

Everything under ``clifforge.fit`` supports the one-time fit stage (U5) and
the aggregate-only parameter pack it emits (KTD-2). Nothing here reads or
writes row-level real data directly except the estimators module (U5) —
``param_pack`` and ``cell_gate`` operate purely on already-aggregated
counts/params so they can be exercised in tests without any real CLIF data
(R1).
"""

from __future__ import annotations

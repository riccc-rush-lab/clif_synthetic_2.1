"""CLIF 2.1.0 conformance harness (U3).

Two gates: :func:`clifforge.conformance.gate.validate` (pandera primary, hard;
clifpy secondary, advisory) plus the dev-only ``synthetic_clif`` schema-agreement
cross-check in :mod:`clifforge.conformance.cross_check` (R18).
"""

from __future__ import annotations

from clifforge.conformance.gate import ConformanceError, GateReport, validate

__all__ = ["ConformanceError", "GateReport", "validate"]

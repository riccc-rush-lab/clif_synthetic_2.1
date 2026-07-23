"""Per-table CLIF generators (Tier 1+).

Each module turns the parameter pack — and, for per-encounter clinical tables,
the latent state spine — into one conformant CLIF 2.1.0 table. Generators read
only ``(spine, param_pack, rng)`` and never another table's output (KTD-6); the
U21 orchestrator threads a single seeded generator through them in tier order.

No real data is present here; everything is sampled offline from the pack.
"""

from __future__ import annotations

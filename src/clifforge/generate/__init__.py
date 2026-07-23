"""Generate-stage package: offline sampling from an aggregate parameter pack.

Everything under ``clifforge.generate`` runs with **no real data present** — it
consumes only the versioned parameter pack the fit stage (U5) emits and a seeded
``numpy.random.Generator``. The latent state spine (U6) is sampled first and is
the sole cross-table channel (KTD-6); every table generator then reads
``(spine, param_pack, rng)`` and never another table's output.
"""

from __future__ import annotations

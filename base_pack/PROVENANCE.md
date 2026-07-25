# Shareable base pack — provenance

A **fully aggregate** CLIF parameter pack: category marginals, semi-Markov
support-level transitions and sojourns, per-acuity AR(1) vital physiology, lab
co-measurement correlations, med marginals, and organ-failure prevalences. It
contains **no row-level records** and no per-patient values.

Others derive CLIF-like variant datasets from this pack with
`clif-forge generate --preset <name>` or `--spec <file>` — no real data and no
credential required. Applying `recalibrate_to_network_median` (+ variant
overrides) transforms it into network-median CLIF-like datasets (this is the same
transform that builds the shared master dataset).

**Compliance.** These are aggregate statistics only (no PHI). Release of this pack
is covered by the project's recorded compliance determination (see the release
gate / `COMPLIANCE_ACK.md`), the same determination that clears the synthetic
datasets it seeds.

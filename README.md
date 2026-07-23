# CLIFForge

**A fully synthetic CLIF 2.1 ICU dataset generator.**

CLIFForge produces ICU datasets in exact [CLIF 2.1](https://clif-consortium.github.io/website/)
format that are **openly redistributable** and **clinically coherent** — built
for the uses a credentialed real dataset cannot serve: public ETL smoke-testing,
CI fixtures, agent development, teaching, and demos.

It is an expansion of the consortium's `synthetic_clif`. Where `synthetic_clif`
generates all 28 tables from hand-specified priors, CLIFForge's differentiator is
**empirical fidelity**: it fits its distributions, couplings, and trajectories to
aggregate ICU statistics so its output matches real CLIF closely enough to train
models against.

## How it stays synthetic

- **Aggregate-only fit-then-sample.** A one-time fit stage emits a versioned
  *parameter pack* — marginals, state-transition distributions, per-state
  physiology parameters, lab correlations, infusion hazards. **No row-level
  record ever leaves the fit stage**, and every fitted parameter is gated on a
  minimum cell count (n ≥ 20).
- **Offline generation.** The `generate` stage samples entirely from the
  parameter pack, with no real data present.
- **Latent state spine.** Each synthetic hospitalization has one internal
  trajectory of acuity, organ-failure flags, and outcome; every table reads from
  that spine, never from its siblings — which is what keeps vasopressors paired
  with hypotension, sedation with mechanical ventilation, prone with severe
  hypoxemia.

## Usage

```bash
# Generate a synthetic dataset (offline, no real data required)
uv run clif-forge generate --n-patients 1000 --seed 42 --out ./output/
```

A single `--seed` reproduces byte-identical output.

## Status

Early scaffold. See `docs/plans/` for the implementation plan. The fit stage
requires a staged real CLIF-MIMIC set and is not part of the public
distribution.

## Provenance & licensing

CLIFForge learns *how a realistic CLIF table is shaped* from aggregate,
non-derivable statistics. That learned-parameter provenance — including the
CLIF-MIMIC citation and the exact mCIDE snapshot — is documented in
`PROVENANCE.md` at the technical/methods level. All runtime dependencies are
permissive (MIT / BSD / Apache-2.0).

*Not medical or legal advice. Redistribution of any parameter pack derived from
credentialed data is gated on the appropriate data-use acknowledgment.*

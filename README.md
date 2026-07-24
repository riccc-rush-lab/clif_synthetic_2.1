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
uv run clif-forge generate --n-patients 1000 --seed 42 --out ./output/ --pack ./data/param_packs/mimic
```

A single `--seed` reproduces byte-identical output. Every table is run through
the conformance gate before anything is written; any validation failure exits
nonzero and writes nothing.

Output is one `clif_<table>.parquet` per table, plus `clif_truth.parquet` — the
latent acuity spine behind each encounter, which makes the dataset usable as a
benchmark with free ground-truth labels.

## Demo dataset

[`demo_output/`](demo_output/) holds a committed **n=100, seed 42** dataset (19
tables) so you can inspect real output without running anything or holding any
credential. It ships with a generated
[`REPORT.md`](demo_output/REPORT.md) and [`PROVENANCE.md`](demo_output/PROVENANCE.md).

## Evaluation

Three evaluation surfaces live under `clifforge.eval` (install the `eval` extra):

- **Utility** — train-on-synthetic / test-on-real mortality AUC and the utility
  gap vs a real-trained baseline, with a leakage guard that recomputes each test
  patient's partition from the pack's split spec and fails if any was used for fitting.
- **Privacy** — distance to closest record, NN-distance ratio, and
  identifiability. Computed standalone (no torch/synthcity dependency).
- **Fidelity** — SDMetrics column-shape and column-pair similarity per table.

`clifforge.eval.report.build_report` rolls all of them plus both conformance
gates into a Markdown report. Comparative sections require a reference dataset;
when none is supplied they are marked *not computed* rather than filled with a
placebo number, and the report always records **what** the reference was.

## Status

The fit and generate stages and all three evaluation surfaces are implemented;
all 19 tables generate and pass conformance. The fit stage requires a staged real
CLIF-MIMIC set and is **not** part of the public distribution — generation needs
only the parameter pack. See `docs/plans/` for the implementation plan.

## Provenance & licensing

CLIFForge learns *how a realistic CLIF table is shaped* from aggregate,
non-derivable statistics. That learned-parameter provenance — including the
CLIF-MIMIC citation and the exact mCIDE snapshot — is documented in
`PROVENANCE.md` at the technical/methods level. All runtime dependencies are
permissive (MIT / BSD / Apache-2.0).

### Release gate

The parameter pack is fitted over MIMIC-IV-Ext-CLIF, governed by a PhysioNet
**credentialed** data use agreement. Fitting locally to an aggregate pack is
permitted; *publishing* a derived artifact is not, until a human records the
PhysioNet/MIMIC and Rush compliance review.

`scripts/release_gate.py` enforces this mechanically rather than by memory — it
exits nonzero unless a completed `COMPLIANCE_ACK.md` exists (see
[`COMPLIANCE_ACK.template.md`](COMPLIANCE_ACK.template.md)). Wire it into CI on
release/tag events:

```bash
uv run python scripts/release_gate.py   # exit 1 until the acknowledgment is recorded
```

*Not medical or legal advice. Redistribution of any parameter pack derived from
credentialed data is gated on the appropriate data-use acknowledgment.*

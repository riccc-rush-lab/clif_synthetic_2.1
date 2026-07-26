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
# Generate from the built-in demo pack — works on a fresh clone, no real data,
# no fit, no credential required.
uv run clif-forge generate --n-patients 1000 --seed 42 --demo --out ./output/

# Or generate from a pack you fitted to your own real CLIF data (see `fit` below).
uv run clif-forge generate --n-patients 1000 --seed 42 --pack ./my-pack --out ./output/
```

`--demo` uses a complete, hand-specified synthetic pack shipped in the package
(`clifforge.demo`): it exercises every table and coupling and passes the
conformance gate, so it is ideal for pipeline tests, CI fixtures, and agent/tool
development — but it is **structural, not statistically calibrated**. Realistic,
network-representative output requires fitting a pack to real CLIF statistics
(`clif-forge fit`, which needs your own staged real CLIF data).

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

## Ideate your own CLIF-like dataset

The shipped network-median dataset is the **master** — an off-the-shelf base
everyone builds on. Anyone can *ideate and create* their own **derivative** —
always distinct, always CLIF 2.1-conformant — from the committed **shareable base
pack** ([`base_pack/`](base_pack/), aggregate-only, no real data, no credential),
on three axes: **size**, **demographics**, and **illness rates**.

**Presets** — start from a shipped example variant, tweak, generate:

```bash
uv run clif-forge generate --preset high-acuity --n-patients 5000 --out ./my-dataset
```

Shipped presets: `high-acuity`, `older-cohort`, `sepsis-heavy` (see [`presets/`](presets/)).

**Your own spec** — a variant is a small TOML recipe (every field defaults to the
master; a minimal spec reproduces it):

```toml
# my-variant.toml
name = "my-icu"
n = 20000

[demographics]
age_shift = 5.0        # relative to the base pack
hispanic_frac = 0.45

[rates]
imv = 0.55             # reaches-invasive-ventilation target
mortality_scale = 1.4  # multiplier on peak mortality
vaso_frac = 0.45       # cardiovascular-failure (vasopressor) rate
crrt_prob = 0.29
prone_severe = 0.03
```

```bash
uv run clif-forge generate --spec my-variant.toml --out ./my-dataset
```

Every generated dataset writes a `manifest.json` recording the resolved spec, seed,
generator version, and per-table content hashes — so a variant is reproducible from
its recipe, and any two variants are provably distinct. The same knobs are available
on the Python API (`clifforge.variants.spec_to_pack`,
`clifforge.generate.recalibrate.recalibrate_to_network_median`), which operate on a
deep copy and never mutate the base pack, so one base pack seeds unlimited variants.

To re-author the base pack itself (or fit your own site's data), see
`scripts/build_base_pack.py` and the `fit` subcommand.

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

### Evaluating against real data

The committed demo report's comparative numbers use a second synthetic draw, so
they measure **generator self-consistency**, not real-data fidelity. Producing the
real numbers needs a staged real CLIF reference and, for a CLIF-MIMIC reference,
the patient-disjoint holdout split the fit stage reserved.

One precondition deserves emphasis. The leakage guard decides a patient's
partition by hashing the `patient_id` **string**, so it is only meaningful when
those ids are the same namespace the pack was fit on. The pack deliberately
stores no identifiers, so this cannot be checked automatically — and a re-hashed
or re-identified reference does not error, it returns a *confidently wrong*
answer: hashing unrelated ids yields a holdout fraction that simply matches the
configured fraction, which is indistinguishable from a correct split. For that
reason `assert_holdout_disjoint` refuses to run until the caller passes
`ids_share_fit_namespace=True`, affirming the one thing only they can verify.

### Validation report

A [synthetic-vs-real validation report](https://claude.ai/code/artifact/d72f6a8d-b209-469d-82ad-b99d2b9c3cc1)
compares a generated master dataset against a real staged CLIF reference across
missingness, length-of-stay, life-support rates, mortality, and the longitudinal
illness trajectory (MAP falling and creatinine rising toward death). It documents
what matches by design and what is intentionally different (demographics, age,
network-median mortality).

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

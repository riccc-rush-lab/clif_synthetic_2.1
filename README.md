# CLIFForge

**A fully synthetic CLIF 2.1 dataset generator.**

CLIFForge produces datasets in exact [CLIF 2.1](https://clif-consortium.github.io/website/)
format that are **openly redistributable** and **clinically coherent** — built
for the uses a credentialed real dataset cannot serve: public ETL smoke-testing,
CI fixtures, agent development, teaching, and demos. It generates two population
shapes off the same engine:

- an **ICU cohort** (every stay an ICU stay), and
- a **whole-hospital population** (ward / ED / stepdown / ICU mix with realistic
  patient flow — most patients never leave the ward, ~15% reach the ICU).

It is an expansion of the consortium's `synthetic_clif`. Where `synthetic_clif`
generates all 28 tables from hand-specified priors, CLIFForge's differentiator is
**empirical fidelity**: it fits its distributions, couplings, and trajectories to
aggregate CLIF statistics so its output lands in the real statistical region —
close enough to train models against — while being provably synthetic (no real
record leaves the fit stage; no synthetic record traces back to a real patient).

## Quickstart (for CLIF researchers)

**Want realistic synthetic CLIF data of your own?** Install and generate — no real
data, no credential, no fit. The shareable base pack (calibrated to real CLIF)
ships inside the package, so the output lands in the real statistical region:

```bash
pip install git+https://github.com/riccc-rush-lab/clif_synthetic_2.1.git

# an ICU cohort (network-median defaults) —
clif-forge generate --preset high-acuity --n-patients 5000 --out ./my-icu

# a whole-hospital population (2-line recipe) —
printf 'name = "my-hospital"\nmode = "full_hospital"\n' > hospital.toml
clif-forge generate --spec hospital.toml --n-patients 20000 --out ./my-hospital
```

Output is one `clif_<table>.parquet` per CLIF 2.1 table — load it with your usual
CLIF tooling. Tune any of the [rules](#each-modes-default-rules--and-how-to-change-them)
(size, demographics, illness rates, population shape) via a TOML spec.

**Want the ready-made datasets to inspect first?** Clone the repo — two off-the-shelf
samples are committed (`sample_dataset/` ICU, `sample_full_hospital/` whole-hospital),
and the full-size masters regenerate from `base_pack/` on demand:

```bash
git clone https://github.com/riccc-rush-lab/clif_synthetic_2.1.git
```

**Is it clinically believable?** See the
[synthetic-vs-real validation report](https://claude.ai/code/artifact/d72f6a8d-b209-469d-82ad-b99d2b9c3cc1)
and [Data available off the shelf](#data-available-off-the-shelf). Validate your own
output against a real CLIF reference with `scripts/validate_against_real.py`.

## How it stays synthetic

- **Aggregate-only fit-then-sample.** A one-time fit stage emits a versioned
  *parameter pack* — marginals, state-transition distributions, per-state
  physiology parameters, lab correlations, infusion hazards. **No row-level
  record ever leaves the fit stage**, and every fitted parameter is gated on a
  minimum cell count (n ≥ 20).
- **Offline generation.** The `generate` stage samples entirely from the
  parameter pack, with no real data present.
- **Latent state spine.** Each synthetic hospitalization has one internal
  trajectory of acuity, organ-failure flags, outcome, and admission route; every
  table reads from that spine, never from its siblings — which is what keeps
  vasopressors paired with hypotension, sedation with mechanical ventilation,
  prone with severe hypoxemia, and a stay's admission type agreeing with where it
  arrives.

### What makes it look real

- **Empirical value shapes.** Labs use an empirical inverse-CDF (quantile) marginal
  driven through a Gaussian copula, so skewed and long-tailed distributions match
  real (creatinine's kidney-disease tail, lactate's skew) — not just their means.
- **Real missingness.** Per-stay lab presence is conditioned on the ICU cohort and
  carries a fitted presence-correlation, so co-ordered panels (a metabolic panel, an
  arterial blood gas) are measured together rather than independently.
- **Real trajectories.** Length-of-stay distributions (median *and* tails), organ
  support, and a deterioration-toward-death course (falling blood pressure, rising
  creatinine) track the real cohort; the full-hospital population adds realistic
  patient flow (ER/OR arrivals, ward→ICU and outside-hospital→ICU transfers).
- **Privacy by construction.** Because generation samples from aggregate parameters
  and never copies a record, no synthetic patient traces back to a real one
  (verified by distance-to-closest-record, NN-distance ratio, and identifiability
  in `clifforge.eval.privacy`).

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

**The id-type rule (hardcoded, applied to every dataset):** `patient_id`,
`hospitalization_id`, and `hospitalization_joined_id` are always emitted as
**integers** (not the CLIF dictionary's nominal VARCHAR), so they load as numbers —
no leading zeros, no string coercion — in Python, R, and Stata, the way analysts
join on them. Every other id column (`device_id`, `provider_id`, `med_order_id`,
`culture_id`, `hospital_id`) is always a **string**. `patient_id` sits in a disjoint
range so it never collides with `hospitalization_id`. The rule lives in one place —
`clifforge.generate._common.enforce_numeric_ids` — and runs on every generated
table, so it is uniform across every dataset you get or make.

## System requirements

Generation is CPU + RAM bound (**no GPU**) and runs on an ordinary laptop. Two
flags let you trade a smaller footprint for a little more wall-clock, so **any
size cohort runs on any machine** — a bigger cohort just takes longer, not more
memory:

- **`--chunk-size N`** — the **RAM dial**. Cohorts larger than `N` are generated in
  bounded-memory batches streamed to disk, so peak memory tracks the *chunk*, not
  the cohort size. Output is identical regardless of the value. Measured peak RAM
  (ICU cohort, the heaviest): **~1.9 GB at `--chunk-size 2000`**, **~5.3 GB at the
  default `10000`**. Whole-hospital stays are lighter (shorter LOS → fewer rows per
  encounter), so they use less.
- **`--max-threads N`** — the **compute dial**. Caps CPU threads so generation
  doesn't claim every core on a shared or low-core machine (also settable via the
  `POLARS_MAX_THREADS` environment variable).

| Machine | Suggested flags |
|---|---|
| 8 GB RAM / few cores | `--chunk-size 2000 --max-threads 4` (~2 GB peak) |
| 16 GB+ | defaults are fine |
| Generating a large master | any `--n-patients`; peak RAM stays at the chunk footprint |

As a rough guide, a full 365k whole-hospital population takes ~20–30 min on a
modern multi-core machine; smaller cohorts scale down proportionally. Disk: the
ICU cohort is ~13 KB/encounter, the whole-hospital population ~3.5 KB/encounter.

## Data available off the shelf

Everything below is **fully synthetic, CLIF 2.1-conformant, and committed to the
repo** — clone and inspect, no generation and no credential required. All of it
is regenerable byte-for-byte from the committed `base_pack/` + recipe.

| Location | Size | What it is |
|---|---|---|
| [`sample_dataset/`](sample_dataset/) | ~10k stays, ~123 MB | **ICU cohort** sample — a representative draw of the network-median ICU master |
| [`sample_full_hospital/`](sample_full_hospital/) | ~8k stays, ~28 MB | **Whole-hospital population** sample — ward/ED/stepdown/ICU with realistic patient flow |
| [`demo_output/`](demo_output/) | n=100 | Tiny hand-specified demo (19 tables) with a generated `REPORT.md` + `PROVENANCE.md` |
| [`base_pack/`](base_pack/) | ~84 KB | The **aggregate parameter pack** — no real data, seeds every dataset above and any you generate |

Each dataset carries a `manifest.json` recording the recipe, seed, generator
version, and per-table SHA-256 content hashes.

**Full-size masters.** The complete datasets — an **85k ICU cohort** and a **365k
whole-hospital population** — are too large to commit here. Download them directly:

<!-- Replace the line below with the shared download link (e.g. Dropbox). -->
> 📥 **Full datasets:** _link coming soon_

They are also reproducible from `base_pack/` on demand (see below), so the shared
files and a local regeneration match by content hash.

**How the shipped data was generated.** The committed samples and the reference
masters were produced by `scripts/generate_deliverable.py` from the aggregate
`base_pack/` (fitted once to real CLIF; no real data is present at generation) on
a **Mac Studio (Apple M4 Max, 64 GB unified memory), Python 3.12, polars**. A 365k
whole-hospital population took ~25 minutes; the committed samples reproduce
byte-for-byte on any machine (see [System requirements](#system-requirements)).
The **method** is empirical-fidelity fit-then-sample — see
[How it stays synthetic](#how-it-stays-synthetic) and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full pipeline.

## Ideate your own CLIF-like dataset

The shipped datasets are **masters** — off-the-shelf bases everyone builds on.
Anyone can *ideate and create* their own **derivative** — always distinct, always
CLIF 2.1-conformant — from the committed **shareable base pack**
([`base_pack/`](base_pack/), aggregate-only, no real data, no credential). No two
recipes produce the same dataset, and every one is reproducible from its recipe.

**Presets** — start from a shipped example variant, tweak, generate:

```bash
uv run clif-forge generate --preset high-acuity --n-patients 5000 --out ./my-dataset
```

Shipped presets: `high-acuity`, `older-cohort`, `sepsis-heavy` (see [`presets/`](presets/)).

**Your own spec** — a variant is a small TOML recipe (every field defaults to the
master; a minimal spec reproduces it). The `mode` picks the population shape:

```toml
# my-variant.toml
name = "my-cohort"
mode = "icu"            # "icu" (default) or "full_hospital" (ward/ED/ICU mix)
n = 20000

[demographics]
age_shift = 5.0        # years, relative to the base pack
hispanic_frac = 0.45
# race_target = { ... } # optional exact race mix

[rates]                 # ICU-mode illness knobs (ignored in full_hospital mode)
imv = 0.55             # reaches-invasive-ventilation target
mortality_scale = 1.4  # multiplier on peak mortality
vaso_frac = 0.45       # cardiovascular-failure (vasopressor) rate
crrt_prob = 0.29       # CRRT fraction among renal-failure stays
prone_severe = 0.03    # prone positioning in severe hypoxemia
```

```bash
uv run clif-forge generate --spec my-variant.toml --out ./my-dataset
# or the full-hospital population:
uv run clif-forge generate --spec my-variant.toml --n-patients 50000 --out ./my-hospital
```

### Each mode's default "rules" — and how to change them

Every generated dataset (each **batch**) is produced by a mode whose defaults are
**calibrated to real CLIF statistics** — these are the mode's *rules*, and they are
a starting point, not a fixed recipe. A CLIF user overrides any of them per
generation (via the spec or the Python API) to make a distinct, still-conformant
derivative:

| Mode | Default rules (calibrated to real CLIF) |
|---|---|
| **`icu`** (ICU cohort) | every stay an ICU stay · in-hospital mortality ~9.5% · invasive ventilation ~41% · vasopressors ~33% · CRRT ~4% · hospital-LOS median ~165 h |
| **`full_hospital`** (whole hospital) | ~15% reach the ICU · hospital-LOS median ~67 h · mortality ~2% · admissions ED 76% / OR 10% / direct 10% / OSH 3% · ICU access = ward→ICU ~12% + OSH→ICU ~3% + planned ~2% |

### What's tweakable

| Axis | Spec field(s) | Notes |
|---|---|---|
| **Population shape** | `mode` | `icu` \| `full_hospital` |
| **Size** | `n` / `--n-patients` | any count; chunked + collision-free |
| **Demographics** | `age_shift`, `hispanic_frac`, `race_target` | relative to the base pack |
| **Illness rates** (ICU) | `imv`, `mortality_scale`, `vaso_frac`, `crrt_prob`, `prone_severe` | organ-support and death targets |
| **Resources** | `--chunk-size`, `--max-threads` | RAM and CPU dials (see System requirements) |
| **Seed** | `seed` / `--seed` | one seed → byte-identical output |

For finer control the **Python API** exposes every recalibration knob directly —
sojourn length + tail shape, non-invasive-support rates, lab panel cadence,
terminal-deterioration window, and (full-hospital) the ICU/stepdown targets and
the `admission_route_marginal` that drives the ER/OR/direct/OSH→ICU flow:

```python
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.recalibrate import (
    recalibrate_to_network_median,   # ICU cohort
    recalibrate_to_full_hospital,    # whole-hospital population
)
from clifforge.generate.orchestrator import generate_dataset

base = ParamPack.load("base_pack")
pack = recalibrate_to_full_hospital(base, icu_target=0.10, mortality_target=0.03)
dataset = generate_dataset(pack, n_patients=50_000, seed=7)   # never mutates `base`
```

Every generated dataset writes a `manifest.json` (resolved recipe, seed, generator
version, per-table content hashes), so a variant is reproducible from its recipe and
any two variants are provably distinct. Recalibration is a pure function on a deep
copy, so one base pack seeds unlimited variants.

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

### Validate against your own real CLIF

`scripts/validate_against_real.py` audits any generated dataset against a real
staged CLIF reference and emits a JSON summary + printed table — cohort size,
length-of-stay *distribution*, mortality, life support, per-stay missingness,
lab-value fidelity, patient flow, and the deterioration-toward-death trajectory
(no row-level data retained):

```bash
uv run python scripts/validate_against_real.py \
    --synthetic ./my-dataset --real /path/to/real-clif --out validation.json
```

### Validation report

A [synthetic-vs-real validation report](https://claude.ai/code/artifact/d72f6a8d-b209-469d-82ad-b99d2b9c3cc1)
audits both shipped populations against real CLIF — the ICU cohort against the
real ICU population and the whole-hospital dataset against the real whole-hospital
population — across missingness, length-of-stay, life support, death, patient flow,
lab-value shape, the longitudinal illness trajectory, and privacy. It documents
what matches and what is intentionally different by design (demographics,
network-median mortality, cleaner patient flow).

## Status

The fit and generate stages, both population modes (ICU cohort and whole-hospital),
and all three evaluation surfaces are implemented; all 19 tables generate and pass
conformance. The fit stage requires a staged real CLIF set and is **not** part of
the public distribution — generation needs only the committed `base_pack/`. See
`docs/REPRODUCIBILITY.md` for the full pipeline and `docs/plans/` for the design.

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

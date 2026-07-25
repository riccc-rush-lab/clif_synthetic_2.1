---
title: Spine over-mixing bounds respiratory/adt fidelity; acuity de-escalation is a low-cost fix
date: 2026-07-25
input_shape: solution
subject: fidelity, spine, respiratory_support, adt
---

## Problem

When shaping a synthetic dataset to match a real ICU reference, most tables reach
statistical region (labs 0.93, vitals 0.82, medication 0.95 after fitting the med
marginal), but `respiratory_support` stalls at ~0.64 and `adt` at ~0.84. Adding
missing devices (Face Mask/NIPPV) and switching charting granularity did **not**
help — the device *proportions* stayed wrong (IMV over-represented, low-flow
oxygen under-represented).

## Root cause

`respiratory_support` and `adt` derive their category from the latent
`support_level` — the ordinal spine ladder *is* the respiratory-support axis. So
their marginal is a deterministic image of the spine's acuity-*time* distribution.
Measured: the spine spends ~46-52% of interval-time at level >= 3 (IMV), vs a real
ICU device fraction of ~0.356. This is the **known U6 over-mixing artifact** — the
first-order semi-Markov chain mixes into high-acuity peaks more than the real data
(real peak-level L4=0.342 vs sampled ~0.587). It is a *spine* property, not a
device-generator bug: no mapping or granularity change in the device generator can
fix a level-time distribution that is itself too hot.

## Fix (measured tradeoff)

`clifforge.generate.recalibrate.recalibrate_spine_acuity(pack, deescalation=...)`
returns a derived pack that moves a fraction of each high level's transition mass
to `discharge` (and lowers high-level start probabilities), spending less time in
the hottest states. Applied to the Chicago pack (3k stays, vs the ICU real
cohort), `deescalation=0.45`:

| table | before | after |
|---|---|---|
| respiratory_support | 0.639 | **0.757** |
| adt | 0.841 | **0.869** |
| vitals | 0.814 | 0.817 |
| labs | 0.908 | 0.906 |
| medication_admin_continuous | 0.950 | 0.948 |

Respiratory and adt rise; the fitted tables move < 0.005. **The key insight: this
has near-zero cost** because vitals/labs fidelity is measured on the *value*
distribution, and re-weighting the *time* spent in each state (with the per-state
AR(1)/copula parameters unchanged) barely shifts the value mixture. The earlier
assumption that "any spine change trades against vitals/labs" was wrong for a
*time-reweighting* change.

## When to reach for it

Applying to a **derived** pack when you want a more realistic acuity mix for a
shaped population; the base fitted pack is left as-fit. It does not reach 0.9 for
respiratory (the crude single-factor de-escalation only partially corrects the
over-mixing) — a fuller fix belongs in the fit stage (calibrate the semi-Markov
transition dynamics to a target peak-level marginal), which this finding scopes.

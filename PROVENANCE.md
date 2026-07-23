# Provenance of generated CLIF 2.1 tables (R4, R14)

`clif-forge` generates every table from an **aggregate parameter pack** and a
latent acuity **spine** — no real patient record ever leaves the fit stage. This
document records, per table, *how* its content is produced, so downstream users
can distinguish empirically-fitted structure from documented priors.

Three provenance classes:

- **fitted** — sampled from parameters fit over real CLIF-MIMIC at the fit stage
  (U5) and stored in the versioned parameter pack.
- **spine-derived** — no separate fitted block; structure is a deterministic or
  heuristic function of the fitted latent spine (organ-support trajectory,
  organ-failure flags, outcome), with un-fitted fields set to documented clinical
  constants (R15).
- **prior-driven** — no fit and no consortium prior file; content comes from
  documented literature / clinical-norm rates keyed to spine acuity (R14).

| Table | Provenance | Basis |
|-------|------------|-------|
| `patient` | fitted | pack demographic marginals |
| `hospitalization` | fitted | pack admission/discharge marginals + spine LOS/outcome (AE4) |
| `vitals` | fitted | pack per-state AR(1) physiology |
| `labs` | fitted | pack Gaussian-copula (correlation + log-normal marginals + presence) |
| `medication_admin_continuous` | fitted | pack per-med infusion hazards + spine couplings |
| *(latent spine)* | fitted | pack semi-Markov state model + per-level flag prevalences |
| `adt` | spine-derived | acuity RLE into ward/ICU segments |
| `respiratory_support` | spine-derived | support-ladder device sequence + R10 matrix, AE1/AE2 |
| `patient_assessments` | spine-derived | RASS↔sedation, GCS↔neuro-failure flag |
| `position` | spine-derived | prone↔severe-hypoxemia (resp+IMV) |
| `medication_admin_intermittent` | spine-derived | documented antibiotic schedule over the stay |
| `microbiology_culture` | spine-derived | documented per-ICU-day culture rate |
| `crrt_therapy` | spine-derived | CRRT during renal-failure windows + documented rates |
| `code_status` | spine-derived | de-escalation before death (spine outcome); **plan intends fitted — no pack block yet** |
| `ecmo_mcs` | prior-driven | ECMO-tier acuity + documented adult VV-ECMO device norms |
| `invasive_hemodynamics` | prior-driven | PA-catheter events during cv-failure windows |
| `transfusion` | prior-driven | peak-acuity-scaled rate + documented product volumes |
| `key_icu_orders` | prior-driven | PT/OT rehab orders for a subset of ICU stays |
| `therapy_details` | prior-driven | documented PT/OT session elements |
| `provider` | prior-driven | one attending + one nurse spanning each stay |

**Release gate:** any public release of a generated dataset or the parameter pack
requires PhysioNet/MIMIC-IV credentialed-DUA and Rush compliance confirmation.
Fitting locally to an aggregate pack is permitted; releasing derived artifacts is
gated.

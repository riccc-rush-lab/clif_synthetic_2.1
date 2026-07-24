# CLIFForge demo evaluation report (n=100, seed 42)

Generated 2026-07-24. All data is fully synthetic — sampled from an aggregate parameter pack, never copied from a real record.

**Reference:** a second independent synthetic draw (seed 43). These comparative numbers measure **generator self-consistency across seeds**, NOT fidelity to real patient data — computing real-data fidelity requires a credentialed CLIF-MIMIC reference and is gated by the PhysioNet DUA.

## 1. Dataset

19 tables, 44,066 rows total.

| Table | Rows | Columns |
|---|---:|---:|
| `adt` | 972 | 8 |
| `code_status` | 127 | 4 |
| `crrt_therapy` | 184 | 10 |
| `ecmo_mcs` | 205 | 9 |
| `hospitalization` | 100 | 9 |
| `invasive_hemodynamics` | 324 | 4 |
| `key_icu_orders` | 288 | 5 |
| `labs` | 3,570 | 6 |
| `medication_admin_continuous` | 4,150 | 11 |
| `medication_admin_intermittent` | 848 | 11 |
| `microbiology_culture` | 106 | 9 |
| `patient` | 100 | 8 |
| `patient_assessments` | 2,122 | 5 |
| `position` | 750 | 4 |
| `provider` | 200 | 6 |
| `respiratory_support` | 1,051 | 12 |
| `therapy_details` | 290 | 5 |
| `transfusion` | 88 | 8 |
| `vitals` | 28,591 | 5 |

## 2. Validation

All tables pass the primary (pandera) conformance gate.

| Table | Primary (pandera) | Secondary (clifpy) | Note |
|---|---|---|---|
| `adt` | pass | failed | clifpy reported 3 advisory issue(s) for 'adt' (recorded, not blocking — pandera  |
| `code_status` | pass | failed | clifpy reported 1 advisory issue(s) for 'code_status' (recorded, not blocking —  |
| `crrt_therapy` | pass | failed | clifpy reported 1 advisory issue(s) for 'crrt_therapy' (recorded, not blocking — |
| `ecmo_mcs` | pass | failed | clifpy reported 3 advisory issue(s) for 'ecmo_mcs' (recorded, not blocking — pan |
| `hospitalization` | pass | failed | clifpy reported 7 advisory issue(s) for 'hospitalization' (recorded, not blockin |
| `invasive_hemodynamics` | pass | skipped | no clifpy validator for 'invasive_hemodynamics' (pandera alone gates this table) |
| `key_icu_orders` | pass | skipped | no clifpy validator for 'key_icu_orders' (pandera alone gates this table) |
| `labs` | pass | failed | clifpy reported 6 advisory issue(s) for 'labs' (recorded, not blocking — pandera |
| `medication_admin_continuous` | pass | failed | clifpy reported 5 advisory issue(s) for 'medication_admin_continuous' (recorded, |
| `medication_admin_intermittent` | pass | failed | clifpy reported 5 advisory issue(s) for 'medication_admin_intermittent' (recorde |
| `microbiology_culture` | pass | failed | clifpy reported 5 advisory issue(s) for 'microbiology_culture' (recorded, not bl |
| `patient` | pass | failed | clifpy reported 4 advisory issue(s) for 'patient' (recorded, not blocking — pand |
| `patient_assessments` | pass | failed | clifpy reported 3 advisory issue(s) for 'patient_assessments' (recorded, not blo |
| `position` | pass | passed | clifpy secondary gate passed for 'position' |
| `provider` | pass | skipped | no clifpy validator for 'provider' (pandera alone gates this table) |
| `respiratory_support` | pass | failed | clifpy reported 8 advisory issue(s) for 'respiratory_support' (recorded, not blo |
| `therapy_details` | pass | skipped | no clifpy validator for 'therapy_details' (pandera alone gates this table) |
| `transfusion` | pass | skipped | no clifpy validator for 'transfusion' (pandera alone gates this table) |
| `vitals` | pass | failed | clifpy reported 1 advisory issue(s) for 'vitals' (recorded, not blocking — pande |

## 3. Fidelity

SDMetrics column-shape and column-pair similarity (1.0 = identical distribution). Mean quality across 19 tables: **0.960**.

| Table | Quality | Column shapes | Column pairs | Cols | Pairs |
|---|---:|---:|---:|---:|---:|
| `transfusion` | 0.873 | 0.881 | 0.864 | 4 | 3 |
| `hospitalization` | 0.887 | 0.920 | 0.853 | 4 | 6 |
| `microbiology_culture` | 0.898 | 0.938 | 0.858 | 5 | 10 |
| `patient` | 0.911 | 0.947 | 0.875 | 6 | 15 |
| `labs` | 0.924 | 0.937 | 0.912 | 3 | 1 |
| `code_status` | 0.947 | 0.947 | 0.947 | 2 | 1 |
| `position` | 0.958 | 0.958 | 0.958 | 2 | 1 |
| `crrt_therapy` | 0.960 | 0.956 | 0.964 | 7 | 11 |
| `invasive_hemodynamics` | 0.961 | 0.961 | 0.961 | 2 | 1 |
| `respiratory_support` | 0.970 | 0.962 | 0.978 | 7 | 7 |
| `ecmo_mcs` | 0.982 | 0.973 | 0.990 | 7 | 9 |
| `key_icu_orders` | 0.991 | 0.993 | 0.989 | 3 | 3 |
| `medication_admin_continuous` | 0.993 | 0.994 | 0.992 | 8 | 20 |
| `vitals` | 0.994 | 0.993 | 0.996 | 3 | 1 |
| `patient_assessments` | 0.997 | 0.993 | 1.000 | 3 | 1 |
| `adt` | 0.999 | 0.999 | 0.998 | 4 | 6 |
| `medication_admin_intermittent` | 0.999 | 0.999 | 0.999 | 8 | 20 |
| `provider` | 1.000 | 1.000 | 1.000 | 2 | 1 |
| `therapy_details` | 1.000 | 1.000 | 1.000 | 3 | 3 |

## 4. Privacy

| Metric | Value | Reading |
|---|---:|---|
| DCR (median) | 6.282 | distance to the closest reference record |
| DCR (5th pct) | 2.336 | closest-match tail — none detected |
| NN-distance ratio (median) | 0.961 | near 1.0 = not singling out one record |
| Identifiability | 0.530 | fraction of reference records closer to a synthetic than to another reference record |

## 5. Utility (TSTR)

In-hospital mortality, LightGBM, both models scored on the same reference test split.

| Metric | Value |
|---|---:|
| TSTR AUC (trained on synthetic) | 0.739 |
| TRTR AUC (trained on reference) | 0.621 |
| Utility gap (TRTR - TSTR) | -0.117 |
| Features / test rows | 52 / 50 |

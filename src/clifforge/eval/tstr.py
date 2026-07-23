"""Train-on-Synthetic, Test-on-Real (TSTR) utility evaluation (U22; R19).

The utility question: does a mortality model trained on the *synthetic* data
generalize to *real* patients? We build a per-hospitalization wide feature frame
(demographics + length of stay + per-vital and per-lab means) with an in-hospital
mortality label, then compare two LightGBM models on the **same** real test set:

* **TSTR** — trained on synthetic, tested on real.
* **TRTR** — trained on real (a disjoint real-train split), tested on real.

The utility gap ``TRTR_AUC - TSTR_AUC`` is how much discriminative signal is lost
by training on synthetic instead of real (R19). LightGBM is pinned deterministic
(``deterministic=True, num_threads=1``, seeded) so the report is reproducible.
All feature work stays in polars; ``.to_pandas()`` happens only at the LightGBM
boundary.

**Leakage guard (R19).** When the real test set is a CLIF-MIMIC split rather than
Rush-CLIF, :func:`assert_holdout_disjoint` recomputes each test patient's
partition from the pack manifest's split spec (the same ``sha1_mod_10000``
predicate the fit stage used) and fails if any test patient was in the fit
partition — a mandatory check that the reported utility isn't inflated by leakage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from clifforge.fit.run_fit import _holdout_mask

__all__ = [
    "TstrReport",
    "assert_holdout_disjoint",
    "build_wide_features",
    "run_tstr",
]

_DEATH_DISCHARGE_CATEGORY = "Expired"
_ID = "hospitalization_id"
_LABEL = "label"


@dataclass(frozen=True)
class TstrReport:
    """Utility-gap result for a synthetic-vs-real comparison."""

    tstr_auc: float  # trained on synthetic, tested on real
    trtr_auc: float  # trained on real, tested on real (baseline)
    auc_gap: float  # trtr_auc - tstr_auc (signal lost by training on synthetic)
    n_features: int
    n_train_synthetic: int
    n_test_real: int


def _mean_pivot(long: pl.DataFrame, category_col: str, value_col: str, prefix: str) -> pl.DataFrame:
    """Per-hospitalization mean of ``value_col`` for each ``category_col``, wide."""
    if long.height == 0:
        return pl.DataFrame({_ID: []}, schema={_ID: pl.String})
    agg = long.group_by(_ID, category_col).agg(pl.col(value_col).mean().alias("_m"))
    wide = agg.pivot(values="_m", index=_ID, on=category_col)
    return wide.rename({c: f"{prefix}_{c}" for c in wide.columns if c != _ID})


def build_wide_features(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per hospitalization: mortality label + numeric features (R19).

    Features: length of stay (hours), female flag, per-vital means, per-lab means.
    The label is in-hospital mortality (``discharge_category == "Expired"``).
    """
    hosp = tables["hospitalization"]
    patient = tables["patient"].select("patient_id", "sex_category")

    base = (
        hosp.join(patient, on="patient_id", how="left")
        .with_columns(
            (
                (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 3600.0
            ).alias("los_hours"),
            (pl.col("sex_category") == "Female").cast(pl.Int8).alias("sex_female"),
            (pl.col("discharge_category") == _DEATH_DISCHARGE_CATEGORY).cast(pl.Int8).alias(_LABEL),
        )
        .select(_ID, _LABEL, "los_hours", "sex_female")
    )

    features = base
    if "vitals" in tables:
        features = features.join(
            _mean_pivot(tables["vitals"], "vital_category", "vital_value", "vital"),
            on=_ID,
            how="left",
        )
    if "labs" in tables:
        features = features.join(
            _mean_pivot(tables["labs"], "lab_category", "lab_value_numeric", "lab"),
            on=_ID,
            how="left",
        )
    return features


def _classifier(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        random_state=seed,
        deterministic=True,
        num_threads=1,
        force_row_wise=True,
        n_estimators=100,
        verbose=-1,
    )


def _xy(features: pl.DataFrame, columns: list[str]) -> tuple[Any, npt.NDArray[Any]]:
    """Split a feature frame into an aligned pandas X and a label array y.

    ``.to_pandas()`` is the single LightGBM boundary; missing feature columns are
    filled with nulls so synthetic and real share an identical feature space.
    """
    present = {c for c in features.columns if c not in (_ID, _LABEL)}
    aligned = features.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in columns if c not in present]
    ).select(columns)
    y = features[_LABEL].to_numpy()
    return aligned.to_pandas(), y


def run_tstr(
    synthetic: Mapping[str, pl.DataFrame],
    real: Mapping[str, pl.DataFrame],
    *,
    seed: int = 0,
) -> TstrReport:
    """Train on synthetic + on real, test both on the same real split; report the gap."""
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    synth_feat = build_wide_features(synthetic)
    real_feat = build_wide_features(real)

    columns = sorted((set(synth_feat.columns) | set(real_feat.columns)) - {_ID, _LABEL})

    # Seeded 50/50 split of the real set so TRTR trains and tests on disjoint reals.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(real_feat.height)
    cut = real_feat.height // 2
    real_train = real_feat[perm[:cut].tolist()]
    real_test = real_feat[perm[cut:].tolist()]

    x_synth, y_synth = _xy(synth_feat, columns)
    x_rtrain, y_rtrain = _xy(real_train, columns)
    x_rtest, y_rtest = _xy(real_test, columns)

    if len(np.unique(y_rtest)) < 2:
        raise ValueError("real test split has a single mortality class; AUC is undefined")

    tstr_model = _classifier(seed).fit(x_synth, y_synth)
    tstr_auc = float(roc_auc_score(y_rtest, tstr_model.predict_proba(x_rtest)[:, 1]))

    trtr_model = _classifier(seed).fit(x_rtrain, y_rtrain)
    trtr_auc = float(roc_auc_score(y_rtest, trtr_model.predict_proba(x_rtest)[:, 1]))

    return TstrReport(
        tstr_auc=tstr_auc,
        trtr_auc=trtr_auc,
        auc_gap=trtr_auc - tstr_auc,
        n_features=len(columns),
        n_train_synthetic=synth_feat.height,
        n_test_real=real_test.height,
    )


def assert_holdout_disjoint(
    real_patient_ids: Iterable[str],
    manifest: Mapping[str, Any],
) -> None:
    """Fail if any real test patient was in the pack's fit partition (R19 leakage guard).

    Recomputes each patient's partition from the manifest's ``split`` spec using the
    identical ``sha1_mod_10000`` predicate the fit stage used. Raises if the split
    spec is absent (so the guard can never be silently skipped) or if any test
    patient hashes into the training partition.
    """
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError(
            "pack manifest has no 'split' spec; cannot verify a leakage-free evaluation"
        )
    method = split.get("method")
    if method != "sha1_mod_10000":
        raise ValueError(f"unsupported split method {method!r}; cannot verify holdout membership")
    seed = int(split["seed"])
    fraction = float(split["holdout_fraction"])

    leaked = [pid for pid in real_patient_ids if not _holdout_mask(pid, seed, fraction)]
    if leaked:
        raise ValueError(
            f"{len(leaked)} test patient(s) were in the fit partition (leakage): "
            f"{sorted(leaked)[:5]}"
        )

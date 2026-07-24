"""Privacy metrics — distance-based memorization risk (U23; R20).

Three standard synthetic-data privacy metrics, computed **standalone** from numpy
+ scikit-learn with **no torch / synthcity dependency** (R20's goal is to contain
the torch tree; here it is contained to zero — synthcity 0.2.11's metrics module
is unusable against its own resolved torch, so the metric definitions are
implemented directly):

* **DCR** (Distance to Closest Record): for each synthetic record, the distance to
  the nearest *real* record. A DCR at/near zero means a synthetic record
  reproduces a real one (memorization). We report the median and 5th percentile —
  the 5th percentile being the privacy-relevant tail (the closest matches).
* **NN-distance ratio** (NNDR): for each synthetic record, nearest-real distance
  divided by second-nearest-real distance. Values near 1 mean a synthetic record
  is not singling out one specific real individual.
* **Identifiability**: the fraction of real records whose nearest neighbour in the
  synthetic set is closer than their nearest neighbour in the real set — the
  Yoon et al. identifiability score. Lower is more private.

Distances are Euclidean on features standardized against the **real** reference
(mean/variance fit on real, so scale can't dominate), with missing per-vital/lab
means imputed to the real column mean. The wide feature frame is the same one the
TSTR utility eval builds. Metrics are deterministic given the inputs.

Because the generators sample from **aggregate** parameters and never copy a real
row, DCR stays well above zero (no memorization) — the R20 "aggregate-only scores
clean" expectation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from clifforge.eval.tstr import build_wide_features

__all__ = ["PrivacyReport", "privacy_metrics"]

_ID = "hospitalization_id"
_LABEL = "label"


@dataclass(frozen=True)
class PrivacyReport:
    """Distance-based privacy metrics for a synthetic-vs-real comparison."""

    dcr_median: float  # median distance from a synthetic record to the nearest real record
    dcr_p5: float  # 5th-percentile DCR (the closest matches; ~0 would signal memorization)
    nndr_median: float  # median nearest/second-nearest real distance ratio (near 1 is safe)
    identifiability: float  # fraction of real records closer to a synthetic than to another real
    n_synthetic: int
    n_real: int


def _aligned_matrix(features: pl.DataFrame, columns: list[str]) -> npt.NDArray[np.float64]:
    """Feature frame -> numeric matrix over ``columns`` (missing columns become NaN)."""
    present = {c for c in features.columns if c not in (_ID, _LABEL)}
    aligned = features.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in columns if c not in present]
    ).select(columns)
    return aligned.to_numpy().astype(np.float64)


def privacy_metrics(
    synthetic: Mapping[str, pl.DataFrame],
    real: Mapping[str, pl.DataFrame],
) -> PrivacyReport:
    """Compute DCR, NN-distance ratio, and identifiability (R20). Deterministic."""
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    synth_feat = build_wide_features(synthetic)
    real_feat = build_wide_features(real)
    columns = sorted((set(synth_feat.columns) | set(real_feat.columns)) - {_ID, _LABEL})

    xs_raw = _aligned_matrix(synth_feat, columns)
    xr_raw = _aligned_matrix(real_feat, columns)

    # Impute + standardize against the real reference so neither missingness nor
    # feature scale distorts the distances.
    imputer = SimpleImputer(strategy="mean").fit(xr_raw)
    scaler = StandardScaler().fit(imputer.transform(xr_raw))
    xr = scaler.transform(imputer.transform(xr_raw))
    xs = scaler.transform(imputer.transform(xs_raw))

    # DCR + NNDR: synthetic -> nearest / second-nearest real.
    nn_real2 = NearestNeighbors(n_neighbors=2).fit(xr)
    d_s2r, _ = nn_real2.kneighbors(xs)
    dcr = d_s2r[:, 0]
    nndr = dcr / (d_s2r[:, 1] + 1e-12)

    # Identifiability: real -> nearest OTHER real (col 1 excludes self) vs nearest synthetic.
    d_r2r, _ = NearestNeighbors(n_neighbors=2).fit(xr).kneighbors(xr)
    d_real_nn = d_r2r[:, 1]
    d_r2s, _ = NearestNeighbors(n_neighbors=1).fit(xs).kneighbors(xr)
    identifiability = float(np.mean(d_r2s[:, 0] < d_real_nn))

    return PrivacyReport(
        dcr_median=float(np.median(dcr)),
        dcr_p5=float(np.percentile(dcr, 5)),
        nndr_median=float(np.median(nndr)),
        identifiability=identifiability,
        n_synthetic=int(xs.shape[0]),
        n_real=int(xr.shape[0]),
    )

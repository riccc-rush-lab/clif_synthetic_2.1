"""Aggregate estimators for the empirical-fidelity fit stage (U5).

Each estimator takes eager polars frames (``run_fit`` owns reading the real
data and collecting — KTD-1) and returns a ``(params, suppression)`` pair:

* ``params`` — a JSON-serializable dict of **aggregate** statistics only
  (marginals, transition probabilities, parametric-family parameters, a
  correlation matrix, hazards). Never a per-record value (R1).
* ``suppression`` — the ``cell_gate.SuppressionRecord`` audit list, so the
  pack manifest can report exactly which cells fell below the n>=20 floor (R2).

Every cell (a category, a transition pair, a per-state physiology fit, a lab,
a med) is routed through :func:`cell_gate.suppress` before it can enter the
pack — that is the single choke point enforcing the count floor.

The estimators are deliberately model-light and inspectable:

* transitions — the **embedded** (jump) chain over the organ-support ladder:
  self-transitions are removed, so the diagonal is zero and each row of the
  emitted matrix sums to 1 over the *other* states plus an absorbing
  ``discharge`` exit; the initial-state law is emitted alongside as
  ``support_level_start_dist`` (both consumed by the U6 spine sampler).
* sojourns — per-state dwell time, best parametric family chosen by AIC among
  exponential / gamma / lognormal / Weibull.
* AR1 — per (vital, support-level) first-order autoregression on a fixed grid.
* lab copula — Spearman correlation over co-measured labs, projected to the
  nearest positive-definite correlation matrix, plus per-lab log-marginals and
  presence rates.
* infusion hazards — per-drug start/stop hazard per interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from clifforge.fit.cell_gate import SuppressionRecord, suppress

#: 1-D/2-D float64 array alias for the numeric estimator internals.
_F64 = NDArray[np.float64]

__all__ = [
    "DISCHARGE_STATE",
    "EstimatorResult",
    "fit_categorical_marginals",
    "fit_continuous_marginals",
    "fit_transitions",
    "fit_sojourns",
    "fit_outcome_rates",
    "fit_flag_prevalence",
    "fit_ar1_by_state",
    "fit_lab_copula",
    "fit_infusion_hazards",
    "nearest_positive_definite_correlation",
]

#: (params-block, suppression-audit) — the shape every estimator returns.
EstimatorResult = tuple[dict[str, object], list[SuppressionRecord]]

#: candidate positive-support families for sojourn dwell times, fit with the
#: location pinned at 0 (durations are non-negative).
_SOJOURN_FAMILIES: dict[str, stats.rv_continuous] = {
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
}

#: AR1 coefficient is clamped just inside the unit circle for stationarity.
_PHI_CLAMP = 0.999


# --------------------------------------------------------------------------- #
# Marginals
# --------------------------------------------------------------------------- #
def fit_categorical_marginals(
    df: pl.DataFrame, fields: Sequence[str], *, min_n: int = 20
) -> EstimatorResult:
    """Per-field category proportions, each category gated at ``min_n`` (R2).

    A category observed fewer than ``min_n`` times is dropped from its field's
    marginal (recorded in the audit); the surviving proportions are then
    renormalized to sum to 1 so the emitted marginal is a proper distribution.
    """
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []
    for field in fields:
        if field not in df.columns:
            continue
        vc = df.select(pl.col(field).drop_nulls()).to_series().value_counts(sort=True)
        counts = {row[0]: int(row[1]) for row in vc.iter_rows()}
        total = sum(counts.values())
        if total == 0:
            continue
        raw_props = {cat: n / total for cat, n in counts.items()}
        survived, records = suppress(counts, raw_props, min_n=min_n)
        audit.extend(
            SuppressionRecord(cell=(field, r.cell), n=r.n, fallback_kind=r.fallback_kind)
            for r in records
        )
        surviving_total = sum(survived.values())
        if surviving_total > 0:
            params[f"{field}_marginal"] = {
                cat: prop / surviving_total for cat, prop in survived.items()
            }
    return params, audit


def fit_continuous_marginals(
    df: pl.DataFrame, fields: Sequence[str], *, n_bins: int = 10, min_n: int = 20
) -> EstimatorResult:
    """Coarse quantile bin-edges per field (a compact, non-leaking marginal).

    Emits at most ``n_bins`` bins, and only when the whole field clears the
    ``min_n`` floor. The edge list has ``<= n_bins + 1`` entries regardless of
    ``n_records``, so it can never approach a raw column (R1/R2).
    """
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []
    for field in fields:
        if field not in df.columns:
            continue
        col = df.select(pl.col(field).drop_nulls().cast(pl.Float64)).to_series()
        n = col.len()
        counts = {field: n}
        quantiles = [i / n_bins for i in range(n_bins + 1)]
        raw_edges = [col.quantile(q, interpolation="linear") for q in quantiles]
        edges = sorted({round(float(e), 6) for e in raw_edges if e is not None})
        survived, records = suppress(counts, {field: edges}, min_n=min_n)
        audit.extend(records)
        if field in survived and len(survived[field]) >= 2:
            params[f"{field}_quantile_bin_edges"] = survived[field]
    return params, audit


# --------------------------------------------------------------------------- #
# Semi-Markov spine: transitions + sojourns
# --------------------------------------------------------------------------- #
def _runs(state_timeline: pl.DataFrame) -> pl.DataFrame:
    """Run-length-encode the per-hospitalization support-level sequence.

    Returns one row per run with ``hospitalization_id``, ``run_idx`` (0-based
    within the hospitalization), ``support_level``, and ``start_interval``.
    Consecutive equal states collapse into one run — the basis for both the
    embedded transition chain and the sojourn dwell times.
    """
    tl = state_timeline.sort("hospitalization_id", "interval_idx")
    prev = pl.col("support_level").shift(1).over("hospitalization_id")
    changed = (pl.col("support_level") != prev) | prev.is_null()
    tl = tl.with_columns(changed.cast(pl.Int64).alias("_chg")).with_columns(
        pl.col("_chg").cum_sum().over("hospitalization_id").alias("run_idx")
    )
    return (
        tl.group_by("hospitalization_id", "run_idx")
        .agg(
            pl.col("support_level").first().alias("support_level"),
            pl.col("interval_idx").min().alias("start_interval"),
        )
        .sort("hospitalization_id", "run_idx")
    )


#: absorbing target that ends a hospitalization; a run with no following run is
#: a discharge (alive or dead — the terminal *outcome* is modelled separately by
#: ``fit_outcome_rates``). Emitted as a competing-risk column in every transition
#: row so the U6 spine sampler terminates naturally instead of running to horizon.
DISCHARGE_STATE = "discharge"


def fit_transitions(state_timeline: pl.DataFrame, *, min_n: int = 20) -> EstimatorResult:
    """Embedded (jump-chain) transition matrix over the support ladder (R2).

    Three parameters are emitted, all gated at ``min_n`` and all aggregate-only:

    * ``support_level_states`` — the sorted set of observed support levels.
    * ``support_level_start_dist`` — the distribution of the **first** run's
      support level per hospitalization; the U6 spine draws its initial state
      from this (R15: the sampler must not invent an initial condition).
    * ``support_level_transition_matrix`` — a nested ``{from: {to: prob}}`` map.
      ``from -> to`` counts observed jumps (``from != to`` by run construction);
      each row additionally carries a :data:`DISCHARGE_STATE` competing-risk
      entry counting the runs at ``from`` that were **terminal** (the
      hospitalization ended rather than jumping onward). The matrix has a **zero
      diagonal**, and each row sums to 1 over its reachable next-states plus
      discharge — so the trajectory always has an exit and terminates without a
      horizon cap. ``discharge`` is absorbing; it has no sojourn and no outgoing
      row.
    """
    runs = _runs(state_timeline)

    # --- initial-state distribution (first run per hospitalization) -----------
    first_levels = (
        runs.sort("hospitalization_id", "run_idx")
        .group_by("hospitalization_id")
        .agg(pl.col("support_level").first().alias("start_level"))
    )
    start_counts = {
        int(r["start_level"]): int(r["n"])
        for r in first_levels.group_by("start_level")
        .len()
        .rename({"len": "n"})
        .iter_rows(named=True)
    }
    start_survived, start_audit = suppress(start_counts, start_counts, min_n=min_n)
    start_total = sum(start_survived.values())
    start_dist = (
        {str(level): n / start_total for level, n in start_survived.items()}
        if start_total > 0
        else {}
    )

    # --- jumps + discharge competing risk -------------------------------------
    nxt = pl.col("support_level").shift(-1).over("hospitalization_id")
    with_next = runs.with_columns(nxt.alias("to_level"))
    jump_counts = (
        with_next.drop_nulls("to_level")
        .group_by("support_level", "to_level")
        .len()
        .rename({"len": "n"})
    )
    discharge_counts = (
        with_next.filter(pl.col("to_level").is_null())
        .group_by("support_level")
        .len()
        .rename({"len": "n"})
    )

    # Cell key = (from_level:int, to:str) where ``to`` is a level string or
    # DISCHARGE_STATE; gate every ordered pair (including the discharge column).
    counts: dict[tuple[int, str], int] = {
        (int(r["support_level"]), str(int(r["to_level"]))): int(r["n"])
        for r in jump_counts.iter_rows(named=True)
    }
    for r in discharge_counts.iter_rows(named=True):
        counts[(int(r["support_level"]), DISCHARGE_STATE)] = int(r["n"])
    survived, audit = suppress(counts, counts, min_n=min_n)

    # Row-normalize surviving counts into a nested {from: {to: prob}} matrix.
    row_totals: dict[int, int] = {}
    for (frm, _to), n in survived.items():
        row_totals[frm] = row_totals.get(frm, 0) + n
    matrix: dict[str, dict[str, float]] = {}
    for (frm, to), n in survived.items():
        if row_totals[frm] > 0:
            matrix.setdefault(str(frm), {})[to] = n / row_totals[frm]

    states = sorted({frm for frm, _to in counts})
    params: dict[str, object] = {
        "support_level_states": states,
        "support_level_start_dist": start_dist,
        "support_level_transition_matrix": matrix,
    }
    return params, audit + start_audit


def fit_sojourns(
    state_timeline: pl.DataFrame, *, grid_step_hours: float = 1.0, min_n: int = 20
) -> EstimatorResult:
    """Per-state dwell-time family chosen by AIC (R2).

    Dwell time of a run = (start of the next run - start of this run) in hours.
    Terminal runs (no following run — right-censored at discharge) are dropped
    to keep the fit a plain complete-data MLE; the count of dropped censored
    runs is not emitted (aggregate-only). Each state is gated at ``min_n``
    non-censored sojourns.
    """
    runs = _runs(state_timeline)
    nxt_start = pl.col("start_interval").shift(-1).over("hospitalization_id")
    durations = (
        runs.with_columns(nxt_start.alias("_next_start"))
        .drop_nulls("_next_start")
        .with_columns(
            (
                (pl.col("_next_start") - pl.col("start_interval")).cast(pl.Float64)
                * grid_step_hours
            ).alias("duration_hours")
        )
        .filter(pl.col("duration_hours") > 0)
        .select("support_level", "duration_hours")
    )

    by_state: dict[int, list[float]] = {}
    for row in durations.iter_rows(named=True):
        by_state.setdefault(int(row["support_level"]), []).append(float(row["duration_hours"]))

    counts = {state: len(vals) for state, vals in by_state.items()}
    fits = {
        state: _best_sojourn_family(np.asarray(vals, dtype=float))
        for state, vals in by_state.items()
    }
    survived, audit = suppress(counts, fits, min_n=min_n)
    params: dict[str, object] = {
        "support_level_sojourn": {str(state): fit for state, fit in survived.items()}
    }
    return params, audit


def _best_sojourn_family(durations: _F64) -> dict[str, object]:
    """Fit each candidate family (loc=0) and return the min-AIC choice."""
    best: dict[str, object] | None = None
    best_aic = np.inf
    for name, dist in _SOJOURN_FAMILIES.items():
        try:
            fitted = dist.fit(durations, floc=0.0)
            loglik = float(np.sum(dist.logpdf(durations, *fitted)))
        except Exception:  # noqa: BLE001 — a family that fails to fit is just skipped
            continue
        if not np.isfinite(loglik):
            continue
        k = len(fitted)
        aic = 2 * k - 2 * loglik
        if aic < best_aic:
            best_aic = aic
            best = {
                "family": name,
                "params": [round(float(p), 6) for p in fitted],
                "aic": round(float(aic), 4),
                "mean_hours": round(float(np.mean(durations)), 4),
            }
    return best or {
        "family": "empirical_mean",
        "params": [round(float(np.mean(durations)), 6)],
        "aic": None,
        "mean_hours": round(float(np.mean(durations)), 4),
    }


# --------------------------------------------------------------------------- #
# Spine attributes: terminal outcome + organ-failure flags
# --------------------------------------------------------------------------- #
def fit_outcome_rates(
    state_timeline: pl.DataFrame, outcomes: pl.DataFrame, *, min_n: int = 20
) -> EstimatorResult:
    """Terminal-outcome marginal + expired rate by peak support level (R2).

    The spine sampler (U6) draws each hospitalization's terminal outcome
    (survive/expire) from pack params, coupled to acuity. This emits both the
    overall outcome marginal and ``P(expired | peak support level reached)`` so
    the coupling is empirical, not assumed. ``outcomes`` carries
    ``hospitalization_id`` and ``outcome`` (``"alive"``/``"expired"``). The
    overall marginal is gated on the hospitalization count; each peak-level cell
    is gated on the hospitalizations that peaked at that level.
    """
    peak = state_timeline.group_by("hospitalization_id").agg(
        pl.col("support_level").max().alias("peak_level")
    )
    joined = peak.join(outcomes, on="hospitalization_id", how="inner")

    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []

    # Overall outcome marginal, gated as a single cell.
    n_total = joined.height
    n_expired = int(joined.filter(pl.col("outcome") == "expired").height)
    marginal = {"expired": n_expired / n_total, "alive": 1.0 - n_expired / n_total}
    surv_marg, marg_audit = suppress({"outcome": n_total}, {"outcome": marginal}, min_n=min_n)
    audit.extend(
        SuppressionRecord(cell=("outcome_marginal", r.cell), n=r.n, fallback_kind=r.fallback_kind)
        for r in marg_audit
    )
    if "outcome" in surv_marg:
        params["outcome_marginal"] = {
            k: round(float(v), 6) for k, v in surv_marg["outcome"].items()
        }

    # Expired rate conditioned on peak acuity, one gated cell per peak level.
    by_level = joined.group_by("peak_level").agg(
        pl.len().alias("n"),
        (pl.col("outcome") == "expired").sum().alias("n_expired"),
    )
    counts = {int(r["peak_level"]): int(r["n"]) for r in by_level.iter_rows(named=True)}
    rates = {
        int(r["peak_level"]): {
            "expired_rate": round(int(r["n_expired"]) / int(r["n"]), 6),
            "n_hospitalizations": int(r["n"]),
        }
        for r in by_level.iter_rows(named=True)
    }
    survived, level_audit = suppress(counts, rates, min_n=min_n)
    audit.extend(
        SuppressionRecord(
            cell=("expired_rate_by_peak_level", r.cell), n=r.n, fallback_kind=r.fallback_kind
        )
        for r in level_audit
    )
    if survived:
        params["expired_rate_by_peak_level"] = {
            str(level): rate for level, rate in survived.items()
        }
    return params, audit


def fit_flag_prevalence(state_timeline: pl.DataFrame, *, min_n: int = 20) -> EstimatorResult:
    """Per-support-level prevalence of each organ-failure flag (R2).

    The spine sampler (U6) draws organ-failure flags coupled to acuity from pack
    params; this emits ``P(flag | support level)`` for each of the four flags
    (respiratory / cardiovascular / renal / neuro) over the intervals observed
    at each level. Each level is gated on its interval count.
    """
    flags = ("resp_flag", "cv_flag", "renal_flag", "neuro_flag")
    present = [f for f in flags if f in state_timeline.columns]
    if not present:
        return {}, []

    by_level = state_timeline.group_by("support_level").agg(
        pl.len().alias("n"),
        *[pl.col(f).sum().alias(f) for f in present],
    )
    counts = {int(r["support_level"]): int(r["n"]) for r in by_level.iter_rows(named=True)}
    prevalence = {
        int(r["support_level"]): {f: round(int(r[f]) / int(r["n"]), 6) for f in present}
        for r in by_level.iter_rows(named=True)
    }
    survived, audit = suppress(counts, prevalence, min_n=min_n)
    audit = [
        SuppressionRecord(
            cell=("flag_prevalence_by_level", r.cell), n=r.n, fallback_kind=r.fallback_kind
        )
        for r in audit
    ]
    params: dict[str, object] = {}
    if survived:
        params["flag_prevalence_by_level"] = {str(level): prev for level, prev in survived.items()}
    return params, audit


# --------------------------------------------------------------------------- #
# Per-state physiology: AR1
# --------------------------------------------------------------------------- #
def fit_ar1_by_state(
    vitals_gridded: pl.DataFrame,
    state_timeline: pl.DataFrame,
    *,
    vitals: Sequence[str],
    min_n: int = 20,
) -> EstimatorResult:
    """First-order autoregression per (vital, support-level) (R2).

    ``vitals_gridded`` must carry columns ``hospitalization_id``,
    ``interval_idx``, ``vital_category``, ``value`` (one mean value per cell).
    The support level is forward-filled onto each vital interval via an as-of
    join; lag-1 pairs are formed only across **adjacent** grid intervals, and
    the pair is attributed to the state at the later interval. Each
    (vital, state) cell is gated at ``min_n`` lag-1 pairs.
    """
    state_sorted = state_timeline.sort("interval_idx")
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []

    for vital in vitals:
        vf = (
            vitals_gridded.filter(pl.col("vital_category") == vital)
            .select("hospitalization_id", "interval_idx", "value")
            .sort("interval_idx")
        )
        if vf.height == 0:
            continue
        # Forward-fill support level onto each vital interval (as-of backward).
        with_state = vf.join_asof(
            state_sorted.select("hospitalization_id", "interval_idx", "support_level"),
            on="interval_idx",
            by="hospitalization_id",
            strategy="backward",
        ).drop_nulls("support_level")

        # Lag-1 pairs across adjacent intervals within a hospitalization.
        with_state = with_state.sort("hospitalization_id", "interval_idx")
        prev_val = pl.col("value").shift(1).over("hospitalization_id")
        prev_iv = pl.col("interval_idx").shift(1).over("hospitalization_id")
        pairs = (
            with_state.with_columns(prev_val.alias("prev_value"), prev_iv.alias("prev_interval"))
            .filter(pl.col("interval_idx") - pl.col("prev_interval") == 1)
            .select("support_level", "prev_value", "value")
            .drop_nulls()
        )

        by_state: dict[int, tuple[_F64, _F64]] = {}
        for state in pairs["support_level"].unique().to_list():
            sub = pairs.filter(pl.col("support_level") == state)
            by_state[int(state)] = (
                sub["prev_value"].to_numpy(),
                sub["value"].to_numpy(),
            )

        counts = {state: len(x) for state, (x, _y) in by_state.items()}
        fits = {state: _fit_ar1(x_prev, x_curr) for state, (x_prev, x_curr) in by_state.items()}
        survived, records = suppress(counts, fits, min_n=min_n)
        audit.extend(
            SuppressionRecord(cell=(vital, r.cell), n=r.n, fallback_kind=r.fallback_kind)
            for r in records
        )
        if survived:
            params[f"{vital}_ar1_by_state"] = {str(state): fit for state, fit in survived.items()}
    return params, audit


def _fit_ar1(x_prev: _F64, x_curr: _F64) -> dict[str, float]:
    """OLS fit of x_t = mean + phi*(x_{t-1}-mean) + eps; phi clamped stationary."""
    mean = float(np.mean(np.concatenate([x_prev, x_curr])))
    cp = x_prev - mean
    cc = x_curr - mean
    denom = float(np.sum(cp * cp))
    phi = float(np.sum(cp * cc) / denom) if denom > 0 else 0.0
    phi = max(-_PHI_CLAMP, min(_PHI_CLAMP, phi))
    resid = cc - phi * cp
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    return {
        "phi": round(phi, 6),
        "sigma": round(sigma, 6),
        "mean": round(mean, 6),
    }


# --------------------------------------------------------------------------- #
# Lab copula
# --------------------------------------------------------------------------- #
def fit_lab_copula(
    labs_gridded: pl.DataFrame,
    *,
    n_hospitalizations: int,
    min_n: int = 20,
) -> EstimatorResult:
    """Co-measurement Spearman correlation + per-lab log-marginals + presence.

    ``labs_gridded`` carries ``hospitalization_id``, ``interval_idx``,
    ``lab_category``, ``value``. Correlation is computed over lab pairs measured
    in the **same** (hospitalization, interval) window, then projected to the
    nearest positive-definite correlation matrix. Per-lab marginals are fit on
    ``log1p`` values (labs are heavy-tailed and non-negative). Each lab is gated
    at ``min_n`` observations; presence rate is the fraction of hospitalizations
    with at least one measurement.
    """
    labs = labs_gridded.drop_nulls("value")
    per_lab_counts = labs.group_by("lab_category").len().rename({"len": "n"})
    counts = {r["lab_category"]: int(r["n"]) for r in per_lab_counts.iter_rows(named=True)}

    marginals: dict[str, dict[str, float]] = {}
    for lab, sub in labs.group_by("lab_category"):
        name = lab[0] if isinstance(lab, tuple) else lab
        vals = sub["value"].to_numpy()
        logv = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        marginals[name] = {
            "log_mean": round(float(np.mean(logv)), 6),
            "log_sd": round(float(np.std(logv, ddof=1)) if logv.size > 1 else 0.0, 6),
        }

    survived_marginals, audit = suppress(counts, marginals, min_n=min_n)
    lab_order = sorted(survived_marginals)

    presence = {}
    for lab in lab_order:
        n_hosp_with = (
            labs.filter(pl.col("lab_category") == lab)
            .select(pl.col("hospitalization_id").n_unique())
            .item()
        )
        presence[lab] = (
            round(n_hosp_with / n_hospitalizations, 6) if n_hospitalizations > 0 else 0.0
        )

    correlation = _co_measurement_correlation(labs, lab_order)

    params: dict[str, object] = {
        "lab_order": lab_order,
        "lab_marginals": survived_marginals,
        "lab_presence": presence,
        "lab_correlation": correlation,
    }
    return params, audit


def _co_measurement_correlation(labs: pl.DataFrame, lab_order: Sequence[str]) -> list[list[float]]:
    """Spearman correlation over co-measured labs -> nearest-PD matrix."""
    k = len(lab_order)
    if k == 0:
        return []
    if k == 1:
        return [[1.0]]

    wide = (
        labs.filter(pl.col("lab_category").is_in(list(lab_order)))
        .group_by("hospitalization_id", "interval_idx", "lab_category")
        .agg(pl.col("value").mean().alias("value"))
        .pivot(on="lab_category", index=["hospitalization_id", "interval_idx"], values="value")
    )
    # Rank each lab column (Spearman = Pearson on ranks), keeping NaN for gaps.
    cols = [c for c in lab_order if c in wide.columns]
    mat = wide.select(cols).to_numpy().astype(float)

    corr = np.eye(k)
    index = {name: i for i, name in enumerate(lab_order)}
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            xa = mat[:, a]
            xb = mat[:, b]
            both = ~np.isnan(xa) & ~np.isnan(xb)
            if both.sum() >= 3:
                rho, _ = stats.spearmanr(xa[both], xb[both])
                if np.isfinite(rho):
                    ia, ib = index[cols[a]], index[cols[b]]
                    corr[ia, ib] = corr[ib, ia] = float(rho)

    pd_corr = nearest_positive_definite_correlation(corr)
    return [[round(float(v), 6) for v in row] for row in pd_corr]


def nearest_positive_definite_correlation(matrix: _F64) -> _F64:
    """Project a symmetric matrix to the nearest PD correlation matrix.

    Clips negative eigenvalues to a small positive floor, reconstructs, then
    rescales to unit diagonal. Sufficient for sampling a Gaussian copula; not
    the full Higham iteration, but PD and correlation-normalized.
    """
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, a_min=1e-6, a_max=None)
    rebuilt = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(rebuilt), a_min=1e-12, a_max=None))
    normalized = rebuilt / np.outer(d, d)
    np.fill_diagonal(normalized, 1.0)
    result: _F64 = ((normalized + normalized.T) / 2.0).astype(np.float64)
    return result


# --------------------------------------------------------------------------- #
# Infusion hazards
# --------------------------------------------------------------------------- #
def fit_infusion_hazards(
    mac_gridded: pl.DataFrame,
    *,
    min_n: int = 20,
) -> EstimatorResult:
    """Per-drug start/stop hazard per interval (R2).

    ``mac_gridded`` carries ``hospitalization_id``, ``interval_idx``,
    ``med_category`` for every interval a continuous drug is active. Start
    hazard = starts / (starts + off-intervals-at-risk) approximated as
    starts / hospitalization-exposures; stop hazard = stops / on-intervals.
    Each drug is gated at ``min_n`` on-intervals. Doses are intentionally not
    emitted here (dose marginals belong to the med marginal block).
    """
    active = mac_gridded.select("hospitalization_id", "interval_idx", "med_category").unique()
    active = active.sort("hospitalization_id", "med_category", "interval_idx")

    prev_iv = pl.col("interval_idx").shift(1).over(["hospitalization_id", "med_category"])
    marked = active.with_columns(
        ((pl.col("interval_idx") - prev_iv != 1) | prev_iv.is_null()).alias("_is_start")
    )

    per_drug = marked.group_by("med_category").agg(
        pl.len().alias("on_intervals"),
        pl.col("_is_start").sum().alias("starts"),
    )

    counts = {r["med_category"]: int(r["on_intervals"]) for r in per_drug.iter_rows(named=True)}
    hazards: dict[str, dict[str, float]] = {}
    for r in per_drug.iter_rows(named=True):
        drug = r["med_category"]
        on = int(r["on_intervals"])
        starts = int(r["starts"])
        # A "run" of consecutive on-intervals begins at each start; the stop
        # hazard is (#runs) / (on-intervals) = mean 1/duration per interval.
        stop_hazard = starts / on if on > 0 else 0.0
        hazards[drug] = {
            "stop_hazard": round(stop_hazard, 6),
            "mean_run_intervals": round(on / starts, 4) if starts > 0 else 0.0,
        }

    survived, audit = suppress(counts, hazards, min_n=min_n)
    params: dict[str, object] = {"infusion_hazards": survived}
    return params, audit

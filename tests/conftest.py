"""Shared pytest fixtures for the CLIFForge test suite."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.fit.param_pack import ParamPack


@pytest.fixture
def seed() -> int:
    """The canonical test seed (mirrors the demo run's --seed 42)."""
    return 42


def build_synthetic_pack() -> ParamPack:
    """A minimal but complete parameter pack: every fitted block the generators
    read, with no real data. Shared by the orchestrator/CLI and eval tests so the
    fixture never drifts between them."""

    def six(mean: float) -> dict[str, dict[str, float]]:
        return {str(s): {"mean": mean - 3 * s, "phi": 0.4, "sigma": 3.0} for s in range(6)}

    return ParamPack(
        manifest={
            "clif_version": "2.1.0",
            "pack_version": "1.0",
            "fit_source": {"dataset_id": "synthetic-test", "commit": "none"},
            "suppression_audit": [],
        },
        tables={
            "patient": {
                "params": {
                    "race_category_marginal": {"White": 0.6, "Unknown": 0.4},
                    "ethnicity_category_marginal": {"Non-Hispanic": 0.7, "Unknown": 0.3},
                    "sex_category_marginal": {"Female": 0.5, "Male": 0.5},
                }
            },
            "hospitalization": {
                "params": {
                    "admission_type_category_marginal": {"ed": 0.8, "direct": 0.2},
                    "discharge_category_marginal": {"Home": 0.7, "Expired": 0.3},
                }
            },
            "vitals": {
                "params": {
                    f"{v}_ar1_by_state": six(base)
                    for v, base in {
                        "heart_rate": 90,
                        "sbp": 130,
                        "dbp": 75,
                        "map": 90,
                        "respiratory_rate": 20,
                        "spo2": 100,
                        "temp_c": 38,
                    }.items()
                }
            },
            "labs": {
                "params": {
                    "lab_order": ["creatinine", "bun", "sodium"],
                    "lab_correlation": [[1.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 1.0]],
                    "lab_marginals": {
                        "creatinine": {"log_mean": 0.79, "log_sd": 0.4},
                        "bun": {"log_mean": 2.8, "log_sd": 0.5},
                        "sodium": {"log_mean": 4.95, "log_sd": 0.03},
                    },
                    "lab_presence": {"creatinine": 0.8, "bun": 0.6, "sodium": 0.9},
                }
            },
            "medication_admin_continuous": {
                "params": {
                    "infusion_hazards": {
                        "norepinephrine": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                        "propofol": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                    }
                }
            },
            "spine": {
                "params": {
                    "state_model": {"grid_step_hours": 1.0, "horizon_intervals": 72},
                    "support_level_states": [0, 1, 2, 3, 4, 5],
                    "support_level_start_dist": {
                        "0": 0.3,
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.15,
                        "4": 0.1,
                        "5": 0.05,
                    },
                    "support_level_transition_matrix": {
                        "0": {"1": 0.2, "3": 0.3, "discharge": 0.5},
                        "1": {"2": 0.3, "0": 0.2, "discharge": 0.5},
                        "2": {"3": 0.4, "1": 0.2, "discharge": 0.4},
                        "3": {"4": 0.3, "2": 0.3, "discharge": 0.4},
                        "4": {"5": 0.3, "3": 0.3, "discharge": 0.4},
                        "5": {"4": 0.4, "discharge": 0.6},
                    },
                    "support_level_sojourn": {
                        str(s): {"family": "empirical_mean", "params": [2.0], "mean_hours": 2.0}
                        for s in range(6)
                    },
                    "flag_prevalence_by_level": {
                        "0": {
                            "resp_flag": 0.0,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.0,
                        },
                        "1": {
                            "resp_flag": 0.1,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.1,
                        },
                        "2": {
                            "resp_flag": 0.4,
                            "cv_flag": 0.1,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.2,
                        },
                        "3": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.2,
                            "renal_flag": 0.1,
                            "neuro_flag": 0.3,
                        },
                        "4": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.8,
                            "renal_flag": 0.3,
                            "neuro_flag": 0.3,
                        },
                        "5": {
                            "resp_flag": 0.8,
                            "cv_flag": 0.9,
                            "renal_flag": 0.8,
                            "neuro_flag": 0.4,
                        },
                    },
                    "outcome_marginal": {"alive": 0.8, "expired": 0.2},
                    "expired_rate_by_peak_level": {
                        str(s): {"expired_rate": 0.05 + 0.09 * s} for s in range(6)
                    },
                }
            },
        },
    )


@pytest.fixture
def pack() -> ParamPack:
    """The shared synthetic parameter pack (no real data)."""
    return build_synthetic_pack()


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """A seeded ``Generator(PCG64)`` threaded through tests (R22).

    Every generator in the pipeline is constructed this way so a single seed
    reproduces byte-identical output.
    """
    return np.random.Generator(np.random.PCG64(seed))

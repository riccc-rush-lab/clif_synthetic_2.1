"""Local fixtures for the ``clifforge.fit`` test suite (U4).

Kept separate from the root ``tests/conftest.py`` to avoid collisions with
parallel unit work on that shared file.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def sample_manifest() -> dict[str, Any]:
    """A minimal, schema-complete, PHI-free manifest (KTD-2)."""
    return {
        "pack_version": "1.0",
        "clif_version": "2.1.0",
        "fit_source": {"dataset_id": "MIMIC-IV-Ext-CLIF-sample", "commit": "abc1234"},
        "suppression_audit": {
            "patient": {
                "cells_considered": 10,
                "cells_suppressed": 1,
                "fallback_kinds": {"prior": 1},
            }
        },
        "tables": {
            "patient": {"fitted": True, "source": "MIMIC-IV-Ext-CLIF v1.1.0"},
        },
    }


@pytest.fixture
def sample_table_blocks() -> dict[str, dict[str, Any]]:
    """A couple of small, legal (non-leaking) table parameter blocks."""
    return {
        "patient": {
            "n_records": 500,
            "fitted": True,
            "params": {
                "sex_category_marginal": {"Male": 0.51, "Female": 0.49},
                "age_quantile_bin_edges": [18.0, 40.0, 55.0, 65.0, 75.0, 90.0],
            },
        },
        "vitals": {
            "n_records": 300,
            "fitted": True,
            "params": {
                "heart_rate_ar1": {"phi": 0.82, "sigma": 4.1, "mean": 84.0},
            },
        },
    }

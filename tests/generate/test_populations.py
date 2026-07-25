"""Unit tests for the reproducible Chicago-population derivation (U5)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.populations import (
    CHICAGO_RACE_TARGET,
    derive_chicago_population,
)


def _base_pack() -> ParamPack:
    return ParamPack(
        manifest={"clif_version": "2.1.0"},
        tables={
            "patient": {"params": {"race_category_marginal": {"White": 1.0}}},
            "hospitalization": {"params": {"admission_type_category_marginal": {"ed": 1.0}}},
            "medication_admin_continuous": {"params": {"infusion_hazards": {}}},
        },
    )


def _real_dir(tmp: Path) -> Path:
    pl.DataFrame({"age_at_admission": list(range(20, 90))}).write_parquet(
        tmp / "clif_hospitalization.parquet"
    )
    pl.DataFrame(
        {
            "med_category": (
                ["norepinephrine"] * 10
                + ["sodium chloride"] * 8
                + ["dextrose"] * 5  # -> dextrose_other
                + ["albumin_infusion"] * 3  # -> albumin
                + ["acetaminophen"] * 4  # dropped
            ),
            "mar_action_category": ["start"] * 30,
        }
    ).write_parquet(tmp / "clif_medication_admin_continuous.parquet")
    return tmp


def test_demographics_reweighted_to_chicago(tmp_path: Path) -> None:
    out = derive_chicago_population(_base_pack(), _real_dir(tmp_path)).tables
    assert out["patient"]["params"]["race_category_marginal"] == CHICAGO_RACE_TARGET
    assert out["patient"]["params"]["ethnicity_category_marginal"]["Hispanic"] == 0.29


def test_age_quantiles_are_real_deciles_shifted(tmp_path: Path) -> None:
    out = derive_chicago_population(_base_pack(), _real_dir(tmp_path), age_shift_years=2.5).tables
    q = out["hospitalization"]["params"]["age_at_admission_quantiles"]
    assert len(q) == 11
    assert q == sorted(q)  # monotonic
    # ages 20..89; the median decile (~54.5) is shifted up by 2.5.
    assert abs(q[5] - (54.5 + 2.5)) < 1.5


def test_med_marginal_normalized_and_dropped(tmp_path: Path) -> None:
    out = derive_chicago_population(_base_pack(), _real_dir(tmp_path)).tables
    marg = out["medication_admin_continuous"]["params"]["med_category_marginal"]
    assert "dextrose_other" in marg and "dextrose" not in marg  # renamed
    assert "albumin" in marg and "albumin_infusion" not in marg  # renamed
    assert "acetaminophen" not in marg  # dropped
    assert "sodium_chloride" in marg  # space -> underscore
    assert abs(sum(marg.values()) - 1.0) < 1e-9  # proper distribution


def test_input_pack_not_mutated(tmp_path: Path) -> None:
    base = _base_pack()
    derive_chicago_population(base, _real_dir(tmp_path))
    assert base.tables["patient"]["params"]["race_category_marginal"] == {"White": 1.0}

"""Unit tests for declarative variant specs (U1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from clifforge.demo import demo_pack
from clifforge.variants import SpecError, VariantSpec, load_spec, spec_to_pack


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "variant.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_minimal_spec_uses_master_defaults(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, 'name = "just-a-name"\n'))
    master = VariantSpec()
    assert spec.name == "just-a-name"
    assert (spec.imv, spec.mortality_scale, spec.vaso_frac, spec.n) == (
        master.imv,
        master.mortality_scale,
        master.vaso_frac,
        master.n,
    )


def test_full_spec_overrides_are_parsed(tmp_path: Path) -> None:
    spec = load_spec(
        _write(
            tmp_path,
            """
            name = "high-acuity"
            n = 20000
            seed = 7
            [demographics]
            age_shift = 5.0
            hispanic_frac = 0.45
            [rates]
            imv = 0.55
            mortality_scale = 1.4
            vaso_frac = 0.45
            crrt_prob = 0.06
            prone_severe = 0.05
            """,
        )
    )
    assert (spec.n, spec.seed, spec.age_shift, spec.hispanic_frac) == (20000, 7, 5.0, 0.45)
    assert (spec.imv, spec.mortality_scale, spec.vaso_frac) == (0.55, 1.4, 0.45)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="unknown key"):
        load_spec(_write(tmp_path, 'name = "x"\nbogus = 1\n'))
    with pytest.raises(SpecError, match="rates"):
        load_spec(_write(tmp_path, "[rates]\nnot_a_rate = 0.5\n"))


def test_out_of_range_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="mortality_scale"):
        load_spec(_write(tmp_path, "[rates]\nmortality_scale = -1\n"))
    with pytest.raises(SpecError, match="imv"):
        load_spec(_write(tmp_path, "[rates]\nimv = 1.5\n"))


def test_spec_overrides_reach_the_pack_and_do_not_mutate_input() -> None:
    base = demo_pack()
    before = base.tables["spine"]["params"]["expired_rate_by_peak_level"]["4"]["expired_rate"]
    spec = VariantSpec(name="v", imv=0.60, vaso_frac=0.50, crrt_prob=0.50, mortality_scale=1.0)
    out = spec_to_pack(spec, base).tables
    # Directly-propagating rate overrides (the imv/mortality effects need a base pack
    # with a real peak profile and are covered end-to-end in the U6 regression test).
    assert out["crrt_therapy"]["params"]["crrt_prob"] == 0.50
    assert out["spine"]["params"]["flag_target_prevalence"]["cv_flag"] == 0.50
    # input pack untouched
    assert (
        base.tables["spine"]["params"]["expired_rate_by_peak_level"]["4"]["expired_rate"] == before
    )


def test_race_override_applies_without_real_data() -> None:
    spec = VariantSpec(name="v", race_target={"White": 0.5, "Black or African American": 0.5})
    out = spec_to_pack(spec, demo_pack()).tables
    assert out["patient"]["params"]["race_category_marginal"] == {
        "White": 0.5,
        "Black or African American": 0.5,
    }

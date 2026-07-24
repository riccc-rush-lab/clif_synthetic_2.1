"""Tests for SDMetrics fidelity scoring (U24, R24).

Two independently-seeded synthetic datasets (one standing in for "real") exercise
the report in CI: scores bounded in [0,1], identifier/datetime columns excluded,
empty tables refused rather than scored perfect, and — the discriminating check —
a well-matched table scoring higher than a distorted or shuffled control.
"""

from __future__ import annotations

import polars as pl
import pytest

from clifforge.eval.fidelity import TableFidelity, fidelity_report, table_fidelity
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset


def _tables(pack: ParamPack, seed: int, n: int = 150) -> dict:
    return generate_dataset(pack, n_patients=n, seed=seed).tables


@pytest.fixture
def synthetic(pack: ParamPack) -> dict:
    return _tables(pack, 1)


@pytest.fixture
def real(pack: ParamPack) -> dict:
    return _tables(pack, 2)


def test_report_covers_tables_and_scores_are_bounded(synthetic: dict, real: dict) -> None:
    report = fidelity_report(synthetic, real)
    assert report  # at least some tables scored
    for name, result in report.items():
        assert isinstance(result, TableFidelity)
        assert 0.0 <= result.quality_score <= 1.0, name
        assert 0.0 <= result.column_shape_score <= 1.0, name
        if result.column_pair_score is not None:
            assert 0.0 <= result.column_pair_score <= 1.0, name
        assert result.n_columns > 0


def test_same_distribution_scores_high(synthetic: dict, real: dict) -> None:
    # Both draws come from the same parameter pack, so vitals marginals should
    # match closely.
    result = table_fidelity("vitals", real["vitals"], synthetic["vitals"])
    assert result is not None
    assert result.quality_score > 0.8


def test_distorted_table_scores_lower(synthetic: dict, real: dict) -> None:
    good = table_fidelity("vitals", real["vitals"], synthetic["vitals"])
    distorted = synthetic["vitals"].with_columns(pl.col("vital_value") * 5.0 + 500.0)
    bad = table_fidelity("vitals", real["vitals"], distorted)
    assert good is not None and bad is not None
    assert bad.quality_score < good.quality_score
    assert bad.column_shape_score < good.column_shape_score  # marginals moved


def test_shuffled_control_breaks_pair_trends(synthetic: dict, real: dict) -> None:
    # Independently permuting each column preserves marginals but destroys the
    # joint structure, so the column-pair component must drop.
    table = "medication_admin_continuous"
    good = table_fidelity(table, real[table], synthetic[table])
    assert good is not None and good.column_pair_score is not None
    shuffled = synthetic[table].select(
        [pl.col(c).shuffle(seed=i) for i, c in enumerate(synthetic[table].columns)]
    )
    bad = table_fidelity(table, real[table], shuffled)
    assert bad is not None and bad.column_pair_score is not None
    assert bad.column_pair_score < good.column_pair_score


def test_identifier_and_datetime_columns_are_excluded(synthetic: dict, real: dict) -> None:
    # patient has patient_id (surrogate key) + category columns; the id must not
    # be scored, so n_columns counts only the real distributions.
    result = table_fidelity("patient", real["patient"], synthetic["patient"])
    assert result is not None
    scored_max = len([c for c in synthetic["patient"].columns if not c.endswith("_id")])
    assert result.n_columns <= scored_max


def test_free_text_columns_are_excluded_from_categorical_scoring() -> None:
    # A per-row surrogate code (product_code, a stringified value, a string
    # timestamp) shares no values between two draws. Scoring it as a category
    # would measure string reuse, not fidelity, and tank an otherwise good table.
    n = 200
    real = pl.DataFrame({"cat": ["a", "b"] * (n // 2), "code": [f"r{i}" for i in range(n)]})
    synth = pl.DataFrame({"cat": ["a", "b"] * (n // 2), "code": [f"s{i}" for i in range(n)]})
    result = table_fidelity("t", real, synth)
    assert result is not None
    assert result.n_columns == 1  # only the bounded-vocabulary column is scored
    assert result.quality_score > 0.95  # not dragged down by the free-text column


def test_empty_table_is_not_scored(synthetic: dict, real: dict) -> None:
    empty = synthetic["vitals"].head(0)
    assert table_fidelity("vitals", real["vitals"], empty) is None
    assert table_fidelity("vitals", empty, synthetic["vitals"]) is None


def test_is_deterministic(synthetic: dict, real: dict) -> None:
    assert fidelity_report(synthetic, real) == fidelity_report(synthetic, real)

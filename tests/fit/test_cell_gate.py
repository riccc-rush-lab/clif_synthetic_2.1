"""Tests for ``clifforge.fit.cell_gate`` (U4): the n >= 20 suppression gate."""

from __future__ import annotations

from clifforge.fit.cell_gate import suppress


def test_cell_with_n19_is_suppressed_and_fallback_recorded() -> None:
    counts = {"male": 19}
    params = {"male": 0.5}
    prior = {"male": 0.4}

    surviving, audit = suppress(counts, params, min_n=20, prior=prior)

    assert surviving == {"male": 0.4}
    assert len(audit) == 1
    record = audit[0]
    assert record.cell == "male"
    assert record.n == 19
    assert record.fallback_kind == "prior"


def test_cell_with_n20_is_kept_as_fitted() -> None:
    counts = {"male": 20}
    params = {"male": 0.5}

    surviving, audit = suppress(counts, params, min_n=20)

    assert surviving == {"male": 0.5}
    assert audit == []


def test_default_min_n_is_20() -> None:
    counts = {"a": 20, "b": 19}
    params = {"a": 1.0, "b": 2.0}

    surviving, audit = suppress(counts, params, prior={"b": 0.0})

    assert surviving["a"] == 1.0
    assert surviving["b"] == 0.0
    assert audit[0].fallback_kind == "prior"


def test_missing_cell_in_counts_treated_as_n_zero() -> None:
    counts: dict[str, int] = {}
    params = {"rare_stratum": 1.0}
    prior = {"rare_stratum": 0.1}

    surviving, audit = suppress(counts, params, min_n=20, prior=prior)

    assert surviving == {"rare_stratum": 0.1}
    assert audit[0].n == 0


def test_coarser_aggregate_fallback_preferred_over_prior() -> None:
    counts = {("icu", "male"): 5, ("icu", None): 40}
    params = {("icu", "male"): 0.7, ("icu", None): 0.5}
    prior = {("icu", "male"): 0.2}

    def coarsen(cell: tuple[str, str | None]) -> tuple[str, str | None]:
        return (cell[0], None)

    surviving, audit = suppress(counts, params, min_n=20, prior=prior, coarsen=coarsen)

    assert surviving[("icu", "male")] == 0.5
    assert audit[0].fallback_kind == "coarser_aggregate"
    assert audit[0].fallback_source == ("icu", None)


def test_coarser_aggregate_itself_sub_threshold_falls_through_to_prior() -> None:
    counts = {("icu", "male"): 5, ("icu", None): 10}
    params = {("icu", "male"): 0.7, ("icu", None): 0.5}
    prior = {("icu", "male"): 0.2}

    def coarsen(cell: tuple[str, str | None]) -> tuple[str, str | None]:
        return (cell[0], None)

    surviving, audit = suppress(counts, params, min_n=20, prior=prior, coarsen=coarsen)

    assert surviving[("icu", "male")] == 0.2
    assert audit[0].fallback_kind == "prior"


def test_no_fallback_available_cell_is_dropped_not_emitted() -> None:
    counts = {"rare": 3}
    params = {"rare": 0.9}

    surviving, audit = suppress(counts, params, min_n=20)

    assert "rare" not in surviving
    assert audit[0].fallback_kind == "none"
    assert audit[0].fallback_source is None


def test_suppress_is_pure_and_deterministic() -> None:
    counts = {"a": 19, "b": 25}
    params = {"a": 1.0, "b": 2.0}
    prior = {"a": 0.0}

    result_1 = suppress(dict(counts), dict(params), min_n=20, prior=dict(prior))
    result_2 = suppress(dict(counts), dict(params), min_n=20, prior=dict(prior))

    assert result_1 == result_2
    # inputs are untouched
    assert counts == {"a": 19, "b": 25}
    assert params == {"a": 1.0, "b": 2.0}

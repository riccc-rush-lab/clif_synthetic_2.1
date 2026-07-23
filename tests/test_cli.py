"""Smoke tests for the CLI scaffold (U1)."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.cli import build_parser, main


def test_parser_program_name() -> None:
    parser = build_parser()
    assert parser.prog == "clif-forge"


def test_no_command_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "generate" in out
    assert "fit" in out


def test_generate_requires_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate"])  # missing --n-patients / --out


def test_generate_parses_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["generate", "--n-patients", "100", "--seed", "42", "--out", "./out"]
    )
    assert args.command == "generate"
    assert args.n_patients == 100
    assert args.seed == 42


def test_rng_fixture_is_seed_reproducible(rng: np.random.Generator, seed: int) -> None:
    first = rng.integers(0, 1_000_000, size=5)
    fresh = np.random.Generator(np.random.PCG64(seed))
    assert (first == fresh.integers(0, 1_000_000, size=5)).all()

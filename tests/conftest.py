"""Shared pytest fixtures for the CLIFForge test suite."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.demo import demo_pack
from clifforge.fit.param_pack import ParamPack

#: Back-compat alias — the demo pack now ships in the package (clifforge.demo) so
#: it is the single source of truth for both the CLI (--demo) and the test suite.
build_synthetic_pack = demo_pack


@pytest.fixture
def seed() -> int:
    """The canonical test seed (mirrors the demo run's --seed 42)."""
    return 42


@pytest.fixture
def pack() -> ParamPack:
    """The shared synthetic parameter pack (no real data)."""
    return demo_pack()


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """A seeded ``Generator(PCG64)`` threaded through tests (R22).

    Every generator in the pipeline is constructed this way so a single seed
    reproduces byte-identical output.
    """
    return np.random.Generator(np.random.PCG64(seed))

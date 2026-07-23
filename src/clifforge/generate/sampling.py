"""Seedable categorical sampling shared by the table generators (U7+).

The Tier 1+ generators repeatedly draw a category (race, discharge disposition,
admission type, location, …) from a fitted marginal in the parameter pack. This
one helper centralizes that draw so every table samples the same way: a single
``rng.random()`` inverse-CDF over the marginal's keys in a **canonical (sorted)
order**, which makes a draw reproducible byte-for-byte under a fixed seed (R22)
regardless of the dict's insertion order when the pack was loaded from JSON.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["categorical"]


def categorical(marginal: Mapping[str, float], rng: np.random.Generator) -> str:
    """Draw one key from a (possibly unnormalized) category ``marginal``.

    Keys are sorted before the cumulative search so the mapping from the drawn
    uniform to a category is stable across pack loads (dict insertion order must
    not change the result). The marginal is renormalized on the fly, so callers
    may pass a conditioned sub-marginal (e.g. discharge categories with the death
    category removed) without pre-normalizing. Raises on non-positive total mass.
    """
    keys = sorted(marginal)
    if not keys:
        raise ValueError("categorical marginal is empty")
    probs = np.asarray([marginal[k] for k in keys], dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("categorical marginal has non-positive total mass")
    cumulative = np.cumsum(probs / total)
    idx = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return keys[min(idx, len(keys) - 1)]

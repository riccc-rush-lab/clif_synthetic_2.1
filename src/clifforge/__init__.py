"""CLIFForge — a fully synthetic CLIF 2.1 ICU dataset generator.

CLIFForge generates ICU datasets in exact CLIF 2.1 format. Its output is
synthetic and openly redistributable: no patient-level record ever leaves the
one-time fit stage — only aggregate parameters (gated at n >= 20) are emitted,
and generation runs entirely offline from a versioned parameter pack.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]

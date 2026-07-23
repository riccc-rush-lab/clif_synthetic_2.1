"""CLIFForge command-line interface.

Two subcommands:

* ``generate`` — sample a synthetic CLIF 2.1 dataset offline from a parameter
  pack (implemented in U21).
* ``fit`` — the one-time fit stage that builds a parameter pack over real
  CLIF-MIMIC (implemented in U5).

At the scaffold stage both are argument-parsing stubs; they parse and validate
their flags but do not yet run a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from clifforge import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``clif-forge`` argument parser (generate + fit)."""
    parser = argparse.ArgumentParser(
        prog="clif-forge",
        description="CLIFForge — generate fully synthetic CLIF 2.1 ICU datasets.",
    )
    parser.add_argument(
        "--version", action="version", version=f"clif-forge {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="{generate,fit}")

    generate = sub.add_parser(
        "generate", help="Generate a synthetic CLIF 2.1 dataset (offline, no real data)."
    )
    generate.add_argument(
        "--n-patients", type=int, required=True, help="Number of synthetic patients."
    )
    generate.add_argument(
        "--seed", type=int, default=42, help="Seed for byte-identical reproducible output."
    )
    generate.add_argument(
        "--out", required=True, help="Output directory for the generated dataset."
    )

    fit = sub.add_parser(
        "fit", help="Fit a parameter pack over real CLIF-MIMIC (one-time, requires real data)."
    )
    fit.add_argument(
        "--real-dir", required=True, help="Directory of real CLIF parquet files."
    )
    fit.add_argument(
        "--out", required=True, help="Output directory for the versioned parameter pack."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Real implementations land in U21 (generate) and U5 (fit).
    print(
        f"clif-forge {args.command}: not yet implemented (scaffold stage).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

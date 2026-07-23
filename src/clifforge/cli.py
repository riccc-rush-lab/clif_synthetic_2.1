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
    parser.add_argument("--version", action="version", version=f"clif-forge {__version__}")
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
    generate.add_argument(
        "--pack",
        default="data/param_packs/mimic",
        help="Directory of the parameter pack to sample from.",
    )

    fit = sub.add_parser(
        "fit", help="Fit a parameter pack over real CLIF-MIMIC (one-time, requires real data)."
    )
    fit.add_argument("--real-dir", required=True, help="Directory of real CLIF parquet files.")
    fit.add_argument(
        "--out", required=True, help="Output directory for the versioned parameter pack."
    )

    return parser


def _run_generate(args: argparse.Namespace) -> int:
    """Generate + gate + write a synthetic dataset; nonzero on any failure (R25)."""
    from clifforge.conformance.gate import ConformanceError
    from clifforge.fit.param_pack import ParamPack
    from clifforge.generate.orchestrator import generate_dataset, write_dataset

    try:
        pack = ParamPack.load(args.pack)
        dataset = generate_dataset(pack, n_patients=args.n_patients, seed=args.seed)
        written = write_dataset(dataset, args.out)
    except ConformanceError as exc:
        print(f"clif-forge generate: conformance failure -> {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary: any failure is a clean nonzero exit
        # A malformed-but-loadable pack (KeyError), a version mismatch, or an
        # unwritable --out (OSError/FileExistsError) must report cleanly, not crash.
        print(f"clif-forge generate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"clif-forge generate: wrote {len(written)} files to {args.out}")
    return 0


def _run_fit(args: argparse.Namespace) -> int:
    """Fit a parameter pack over real CLIF-MIMIC (U5)."""
    from clifforge.fit.run_fit import run_fit

    try:
        run_fit(args.real_dir, args.out)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: any failure is a clean nonzero exit
        print(f"clif-forge fit: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"clif-forge fit: wrote parameter pack to {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _run_generate(args)
    if args.command == "fit":
        return _run_fit(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

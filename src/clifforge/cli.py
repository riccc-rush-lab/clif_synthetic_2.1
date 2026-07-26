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
import os
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
        "--n-patients",
        type=int,
        default=None,
        help="Number of synthetic patients (overrides a spec's n).",
    )
    generate.add_argument(
        "--seed", type=int, default=None, help="Seed for byte-identical reproducible output."
    )
    generate.add_argument(
        "--out", required=True, help="Output directory for the generated dataset."
    )
    generate.add_argument(
        "--pack",
        default=None,
        help="Directory of a fitted parameter pack to sample from.",
    )
    generate.add_argument(
        "--demo",
        action="store_true",
        help="Use the built-in synthetic demo pack (no real data or fit required). "
        "Structurally valid CLIF 2.1 output for testing; not statistically calibrated.",
    )
    generate.add_argument(
        "--spec", default=None, help="A variant spec (TOML) describing a derivative to generate."
    )
    generate.add_argument(
        "--preset", default=None, help="A shipped preset variant name (see clifforge.variants)."
    )
    generate.add_argument(
        "--base-pack",
        default="base_pack",
        help="Base pack a --spec/--preset derives from (default: the shipped shareable base pack).",
    )
    generate.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Encounters generated per in-memory batch (the RAM dial). Cohorts larger "
        "than this stream to disk in batches so peak memory stays bounded; smaller "
        "values use less RAM on modest machines (a touch slower). Output is identical "
        "regardless of chunk size. Default 10000.",
    )
    generate.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Cap CPU threads (the compute dial). Fewer threads use less CPU on shared "
        "or low-core machines. Default: all available cores.",
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
    import dataclasses

    from clifforge.conformance.gate import ConformanceError
    from clifforge.demo import demo_pack
    from clifforge.fit.param_pack import ParamPack
    from clifforge.generate.orchestrator import (
        generate_dataset,
        generate_streaming,
        write_dataset,
    )
    from clifforge.manifest import write_manifest
    from clifforge.variants import load_preset, load_spec, spec_to_pack

    use_spec = args.spec is not None or args.preset is not None
    if not use_spec and not args.demo and args.pack is None:
        print(
            "clif-forge generate: provide --spec/--preset (a variant recipe), --pack <dir> "
            "(a fitted pack), or --demo (built-in synthetic pack, no real data required).",
            file=sys.stderr,
        )
        return 1

    spec = None
    try:
        if use_spec:
            spec = load_spec(args.spec) if args.spec is not None else load_preset(args.preset)
            base = ParamPack.load(args.base_pack)
            pack = spec_to_pack(spec, base)  # no real_dir -> demographic-override path
            n_patients = args.n_patients if args.n_patients is not None else spec.n
            seed = args.seed if args.seed is not None else spec.seed
        else:
            pack = demo_pack() if args.demo else ParamPack.load(args.pack)
            if args.n_patients is None:
                print(
                    "clif-forge generate: --n-patients is required without --spec/--preset.",
                    file=sys.stderr,
                )
                return 1
            n_patients = args.n_patients
            seed = args.seed if args.seed is not None else 42
        chunk_size = getattr(args, "chunk_size", 10_000)
        if n_patients > chunk_size:
            # Stream in bounded-memory batches (identical output, lower peak RAM).
            written = generate_streaming(
                pack, args.out, n_patients=n_patients, seed=seed, chunk_size=chunk_size
            )
        else:
            written = write_dataset(
                generate_dataset(pack, n_patients=n_patients, seed=seed), args.out
            )
        manifest_spec = dataclasses.asdict(spec) if spec is not None else "master"
        write_manifest(args.out, spec=manifest_spec, seed=seed)
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
        # Cap threads before polars/numpy are imported (they read the count at import).
        max_threads = getattr(args, "max_threads", None)
        if max_threads is not None:
            if max_threads < 1:
                print("clif-forge generate: --max-threads must be >= 1", file=sys.stderr)
                return 1
            os.environ["POLARS_MAX_THREADS"] = str(max_threads)
            os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
        return _run_generate(args)
    if args.command == "fit":
        return _run_fit(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

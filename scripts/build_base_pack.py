"""Build the shareable base pack (authoring step, credentialed).

The base pack that others derive variants from is an **aggregate-only** parameter
pack — marginals, transition probabilities, per-state physiology, correlations,
prevalences; no row-level data. It is produced once by a credentialed maintainer
from a fitted pack + the staged real CLIF dataset (for the aggregate age quantiles
and med marginal), then committed so anyone can derive CLIF-like variants with
``clif-forge generate --preset/--spec`` and no credential.

The pack carries the full acuity structure (including the level-2 high-flow tier),
so ``recalibrate_to_network_median`` + variant overrides land correctly — the same
transform that builds the master dataset. Releasing the pack is governed by the
same compliance determination as the datasets (aggregate statistics, no PHI).

Usage:
    uv run python scripts/build_base_pack.py \
        --fitted-pack data/param_packs/mimic_refit --real-dir ~/Data/clif-mimic --out base_pack
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.populations import derive_chicago_population


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fitted-pack", required=True, help="Fitted base pack (e.g. mimic_refit).")
    ap.add_argument("--real-dir", required=True, help="Real CLIF dir (age/med aggregate inputs).")
    ap.add_argument("--out", default="base_pack", help="Output pack directory.")
    args = ap.parse_args(argv)

    fitted = ParamPack.load(args.fitted_pack)
    base = derive_chicago_population(fitted, Path(args.real_dir).expanduser())
    base.write(args.out)
    print(f"wrote shareable base pack -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

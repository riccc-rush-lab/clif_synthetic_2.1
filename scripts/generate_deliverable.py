"""Generate the full network-median synthetic ICU deliverable, reproducibly.

Runs the end-to-end chain — fitted base pack -> Chicago-population derivation ->
network-median recalibration -> chunked generation -> streaming concat — and
writes one ``clif_<table>.parquet`` per table. Chunking keeps peak memory bounded
for a cohort of ~100M rows.

The generated dataset is a MIMIC-derived artifact and is **not** committed; only
this script is. Requires the staged real dataset (for the Chicago derivation's
aggregate age/med fit) and a fitted base pack.

Usage:
    uv run python scripts/generate_deliverable.py \
        --base-pack data/param_packs/mimic_refit \
        --real-dir ~/Data/clif-mimic \
        --out ~/Desktop/clif_synthetic_chicago_icu \
        --n 85248
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.generate.populations import derive_chicago_population
from clifforge.generate.recalibrate import recalibrate_to_network_median

CHUNK = 8_000
BASE_SEED = 2025


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-pack", required=True, help="Fitted base parameter pack directory.")
    ap.add_argument("--real-dir", required=True, help="Real CLIF dir (Chicago derivation inputs).")
    ap.add_argument("--out", required=True, help="Output directory for the dataset.")
    ap.add_argument("--n", type=int, default=85_248, help="Number of ICU encounters.")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    parts = out / "_parts"
    if out.exists():
        shutil.rmtree(out)
    parts.mkdir(parents=True)

    base = ParamPack.load(args.base_pack)
    chicago = derive_chicago_population(base, Path(args.real_dir).expanduser())
    pack = recalibrate_to_network_median(chicago)

    n_chunks = (args.n + CHUNK - 1) // CHUNK
    table_names: list[str] = []
    for c in range(n_chunks):
        size = min(CHUNK, args.n - c * CHUNK)
        ds = generate_dataset(pack, n_patients=size, seed=BASE_SEED + c, id_offset=c * CHUNK)
        for name, frame in ds.tables.items():
            (parts / name).mkdir(exist_ok=True)
            frame.write_parquet(parts / name / f"chunk_{c:03d}.parquet")
        (parts / "truth").mkdir(exist_ok=True)
        ds.truth.write_parquet(parts / "truth" / f"chunk_{c:03d}.parquet")
        table_names = list(ds.tables.keys())
        print(f"chunk {c + 1}/{n_chunks} ({min((c + 1) * CHUNK, args.n):,}/{args.n:,})", flush=True)

    total = 0
    for name in [*table_names, "truth"]:
        dst = out / f"clif_{name}.parquet"
        pl.scan_parquet(str(parts / name / "*.parquet")).sink_parquet(dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
        total += rows
        print(f"  clif_{name:28s} {rows:>12,} rows", flush=True)
    shutil.rmtree(parts)
    print(f"\nDONE -> {out}  |  {args.n:,} encounters | {total:,} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

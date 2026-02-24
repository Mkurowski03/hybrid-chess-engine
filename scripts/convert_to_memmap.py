#!/usr/bin/env python3
"""Convert HDF5 dataset → numpy memmap files for fast training I/O.

Usage
-----
    python scripts/convert_to_memmap.py data/train.h5 data/

Creates:
    data/states.npy   (N, 18, 8, 8) float32
    data/policies.npy (N,)          int64
    data/values.npy   (N,)          float32
    data/meta.json    { "n_positions": N }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def convert(h5_path: Path, out_dir: Path, chunk: int = 100_000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n = f["states"].shape[0]
        print(f"  Positions: {n:,}")
        print(f"  Output:    {out_dir}")

        # Create memmap files
        states_mm = np.lib.format.open_memmap(
            str(out_dir / "states.npy"), mode="w+",
            dtype=np.float32, shape=(n, 18, 8, 8),
        )
        policies_mm = np.lib.format.open_memmap(
            str(out_dir / "policies.npy"), mode="w+",
            dtype=np.int64, shape=(n,),
        )
        values_mm = np.lib.format.open_memmap(
            str(out_dir / "values.npy"), mode="w+",
            dtype=np.float32, shape=(n,),
        )

        # Copy in chunks with progress bar
        for start in tqdm(range(0, n, chunk), desc="Converting", unit="chunk"):
            end = min(start + chunk, n)
            states_mm[start:end] = f["states"][start:end]
            policies_mm[start:end] = f["policies"][start:end]
            values_mm[start:end] = f["values"][start:end]

        # Flush
        del states_mm, policies_mm, values_mm

    # Save metadata
    meta = {"n_positions": n}
    with open(out_dir / "meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)

    # Report sizes
    s_mb = (out_dir / "states.npy").stat().st_size / 1e6
    p_mb = (out_dir / "policies.npy").stat().st_size / 1e6
    v_mb = (out_dir / "values.npy").stat().st_size / 1e6
    print(f"\n✅ Done!")
    print(f"   states.npy   : {s_mb:,.1f} MB")
    print(f"   policies.npy : {p_mb:,.1f} MB")
    print(f"   values.npy   : {v_mb:,.1f} MB")
    print(f"   Total        : {s_mb + p_mb + v_mb:,.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="HDF5 → numpy memmap conversion")
    parser.add_argument("h5_path", type=Path, help="Path to train.h5")
    parser.add_argument("out_dir", type=Path, help="Output directory for .npy files")
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  HDF5 → Memmap Conversion")
    print("=" * 50)

    t0 = time.perf_counter()
    convert(args.h5_path, args.out_dir)
    elapsed = time.perf_counter() - t0
    print(f"   Time        : {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()

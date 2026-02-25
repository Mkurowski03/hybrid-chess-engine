import argparse
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_memmap(path, shape, dtype):
    """Helper to create a writeable memmap file ensures consistent mode."""
    return np.lib.format.open_memmap(
        str(path), mode="w+", dtype=dtype, shape=shape
    )


def convert_dataset(h5_path: Path, output_dir: Path, chunk_size=100_000):
    """
    Streams data from HDF5 to Numpy Memmap to allow fast random access during training.
    """
    if not h5_path.exists():
        logger.error(f"Input file not found: {h5_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Source: {h5_path}")
    logger.info(f"Destination: {output_dir}")

    try:
        with h5py.File(h5_path, "r") as f:
            # Check for expected datasets
            if "states" not in f or "policies" not in f or "values" not in f:
                logger.error("HDF5 file is missing required datasets (states, policies, values).")
                return

            total_samples = f["states"].shape[0]
            logger.info(f"Found {total_samples:,} samples to process.")

            # Prepare output paths
            path_states = output_dir / "states.npy"
            path_policies = output_dir / "policies.npy"
            path_values = output_dir / "values.npy"

            # Initialize memmaps (N, 18, 8, 8 is the standard board tensor shape)
            logger.info("Allocating memmap files...")
            states_mm = create_memmap(path_states, (total_samples, 18, 8, 8), np.float32)
            policies_mm = create_memmap(path_policies, (total_samples,), np.int64)
            values_mm = create_memmap(path_values, (total_samples,), np.float32)

            # Process in chunks to maintain low RAM usage
            logger.info(f"Starting conversion (chunk size: {chunk_size})...")
            
            for start_idx in tqdm(range(0, total_samples, chunk_size), unit="chunk"):
                end_idx = min(start_idx + chunk_size, total_samples)
                
                # Direct slice copy - efficient for HDF5 -> Memmap
                states_mm[start_idx:end_idx] = f["states"][start_idx:end_idx]
                policies_mm[start_idx:end_idx] = f["policies"][start_idx:end_idx]
                values_mm[start_idx:end_idx] = f["values"][start_idx:end_idx]

            # Explicit cleanup is often necessary on Windows to release file handles
            del states_mm
            del policies_mm
            del values_mm

        # Save metadata for the dataloader
        meta_path = output_dir / "meta.json"
        with open(meta_path, "w") as mf:
            json.dump({"n_positions": total_samples}, mf, indent=2)

        # Log file sizes
        s_size = path_states.stat().st_size / (1024**2)
        p_size = path_policies.stat().st_size / (1024**2)
        v_size = path_values.stat().st_size / (1024**2)
        
        logger.info("Conversion complete.")
        logger.info(f"Final sizes: states={s_size:.1f}MB, policies={p_size:.1f}MB, values={v_size:.1f}MB")

    except Exception as e:
        logger.exception("Critical error during conversion")
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to Numpy Memmap.")
    parser.add_argument("h5_path", type=Path, help="Path to input .h5 file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--chunk", type=int, default=100_000, help="Processing chunk size")
    
    args = parser.parse_args()

    start_time = time.perf_counter()
    convert_dataset(args.h5_path, args.output_dir, args.chunk)
    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
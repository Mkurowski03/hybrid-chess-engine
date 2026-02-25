"""
Rust MCTS Integration & Performance Test.

Verifies:
1. Python -> Rust initialization.
2. Batch leaf selection (Tree Traversal).
3. Tensor shape correctness.
4. Backpropagation mechanics.
5. High-throughput performance (Sims/Sec).
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Attempt to load Rust extension
try:
    import chess_engine_core
except ImportError:
    logger.critical("chess_engine_core not found! Run `maturin develop --release`.")
    sys.exit(1)


def verify_integration():
    """Checks basic functionality of the Rust MCTS wrapper."""
    logger.info("--- Integration Verification ---")
    
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # 1. Initialization
    try:
        # Args: FEN, CPUCT, Discount, SyzygyPath (None), SimplificationFactor (0.0)
        mcts = chess_engine_core.RustMCTS(fen, 1.25, 0.90, None, 0.0)
        logger.info("[PASS] RustMCTS initialized.")
    except Exception as e:
        logger.error(f"[FAIL] Initialization error: {e}")
        return

    # 2. Leaf Selection (Batch 8)
    batch_size = 8
    start = time.perf_counter()
    tensors, node_ids = mcts.select_leaves(batch_size)
    duration = time.perf_counter() - start
    
    if len(node_ids) == batch_size:
        logger.info(f"[PASS] Selected {len(node_ids)} leaves in {duration*1000:.2f}ms.")
    else:
        logger.warning(f"[WARN] Requested {batch_size}, got {len(node_ids)} leaves.")

    # 3. Tensor Shape Check
    # Rust returns a list of 3D numpy arrays (18, 8, 8)
    if tensors and tensors[0].shape == (18, 8, 8):
        logger.info(f"[PASS] Tensor shape correct: {tensors[0].shape}")
    else:
        logger.error(f"[FAIL] Invalid tensor shape: {tensors[0].shape if tensors else 'None'}")

    # 4. Backpropagation
    # Simulate Neural Network output
    fake_values = [0.1] * len(node_ids)
    # Uniform policy (1/4096)
    fake_policies = [[1.0 / 4096.0] * 4096 for _ in range(len(node_ids))]
    
    start = time.perf_counter()
    mcts.backpropagate(node_ids, fake_values, fake_policies)
    duration = time.perf_counter() - start
    logger.info(f"[PASS] Backpropagated {len(node_ids)} nodes in {duration*1000:.2f}ms.")

    # 5. Best Move
    best_move = mcts.best_move()
    logger.info(f"[PASS] Best move suggestion: {best_move}")


def benchmark_throughput(batch_size: int = 16, batches: int = 2000):
    """Measures raw tree traversal and backprop speed (no NN overhead)."""
    logger.info(f"\n--- Speed Benchmark (Batch={batch_size}, Iter={batches}) ---")
    
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    mcts = chess_engine_core.RustMCTS(fen, 1.25, 0.90, None, 0.0)
    
    # Pre-allocate dummy data to isolate backprop speed
    dummy_val = [0.05] * batch_size
    dummy_pol = [[0.0] * 4096 for _ in range(batch_size)]
    
    total_sims = 0
    start_time = time.perf_counter()
    
    for _ in range(batches):
        tensors, node_ids = mcts.select_leaves(batch_size)
        
        count = len(node_ids)
        if count == 0:
            break
            
        # Handle partial batches at end of game/tree
        if count != batch_size:
            current_val = dummy_val[:count]
            current_pol = dummy_pol[:count]
        else:
            current_val = dummy_val
            current_pol = dummy_pol
            
        mcts.backpropagate(node_ids, current_val, current_pol)
        total_sims += count

    total_time = time.perf_counter() - start_time
    throughput = total_sims / total_time
    
    logger.info(f"Total Sims:  {total_sims:,}")
    logger.info(f"Total Time:  {total_time:.4f}s")
    logger.info(f"Throughput:  {throughput:,.0f} sims/sec")
    logger.info("Note: This measures pure Rust Tree + Python Overhead (No GPU Inference).")


if __name__ == "__main__":
    verify_integration()
    benchmark_throughput()
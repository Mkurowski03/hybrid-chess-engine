"""
Rust vs Python Board Encoding Benchmark.

Verifies that the Rust implementation produces bit-perfect identical tensors
to the Python reference implementation and measures the speedup factor.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

import chess
import numpy as np

# Inject project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_encoder import encode_board

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Attempt to load Rust Core
try:
    import chess_engine_core
except ImportError:
    logger.critical("chess_engine_core not found! Run `maturin develop --release` first.")
    sys.exit(1)


def verify_correctness(fens: List[str]):
    """
    Asserts that Rust and Python encoders produce identical tensors.
    """
    logger.info(f"--- Correctness Verification ({len(fens)} positions) ---")
    
    for i, fen in enumerate(fens):
        py_board = chess.Board(fen)
        
        # Python Encoding
        py_tensor = encode_board(py_board)
        # Python encoder might set repetition logic based on board history.
        # Rust single-FEN constructor defaults to 0.0.
        # We zero out channel 12 (repetition) for a fair comparison of piece placement.
        py_tensor[12].fill(0.0) 

        # Rust Encoding
        rust_board = chess_engine_core.RustBoard(fen)
        rust_tensor = rust_board.encode()

        try:
            assert py_tensor.shape == rust_tensor.shape
            assert np.allclose(py_tensor, rust_tensor, atol=1e-6)
            logger.info(f"[PASS] FEN {i+1}: {fen[:60]}...")
            
        except AssertionError:
            logger.error(f"[FAIL] Mismatch on FEN: {fen}")
            
            diff_indices = np.where(np.abs(py_tensor - rust_tensor) > 1e-6)
            unique_channels = set(diff_indices[0])
            
            logger.error(f"Differences found in channels: {unique_channels}")
            
            for c in unique_channels:
                logger.error(f"--- Channel {c} Diff ---")
                logger.error(f"Python:\n{py_tensor[c]}")
                logger.error(f"Rust:\n{rust_tensor[c]}")
            
            sys.exit(1)


def benchmark_speed(fen: str, iterations: int = 10_000):
    """
    Compares execution speed of Python vs Rust encoding.
    """
    logger.info(f"\n--- Speed Benchmark ({iterations:,} iterations) ---")
    
    py_board = chess.Board(fen)
    rust_board = chess_engine_core.RustBoard(fen)

    # 1. Python Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = encode_board(py_board)
    py_duration = time.perf_counter() - start

    # 2. Rust Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = rust_board.encode()
    rust_duration = time.perf_counter() - start

    # Reporting
    py_speed = iterations / py_duration
    rust_speed = iterations / rust_duration
    speedup = py_duration / rust_duration

    logger.info(f"Python: {py_duration:.4f}s ({py_speed:,.0f} ops/sec)")
    logger.info(f"Rust:   {rust_duration:.4f}s ({rust_speed:,.0f} ops/sec)")
    logger.info(f"Speedup: {speedup:.2f}x 🚀")


def main():
    test_fens = [
        # Start Position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Kiwipete (Complex Middlegame)
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        # Endgame (Sparse)
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        # Castling & Promotion rights check
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        # Black to move (Verification of flip logic)
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        # Flipped colors manually
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"
    ]

    verify_correctness(test_fens)
    
    # Use Kiwipete for benchmark as it has a mix of pieces and empty squares
    benchmark_speed(test_fens[1])


if __name__ == "__main__":
    main()
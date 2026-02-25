#!/usr/bin/env python3
"""
Performance Benchmark for ChessNet-3070 (Hybrid Engine).

Measures Nodes Per Second (NPS) including:
1. Rust MCTS Tree Traversal
2. Python/PyTorch Tensor Batching
3. GPU Inference Latency
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import chess

# Inject project root for imports (works from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_engine import HybridEngine
from config import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_benchmark(args):
    # 1. Initialize Engine
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return

    logger.info(f"Initializing HybridEngine from {checkpoint.name}...")
    try:
        engine = HybridEngine(
            checkpoint_path=checkpoint,
            device=args.device,
            model_cfg=ModelConfig()
        )
    except Exception as e:
        logger.critical(f"Engine init failed: {e}")
        return

    # 2. Setup Position
    try:
        board = chess.Board(args.fen)
    except ValueError:
        logger.error("Invalid FEN string provided.")
        return

    logger.info(f"Position: {args.fen}")
    logger.info(f"Configuration: Batch={args.batch_size}, Sims={args.sims:,}")

    # 3. Warmup Phase (Crucial for CUDA/CuDNN autotuner)
    logger.info("Warming up GPU/JIT...")
    engine.select_move(board, sims=500, batch_size=args.batch_size)

    # 4. Benchmarking
    logger.info("Starting benchmark run...")
    
    start_time = time.perf_counter()
    
    best_move = engine.select_move(
        board,
        sims=args.sims,
        cpuct=1.25,
        material_weight=0.15,
        discount=0.90,
        batch_size=args.batch_size,
        wtime=None, # Disable time management to force full simulation count
        btime=None
    )
    
    duration = time.perf_counter() - start_time
    nps = args.sims / duration

    # 5. Report
    print("\n" + "-" * 40)
    print(f"Total Sims:  {args.sims:,}")
    print(f"Total Time:  {duration:.4f}s")
    print(f"Throughput:  {nps:,.2f} NPS")
    print(f"Best Move:   {best_move.uci()}")
    print("-" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ChessNet Performance Benchmark")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/baseline/chessnet_epoch9.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--fen", 
        type=str, 
        default="r1bq1rk1/pp2bppp/2n2n2/3p4/3N4/2N1B3/PPP1BPPP/R2Q1RK1 w - - 4 10",
        help="FEN string to evaluate"
    )
    parser.add_argument("--sims", type=int, default=100_000, help="Target simulations")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
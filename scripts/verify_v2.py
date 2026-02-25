#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

import chess
import torch

# Inject project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig
from src.hybrid_engine import HybridEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def benchmark_nps(engine, duration=5.0):
    """
    Runs a search on startpos to measure NPS and warm up JIT.
    """
    logger.info(f"--- Benchmark: NPS Warmup (~{duration}s) ---")
    board = chess.Board()
    
    start = time.perf_counter()
    
    # We simulate a clock with ~30s remaining to encourage the engine 
    # to think for about 5 seconds (assuming 1/6th time allocation).
    # If your engine supports fixed-time search, use that instead.
    simulated_clock_ms = int(duration * 6 * 1000)
    
    move = engine.select_move(
        board, 
        sims=100000, # High cap to ensure time is the bottleneck
        wtime=simulated_clock_ms, 
        btime=simulated_clock_ms,
        book_strategy="none"
    )
    
    elapsed = time.perf_counter() - start
    logger.info(f"Benchmark finished in {elapsed:.2f}s.")
    logger.info(f"Best move: {move}")
    logger.info("Check the internal engine logs above for precise NPS (Target: >4500).")


def verify_smart_pruning(engine):
    """
    Verifies that the engine detects an overwhelmingly dominant move 
    (Promotion) and stops the search early (Smart Pruning).
    """
    logger.info("--- Verify: Smart Pruning ---")
    
    # FEN: White has a8=Q as a dominant move. No mate in 1, so forced mate guard won't trigger.
    fen = "8/P1p3p1/2p3p1/8/8/4k3/4p3/4K3 w - - 0 1"
    board = chess.Board(fen)
    
    logger.info(f"Position: {fen}")
    
    start = time.perf_counter()
    
    # We give it a high sim cap (80k). If pruning works, it should finish MUCH faster.
    move = engine.select_move(board, sims=80000, wtime=None, btime=None, book_strategy="none")
    
    elapsed = time.perf_counter() - start
    logger.info(f"Selected: {move} | Time: {elapsed:.2f}s")
    
    if elapsed < 2.0:
        logger.info("Search terminated early (Pruning active).")
    else:
        logger.warning(f"Search took {elapsed:.2f}s. Pruning logic may be inactive.")


def verify_simplification_bias(engine):
    """
    Verifies 'Killer Instinct': Engine should trade pieces (simplify)
    when it leads to a mathematically winning tablebase position.
    """
    logger.info("--- Verify: Simplification Bias ---")
    
    # FEN: White to move. e2e7 (Rxe7) forces a winning K+R vs K endgame.
    fen = "4k3/4n3/8/8/8/8/4R2P/4R1K1 w - - 0 1"
    board = chess.Board(fen)
    
    # Give it enough time to find the tactical shot
    move = engine.select_move(board, sims=10000, wtime=60000, btime=60000, book_strategy="none")
    
    expected_uci = "e2e7"
    if move and move.uci() == expected_uci:
        logger.info(f"Engine chose {expected_uci} (Simplification).")
    else:
        logger.error(f"Engine chose {move}, expected {expected_uci}.")


def main():
    parser = argparse.ArgumentParser(description="ChessNet Verification Suite")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        default=Path("checkpoints/baseline/chessnet_epoch9.pt"),
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda/cpu)"
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Initializing HybridEngine on {args.device}...")
    
    try:
        # Load Engine
        engine = HybridEngine(
            checkpoint_path=str(args.checkpoint),
            model_cfg=ModelConfig(),
            device=args.device
        )
        
        # Run Suite
        benchmark_nps(engine)
        print() # Spacer
        verify_smart_pruning(engine)
        print() # Spacer
        verify_simplification_bias(engine)
        
        logger.info("Suite execution complete.")

    except KeyboardInterrupt:
        logger.warning("\nVerification suite interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.exception("An unexpected error occurred during verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
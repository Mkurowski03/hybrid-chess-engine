#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

import chess
import torch

# Inject project root for imports
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


def verify_syzygy_integration(checkpoint_path: Path):
    """
    Verifies that the engine bypasses MCTS for 3-4-5 piece endgames
    by probing Syzygy tablebases instantly.
    """
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # 1. Initialize Engine
    logger.info("Initializing Hybrid Engine...")
    
    # We use a dummy model config, assuming standard architecture
    config = ModelConfig()
    
    try:
        engine = HybridEngine(
            checkpoint_path=str(checkpoint_path),
            model_cfg=config,
            # Ensure tablebases directory is detected. 
            # In production, this might be set via env var or config.
            # Assuming 'tablebases/' is in project root.
        )
    except Exception as e:
        logger.exception("Failed to initialize engine")
        sys.exit(1)

    # 2. Setup Test Case (K+R vs K)
    # Position: White King c1, Black King c3, Black Rook b2. White to move.
    # This is a known lost position for White, but probing should be instant.
    fen = "8/8/8/8/8/2k5/1r6/2K5 w - - 0 1"
    board = chess.Board(fen)
    
    logger.info(f"Test Position: {fen}")
    logger.info(f"Pieces: {len(board.piece_map())} (Syzygy active <= 5)")

    # 3. Execute Probe
    logger.info("Requesting move (Sims=1000)...")
    
    start = time.perf_counter()
    
    # We give it enough time/sims that if MCTS triggers, it would take >1s.
    # If Syzygy triggers, it should return in <10ms.
    best_move = engine.select_move(
        board,
        sims=1000,
        wtime=60000, btime=60000, winc=0, binc=0
    )
    
    duration = time.perf_counter() - start
    duration_ms = duration * 1000

    # 4. Analyze Results
    logger.info(f"Selected Move: {best_move}")
    logger.info(f"Latency:       {duration_ms:.2f} ms")

    # Threshold: 50ms is extremely generous for a tablebase lookup (usually <1ms).
    # MCTS initialization alone usually takes >100ms.
    if duration_ms < 50:
        logger.info("Instant response detected. Syzygy probe active.")
    else:
        logger.warning(f"WARNING: Response took {duration_ms:.2f} ms.")
        logger.warning("This suggests the engine might be falling back to MCTS search.")
        logger.warning("Check if 'tablebases/' directory exists and contains .rtbw files.")


def main():
    parser = argparse.ArgumentParser(description="Verify Syzygy Tablebase Integration")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        default=Path("checkpoints/baseline/chessnet_epoch9.pt"),
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    verify_syzygy_integration(args.checkpoint)


if __name__ == "__main__":
    main()
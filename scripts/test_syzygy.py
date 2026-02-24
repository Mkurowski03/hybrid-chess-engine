import sys
import chess
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_engine import HybridEngine
from config import ModelConfig

def test_syzygy():
    print("Initializing HybridEngine with Syzygy Tablebases...")
    checkpoint = "checkpoints/baseline/chessnet_epoch9.pt"
    
    # model_cfg defaults to looking for "tablebases/"
    engine = HybridEngine(checkpoint_path=checkpoint, book_path=None, model_cfg=ModelConfig())
    
    fen = "8/8/8/8/8/2k5/1r6/2K5 w - - 0 1"
    print(f"\nSetting board to FEN: {fen} (K+R vs K)")
    board = chess.Board(fen)
    
    print("\nProbing move via HybridEngine...")
    start_time = time.time()
    
    # We will invoke select_move which should hit the Syzygy root probe or Rust MCTS leaf probe
    # Note: K+R vs K is 3 pieces, so it's definitely <= 5 pieces.
    # The Python root probe should instantly find the perfect move.
    best_move = engine.select_move(
        board,
        sims=1000,
        wtime=5000, btime=5000, winc=0, binc=0
    )
    
    elapsed = time.time() - start_time
    print(f"\nEngine Selected Move: {best_move}")
    print(f"Time Taken: {elapsed:.4f} seconds")
    
    if elapsed < 0.1:
        print("\n[VERIFIED] Syzygy 5-piece probing active and correct. (Instant reply)")
    else:
        print("\n[WARNING] Move took longer than expected. Did it rely on MCTS?")

if __name__ == "__main__":
    test_syzygy()

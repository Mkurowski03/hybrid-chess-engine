import sys
import time
import chess
import logging
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_engine import HybridEngine
from config import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_benchmark():
    print("Loading engine for benchmarking...")
    checkpoint = "checkpoints/baseline/chessnet_epoch9.pt"
    # we enforce no opening book so it's forced to calculate MCTS
    engine = HybridEngine(checkpoint_path=checkpoint, book_path=None)
    
    board = chess.Board() # Start position
    
    # We will give it a massive time pool so it can think for exactly 5 seconds
    wtime = 300000 
    btime = 300000
    
    print("Starting 5 second benchmark run...")
    
    target_time_ms = 5000
    wtime_bench = target_time_ms * 20
    
    start = time.time()
    best_move = engine.select_move(
        board, 
        sims=1000000, 
        wtime=wtime_bench, 
        btime=wtime_bench,
        winc=0, 
        binc=0,
        batch_size=1024 # Increased batch size to saturate the RTX 3070 Ti better
    )
    end = time.time()
    
    print(f"Benchmark completed!")
    print(f"Move selected: {best_move}")

if __name__ == "__main__":
    run_benchmark()

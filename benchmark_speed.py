import time
import chess
from src.hybrid_engine import HybridEngine

def run_benchmark():
    print("Initializing HybridEngine (Rust + CUDA)...")
    engine = HybridEngine("checkpoints/baseline/chessnet_epoch9.pt")
    
    fen = "r1bq1rk1/pp2bppp/2n2n2/3p4/3N4/2N1B3/PPP1BPPP/R2Q1RK1 w - - 4 10"
    board = chess.Board(fen)
    
    print(f"\nPosition: {fen}")
    print("Warming up GPU...")
    _ = engine.select_move(board, sims=1000)

    sims_to_run = 100_000
    print(f"\nRunning {sims_to_run} simulations deep search...")
    
    t0 = time.time()
    best_move = engine.select_move(
        board,
        sims=sims_to_run,
        cpuct=1.25,
        material_weight=0.15,
        discount=0.90,
        batch_size=16
    )
    t1 = time.time()
    
    elapsed = t1 - t0
    nps = sims_to_run / elapsed
    
    print("-" * 40)
    print(f"Total Time: {elapsed:.2f} seconds")
    print(f"Total Sims: {sims_to_run:,}")
    print(f"Speed:      {nps:,.2f} NPS")
    print(f"Best Move:  {best_move.uci()}")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()

import time
import chess
import logging
from src.hybrid_engine import HybridEngine
from config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_tests():
    print("\n" + "="*50)
    print("🚀 CHESSNET-3070 VERIFICATION SUITE V2")
    print("="*50 + "\n")
    
    # Initialize Engine
    print("Loading HybridEngine (CUDA)...")
    try:
        engine = HybridEngine("checkpoints/baseline/chessnet_epoch9.pt", device="cuda")
    except Exception as e:
        print(f"FAILED TO LOAD ENGINE: {e}")
        return
        
    # ---------------------------------------------------------
    # TEST 1: NPS Benchmark (5 seconds, startpos)
    # ---------------------------------------------------------
    print("\n--- TEST 1: NPS BENCHMARK ---")
    board = chess.Board()
    start_time = time.time()
    
    try:
        # 5 seconds ~ 5000 wtime * 6 (alloc factor ~16%) = 30000 wtime
        # We'll just pass a large sims count and restrict wtime directly
        move = engine.select_move(board, sims=80000, wtime=30000, book_strategy="none")
        elapsed = time.time() - start_time
        
        # Read final generic log from Hybrid Engine directly or check stdout
        print(f"Test 1 Complete. Time elapsed: {elapsed:.2f}s. Move chosen: {move}")
        print("Note: Check the info depth ... stdout logs above for Final NPS. Target: > 4500.")
    except Exception as e:
        print(f"NPS Benchmark Failed: {e}")

    # ---------------------------------------------------------
    # TEST 2: Smart Pruning (Overwhelming Dominant Move)
    # ---------------------------------------------------------
    print("\n--- TEST 2: SMART PRUNING ---")
    board = chess.Board("8/P1p3p1/2p3p1/8/8/4k3/4p3/4K3 w - - 0 1")
    print(f"FEN: {board.fen()} (White has a8=Q as overwhelmingly best move, no mate in 1)")
    start_time = time.time()
    
    try:
        # Give it unlimited time but cap at 80,000 sims. It should stop well before 80k.
        move = engine.select_move(board, sims=80000, wtime=None, book_strategy="none")
        elapsed = time.time() - start_time
        print(f"Test 2 Complete. Move chosen: {move}. Time elapsed: {elapsed:.2f}s.")
        print("Note: Check stdout for '[SMART PRUNING] Dominant move found...'. Sims should be << 80000.")
    except Exception as e:
        print(f"Smart Pruning Test Failed: {e}")

    # ---------------------------------------------------------
    # TEST 3: Killer Instinct / Simplification Bias
    # ---------------------------------------------------------
    print("\n--- TEST 3: KILLER INSTINCT (SIMPLIFICATION BIAS) ---")
    board = chess.Board("4k3/4n3/8/8/8/8/4R2P/4R1K1 w - - 0 1")
    print(f"FEN: {board.fen()} (6 pieces - White has obvious Rxe7+ into tablebase win)")
    
    try:
        # We give it minimal time to ensure the +0.2 EV boost overrides baseline values quickly
        move = engine.select_move(board, sims=10000, wtime=60000, book_strategy="none")
        print(f"Test 3 Complete. Move chosen: {move}")
        if move and move.uci() == "e2e7":
            print("SUCCESS! Engine chose e2e7, capturing the knight and simplifying the endgame.")
        else:
            print("WARNING: Engine did NOT choose to simplify (e2e7). Check PUCT logic or Anti-Shuffle.")
    except Exception as e:
        print(f"Killer Instinct Test Failed: {e}")
        
    print("\n" + "="*50)
    print("✅ VERIFICATION SUITE COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_tests()

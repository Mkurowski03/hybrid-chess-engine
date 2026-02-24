import time
import numpy as np
import chess
from src.board_encoder import encode_board

try:
    import chess_engine_core
except ImportError:
    print("chess_engine_core not found! Please build the rust extension first.")
    exit(1)

def main():
    # A complex middlegame FEN to test pieces, castling, and flip logic
    fens_to_test = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Start
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", # Kiwipete
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", # Endgame
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", # Promotion/Castling
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", # Black to move (test flip logic)
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1" # Flipped colors
    ]

    print("--- Correctness Verification ---")
    for fen in fens_to_test:
        py_board = chess.Board(fen)
        rust_board = chess_engine_core.RustBoard(fen)
        
        py_tensor = encode_board(py_board)
        # Force rep count to 0.0 in Python for fair matching (since RustBoard single FEN doesn't track history)
        py_tensor[12].fill(0.0) 
        rust_tensor = rust_board.encode()
        
        try:
            assert py_tensor.shape == rust_tensor.shape
            assert np.allclose(py_tensor, rust_tensor)
            print(f"[OK] FEN matches perfect bitwise: {fen}")
        except AssertionError:
            print(f"[FAIL] Mismatch on FEN: {fen}")
            diff = np.where(py_tensor != rust_tensor)
            print(f"Differences found at channels: {set(diff[0])}")
            for c in set(diff[0]):
                print(f"--- Channel {c} ---")
                print("Python:")
                print(py_tensor[c])
                print("Rust:")
                print(rust_tensor[c])
            return

    print("\n--- Speed Benchmark (10,000 Encodings) ---")
    fen = fens_to_test[1]
    py_board = chess.Board(fen)
    rust_board = chess_engine_core.RustBoard(fen)
    
    iters = 10000
    
    t0 = time.time()
    for _ in range(iters):
        _ = encode_board(py_board)
    py_time = time.time() - t0
    
    t0 = time.time()
    for _ in range(iters):
        _ = rust_board.encode()
    rust_time = time.time() - t0
    
    print(f"Python Encode Time: {py_time:.4f}s")
    print(f"Rust Encode Time:   {rust_time:.4f}s")
    print(f"Speedup:            {py_time/rust_time:.2f}X")

if __name__ == "__main__":
    main()

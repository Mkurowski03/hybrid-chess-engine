import time
import numpy as np

try:
    import chess_engine_core
except ImportError:
    print("chess_engine_core not found! Please build the rust extension first.")
    exit(1)

def main():
    print("--- Rust Batch MCTS Verification ---")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Init MCTS
    mcts = chess_engine_core.RustMCTS(fen, 1.25, 0.90)
    print("[OK] RustMCTS initialized")
    
    # Select Leaves (Batch of 8)
    t0 = time.time()
    tensors, node_ids = mcts.select_leaves(8)
    t_select = time.time() - t0
    
    print(f"[OK] Selected {len(node_ids)} leaves in {t_select:.4f}s")
    print(f"     Node IDs: {node_ids}")
    print(f"     Tensor Shape: {tensors[0].shape}")
    
    # Fake Evaluation
    values = [0.1] * len(node_ids)
    policies = [[1.0 / 4096.0] * 4096 for _ in range(len(node_ids))]
    
    # Backpropagate
    t0 = time.time()
    mcts.backpropagate(node_ids, values, policies)
    t_backprop = time.time() - t0
    print(f"[OK] Backpropagated {len(node_ids)} nodes in {t_backprop:.4f}s")
    
    # Best Move
    best_move = mcts.best_move()
    print(f"[OK] Suggested Best Move (1 Iteration): {best_move}")
    
    print("\n--- Speed Test ---")
    # Run 100 batches of 16 (1600 sims)
    t0 = time.time()
    total_sims = 0
    for _ in range(100):
        tensors, node_ids = mcts.select_leaves(16)
        if not node_ids:
            break
        # Fast dummy backprop
        v = [0.0] * len(node_ids)
        p = [[0.0] * 4096 for _ in range(len(node_ids))]
        mcts.backpropagate(node_ids, v, p)
        total_sims += len(node_ids)
        
    t_total = time.time() - t0
    print(f"Processed {total_sims} simulations in {t_total:.4f}s")
    print(f"Throughput: {total_sims / t_total:.2f} sims/sec (Tree Operations Only)")

if __name__ == "__main__":
    main()

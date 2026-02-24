"""Hybrid Rust/Python Engine combining Rust MCTS with PyTorch."""

import time
import sys
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
import logging
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.board_encoder import encode_board
from src.model import ChessNet, build_model

def _move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
    """Encode a ``chess.Move`` into a policy index."""
    from_sq = move.from_square
    to_sq = move.to_square
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq

try:
    import chess_engine_core
except ImportError:
    print("chess_engine_core not found! Please build the rust extension first.")
    sys.exit(1)


class HybridEngine:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        model_cfg: ModelConfig | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---- model ----
        self.model: ChessNet = build_model(model_cfg)
        if checkpoint_path is not None:
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True,
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def select_move(
        self, 
        board: chess.Board, 
        sims: int = 600,
        cpuct: float = 1.25,
        material_weight: float = 0.15,
        discount: float = 0.90,
        batch_size: int = 16,
        wtime: Optional[int] = None,
        btime: Optional[int] = None,
        winc: int = 0,
        binc: int = 0
    ) -> chess.Move:
        """Select a move using Rust MCTS + PyTorch Network."""
        
        # --- SAFETY OVERRIDE: HARDCODED MATE-IN-1 ---
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                # print(f"DEBUG: Executing forced mate sequence: {move.uci()}")
                board.pop()
                return move
            board.pop()
        # --------------------------------------------

        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        # --- DYNAMIC TIME MANAGEMENT ---
        start_time = time.time()
        last_log_time = start_time
        alloc_time_ms = None
        safe_time_ms = None
        
        my_time = wtime if board.turn == chess.WHITE else btime
        my_inc = winc if board.turn == chess.WHITE else binc

        if my_time is not None:
            alloc_time_ms = (my_time / 20.0) + (my_inc * 0.75)
            safe_time_ms = my_time - 1000  # Leave absolute 1 second buffer
            if safe_time_ms < 100:
                safe_time_ms = 100 # absolute minimum allowed to think

            # Scale down target sims based on estimated NPS (Assume 6500 conservative)
            ESTIMATED_NPS = 6500
            max_possible_sims = int((alloc_time_ms / 1000.0) * ESTIMATED_NPS)
            
            # If we literally have less than 1 second allocated and very low buffer, reduce drastically
            if my_time < 5000:
                max_possible_sims = min(max_possible_sims, int((safe_time_ms / 1000.0) * ESTIMATED_NPS))
            
            sims = max(batch_size, min(sims, max_possible_sims))
            logging.info(f"Panic Mode: Time low! Capping sims to {sims}. alloc={alloc_time_ms:.1f}ms safe={safe_time_ms:.1f}ms")
            # print(f"DEBUG TIME: my_time={my_time}ms, alloc={alloc_time_ms}ms, safe={safe_time_ms}ms. Targeting sims: {sims}")
        # --------------------------------

        # Initialize Rust MCTS Tree
        fen = board.fen()
        rust_mcts = chess_engine_core.RustMCTS(fen, cpuct, discount)

        current_sims = 0
        
        while current_sims < sims:
            # 1. Rust finds leaves to evaluate
            # tensors is a list of 3D numpy arrays, node_ids is a list of ints
            tensors, node_ids = rust_mcts.select_leaves(batch_size) 
            
            if not node_ids: 
                break # Tree fully explored or game over
                
            batch_len = len(node_ids)
            current_sims += batch_len

            # Emergency Brake Time Check
            if alloc_time_ms is not None:
                elapsed_ms = (time.time() - start_time) * 1000.0
                if elapsed_ms > alloc_time_ms or elapsed_ms > safe_time_ms:
                    # Timeout! Abort search immediately
                    break

            # 2. PyTorch evaluates on GPU
            # Convert list of 3D numpy arrays to a single 4D Torch tensor (B, 18, 8, 8)
            states_np = np.stack(tensors)
            input_cuda = torch.from_numpy(states_np).to(self.device)
            
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        policy, value = self.model(input_cuda)
                else:
                    policy, value = self.model(input_cuda)

            policy_np = policy.cpu().numpy() # Shape: (B, 4096)
            value_np = value.cpu().numpy().flatten() # Shape: (B,)
            
            # Blend Master Material Weight into Value
            # HybridEngine delegates exact board construction to Rust, but we can approximate material or apply it on Python side if needed.
            # For simplicity and extreme speed, we use pure NN value here, or build a fast material evaluator.
            # We'll stick to pure NN value for Phase 3 baseline testing.

            # Convert to lists for PyO3 boundary (Rust expects Vec<f32> and Vec<Vec<f32>>)
            # Note: PyO3 can be made to accept numpy arrays directly for massive speedup, but list works for MVP.
            val_list = value_np.tolist()
            pol_list = policy_np.tolist()

            # 3. Rust updates the tree
            rust_mcts.backpropagate(node_ids, val_list, pol_list)
            
            # --- LIVE LOGGING ---
            curr_time = time.time()
            if current_sims % 2000 < batch_size or curr_time - last_log_time >= 0.5:
                # Output UCI Info
                temp_best = rust_mcts.best_move()
                elapsed_ms = int((curr_time - start_time) * 1000)
                nps = int(current_sims / ((curr_time - start_time) + 1e-6))
                
                info_str = f"info depth {current_sims} score cp 0 time {elapsed_ms} nodes {current_sims} nps {nps} pv {temp_best}"
                print(info_str)
                sys.stdout.flush()
                sys.stderr.write(info_str + "\n")
                sys.stderr.flush()
                
                last_log_time = curr_time
            
        best_uci = rust_mcts.best_move()
        
        # Log final stats
        final_time = time.time()
        final_elapsed = final_time - start_time
        final_nps = int(current_sims / (final_elapsed + 1e-6))
        logging.info(f"[DONE] Move: {best_uci} | Sims: {current_sims} | Time Used: {final_elapsed:.2f}s | NPS: {final_nps}")
        
        try:
            return chess.Move.from_uci(best_uci)
        except ValueError:
            # Fallback if tree fails
            return legal[0]

    def evaluate(self, board: chess.Board) -> float:
        """Return the value-head evaluation in ``[-1, 1]`` for the side to move."""
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _policy, value = self.model(state)
            else:
                _policy, value = self.model(state)
        return float(value.item())

    @torch.no_grad()
    def top_moves(self, board: chess.Board, n: int = 5, **kwargs) -> list[dict]:
        """Return top-N legal moves with probabilities from the policy head."""
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, _value = self.model(state)
        else:
            policy_logits, _value = self.model(state)

        logits = policy_logits.squeeze(0).cpu().numpy()

        legal_moves = []
        for move in board.legal_moves:
            idx = _move_to_policy_index(move, board.turn)
            legal_moves.append((move, idx, logits[idx]))

        if not legal_moves:
            return []

        # Softmax over legal moves only
        raw = np.array([x[2] for x in legal_moves], dtype=np.float32)
        raw -= raw.max()
        exp = np.exp(raw)
        probs = exp / exp.sum()

        ranked = sorted(zip(legal_moves, probs), key=lambda x: -x[1])

        result = []
        for (move, _idx, _logit), prob in ranked[:n]:
            result.append({
                "move": move.uci(),
                "san": board.san(move),
                "prob": round(float(prob), 4),
            })
        return result

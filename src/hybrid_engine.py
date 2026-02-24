"""Hybrid Rust/Python Engine combining Rust MCTS with PyTorch."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch

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
    logging.error("chess_engine_core not found! Please build the rust extension first.")
    sys.exit(1)


class HybridEngine:
    """Hybrid Rust/Python Engine combining Rust MCTS with PyTorch."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        model_cfg: Optional[ModelConfig] = None,
        book_path: str | Path | None = None,
    ) -> None:
        """Initialize the Hybrid Engine.

        Args:
            checkpoint_path (str | Path | None): Path to the PyTorch model checkpoint.
            device (str): Device to use for inference (e.g., "cuda" or "cpu").
            model_cfg (Optional[ModelConfig]): Configuration for the model.
            book_path (str | Path | None): Path to a Polyglot opening book (.bin).
        """
        self.book_path = book_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        
        # ---- model ----
        self.model: ChessNet = build_model(model_cfg)
        if checkpoint_path is not None:
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True,
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
        if self.device.type == "cuda":
            self.model.half()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.model.to(self.device)
        self.model.eval()
        
        # ---- syzygy ----
        self.tablebase = None
        if self.model_cfg and self.model_cfg.syzygy_path: # Use self.model_cfg
            tb_path = Path(self.model_cfg.syzygy_path) # Use self.model_cfg
            if tb_path.exists() and any(tb_path.iterdir()):
                try:
                    self.tablebase = chess.syzygy.open_tablebase(str(tb_path))
                    logging.info(f"Syzygy tablebase loaded from {tb_path}")
                except Exception as e:
                    logging.warning(f"Could not load Syzygy: {e}")

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
        binc: int = 0,
        book_strategy: str = "best"
    ) -> chess.Move:
        """Select a move using Rust MCTS + PyTorch Network.

        Args:
            board (chess.Board): The current board state.
            sims (int): Number of MCTS simulations. Default is 600.
            cpuct (float): PUCT exploration constant. Default is 1.25.
            material_weight (float): Material bias weight. Default is 0.15.
            discount (float): Discount factor for delayed checkmates. Default is 0.90.
            batch_size (int): Execution batch size. Default is 16.
            wtime (Optional[int]): White time remaining in milliseconds.
            btime (Optional[int]): Black time remaining in milliseconds.
            winc (int): White increment per move in milliseconds.
            binc (int): Black increment per move in milliseconds.

        Returns:
            chess.Move: The selected best move.
        """
        
        # --- SAFETY OVERRIDE: HARDCODED MATE-IN-1 ---
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                # print(f"DEBUG: Executing forced mate sequence: {move.uci()}")
                board.pop()
                return move
            board.pop()
        # --------------------------------------------
        
        # --- OPENING BOOK SUPPORT ---
        if self.book_path and Path(self.book_path).exists():
            try:
                with chess.polyglot.open_reader(self.book_path) as reader:
                    if book_strategy == "best":
                        best_entry = None
                        for entry in reader.find_all(board):
                            if best_entry is None or entry.weight > best_entry.weight:
                                best_entry = entry
                        if best_entry:
                            logging.info(f"[BOOK] Found best move {best_entry.move.uci()} (weight: {best_entry.weight})")
                            return best_entry.move
                    else:
                        entry = reader.weighted_choice(board)
                        if entry:
                            logging.info(f"[BOOK] Found move {entry.move.uci()} (weight: {entry.weight})")
                            return entry.move
            except IndexError:
                pass
            except Exception as e:
                logging.warning(f"Error reading opening book: {e}")
        # --------------------------------------------
        
        # --- SYZYGY ENDGAME ROOT PROBING ---
        if self.tablebase is not None:
            piece_count = len(board.piece_map())
            if piece_count <= 5: # Assuming 5-piece tablebases are populated
                try:
                    # DTZ (Distance To Zero) gives the optimal move to win/draw
                    wdl = self.tablebase.probe_wdl(board)
                    if wdl is not None:
                        # Find the best move according to DTZ, which preserves perfect play
                        best_move = None
                        best_dtz = None
                        
                        # We want the lowest absolute DTZ for winning, highest for losing etc.
                        # python-chess has a convenient probe_dtz
                        
                        # Just grab the first move that maintains the WDL or optimal DTZ
                        for m in board.legal_moves:
                            board.push(m)
                            # After our move, it's opponents turn, so DTZ flips
                            try:
                                dtz = self.tablebase.probe_dtz(board)
                                # WDL flips too (from opponents perspective)
                                reply_wdl = self.tablebase.probe_wdl(board)
                                board.pop()
                                
                                # Simplified logic: If we are winning (wdl > 0), we want the reply_wdl to be < 0 for opponent
                                if wdl > 0 and reply_wdl < 0:
                                    if best_dtz is None or dtz > best_dtz: # Negative DTZ is worse for opponent
                                        best_dtz = dtz
                                        best_move = m
                                elif wdl == 0 and reply_wdl == 0:
                                    best_move = m
                                elif wdl < 0 and reply_wdl > 0:
                                    # We are losing. Maximize DTZ to delay mate
                                    if best_dtz is None or dtz > best_dtz:
                                        best_dtz = dtz
                                        best_move = m
                            except:
                                board.pop()
                                
                        if best_move:
                            logging.info(f"[SYZYGY] Perfect endgame move found: {best_move.uci()} (WDL: {wdl})")
                            return best_move
                except chess.syzygy.MissingTableError:
                    pass
                except Exception as e:
                    logging.warning(f"Syzygy probing error: {e}")
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
            # Deep Thinker: Spend ~16.6% of remaining time per move (more aggressive)
            alloc_time_ms = (my_time / 6.0) + (my_inc * 0.75)
            safe_time_ms = my_time - 1000  # Leave absolute 1 second buffer
            if safe_time_ms < 100:
                safe_time_ms = 100 # absolute minimum allowed to think

            # Scale down target sims based on estimated NPS (Assume 6500 conservative)
            ESTIMATED_NPS = 6500
            max_possible_sims = int((alloc_time_ms / 1000.0) * ESTIMATED_NPS)
            
            # Deep Thinker: Raised panic threshold to 15 seconds. Below that, cap to 5000 sims or lower
            if my_time < 15000:
                max_possible_sims = min(max_possible_sims, 5000)
                max_possible_sims = min(max_possible_sims, int((safe_time_ms / 1000.0) * ESTIMATED_NPS))
            
            sims = max(batch_size, min(sims, max_possible_sims))
            logging.info(f"Panic Mode: Time low! Capping sims to {sims}. alloc={alloc_time_ms:.1f}ms safe={safe_time_ms:.1f}ms")
            # print(f"DEBUG TIME: my_time={my_time}ms, alloc={alloc_time_ms}ms, safe={safe_time_ms}ms. Targeting sims: {sims}")
        # --------------------------------

        # Initialize Rust MCTS Tree
        fen = board.fen()
        
        # Pass syzygy path if available
        tb_path_str = None
        if self.tablebase is not None and self.model_cfg and self.model_cfg.syzygy_path:
            tb_path_str = self.model_cfg.syzygy_path
            
        rust_mcts = chess_engine_core.RustMCTS(fen, cpuct, discount, tb_path_str)

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
            if self.device.type == "cuda":
                input_cuda = input_cuda.half()
            
            with torch.inference_mode():
                # We have explicitly made model and input .half()
                policy, value = self.model(input_cuda)

            # Ensure we convert back to float32 before numpy/Rust transition 
            policy_np = policy.float().cpu().numpy() # Shape: (B, 4096)
            value_np = value.float().cpu().numpy().flatten() # Shape: (B,)
            
            # --- SYZYGY MCTS LEAF OVERRIDE ---
            if self.tablebase is not None:
                for idx, node_id in enumerate(node_ids):
                    # We need the FEN or board to probe. 
                    # Optimization Note: Reconstructing the board from Rust node is expensive in Python.
                    # Since this runs inside the hot loop, a pure Python override is too slow (re-parsing FENs).
                    # For phase 1, we rely solely on Root Probing (implemented above), which already catches
                    # 5-piece endgames at the start of the turn instantly. 
                    # To do leaf-probing efficiently, it MUST be implemented in Rust via shakmaty-syzygy.
                    pass
            # ---------------------------------
            
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
        """Return the value-head evaluation in ``[-1, 1]`` for the side to move.

        Args:
            board (chess.Board): The current board state.

        Returns:
            float: Evaluation score from -1.0 to 1.0.
        """
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            state = state.half()
            
        with torch.inference_mode():
            _policy, value = self.model(state)
        return float(value.item())

    @torch.no_grad()
    def top_moves(self, board: chess.Board, n: int = 5, **kwargs: Any) -> list[dict]:
        """Return top-N candidate moves with probabilities from the policy head.

        Args:
            board (chess.Board): The current board state.
            n (int): Number of top moves to return.
            kwargs (Any): Additional keyword arguments.

        Returns:
            list[dict]: List of dictionaries containing top moves and probabilities.
        """
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            state = state.half()

        with torch.inference_mode():
            policy_logits, _value = self.model(state)

        logits = policy_logits.squeeze(0).float().cpu().numpy()

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

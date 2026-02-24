"""Chess inference engine with Alpha-Beta search and opening-book support.

The neural network provides two things:
  1. **Policy head** → move ordering (search best candidates first)
  2. **Value head**  → leaf-node evaluation (replaces hand-crafted eval)

This hybrid approach turns raw pattern-matching into tactical awareness.

Usage
-----
>>> from src.engine import ChessEngine
>>> engine = ChessEngine("checkpoints/baseline/best.pt")
>>> board = chess.Board()
>>> move = engine.select_move(board)             # depth-3 alpha-beta
>>> move = engine.select_move(board, depth=0)     # raw policy (no search)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import chess
import chess.polyglot
import numpy as np
import torch

# Allow direct imports when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.board_encoder import encode_board
from src.model import ChessNet, build_model
from src.mcts import MCTS


def _policy_index_to_move(index: int, turn: chess.Color) -> chess.Move:
    """Decode a policy index (``from_sq * 64 + to_sq``) back to a ``chess.Move``."""
    from_sq = index // 64
    to_sq = index % 64
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return chess.Move(from_sq, to_sq)


def _move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
    """Encode a ``chess.Move`` into a policy index."""
    from_sq = move.from_square
    to_sq = move.to_square
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq


class ChessEngine:
    """Neural-network chess engine with Alpha-Beta search.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to a saved ``.pt`` checkpoint.
    book_path : str | Path | None
        Path to a Polyglot ``.bin`` opening book.
    device : str
        ``"cuda"`` or ``"cpu"``.
    model_cfg : ModelConfig | None
        Override the default model architecture.
    search_depth : int
        Default search depth for ``select_move``.  0 = raw policy (no search).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        book_path: str | Path | None = None,
        device: str = "cuda",
        model_cfg: ModelConfig | None = None,
        search_depth: int = 3,
        syzygy_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.default_depth = search_depth
        
        # ---- Syzygy Tablebases ----
        self.tablebase = None
        if syzygy_path:
             try:
                 import chess.syzygy
                 self.tablebase = chess.syzygy.open_tablebase(str(syzygy_path))
                 print(f"Syzygy Tablebase loaded from {syzygy_path}")
             except Exception as e:
                 print(f"Warning: Could not open Syzygy at {syzygy_path}: {e}")

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

        # ---- opening book ----
        self._book_path: Optional[Path] = None
        if book_path is not None:
            p = Path(book_path)
            if p.is_file():
                self._book_path = p

        # ---- search stats (populated after each search) ----
        self.last_search_nodes = 0
        self.last_search_time = 0.0

        # ---- transposition table (MCTS doesn't use this directly yet, or uses its own tree) ----
        # We can remove self.tt if MCTS manages its own tree.
        # But for now let's keep it if we want to add it back later.
        # Actually MCTS builds a tree. We validly don't need TT hash map the same way.
        self.tt = {} 
        
        # ---- MCTS ----
        self.mcts = MCTS(self.model, device=self.device, batch_size=8)

    def clear_tt(self):
        """Clear the MCTS tree (re-initialize)."""
        self.mcts = MCTS(self.model, device=self.device, batch_size=8)

        # ---- search stats (populated after each search) ----
        self.last_search_nodes = 0
        self.last_search_time = 0.0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def select_move(
        self, board: chess.Board, depth: int | None = None, simulations: int = 800, **kwargs
    ) -> chess.Move:
        """Return the best move for the current position using MCTS.

        1. Try the opening book first (instant).
        2. If *depth* == 0, use raw policy argmax (instinct).
        3. Otherwise, run MCTS search.
        """
        book_move = self._try_book(board)
        if book_move is not None:
            return book_move

        # If depth=0 is requested, skip MCTS
        if depth == 0:
            return self._nn_move(board)
            
        # Run MCTS
        # We can map 'depth' to simulations if we want, but default 800 is good.
        sims = simulations
        if depth is not None and depth > 0:
             # Heuristic: depth 1 -> 100, depth 3 -> 800, depth 5 -> 3000
             if depth == 1: sims = 100
             elif depth == 2: sims = 400
             elif depth == 3: sims = 800
             elif depth >= 4: sims = 2000

        # Adaptive Endgame Boost (simulating "Math is Key")
        # If pieces <= 12, the search space is smaller, so we can search much deeper.
        # We boost simulations 3x and reduce exploration to focus on best lines.
        cpuct = kwargs.get('cpuct', 2.0)
        piece_count = len(board.piece_map())
        if piece_count <= 12:
            sims = int(sims * 3.0)
            cpuct = kwargs.get('cpuct', 1.25)
            
        if sims > 5000: sims = 5000 # Safety cap

        # Create a fresh MCTS runner for this thread/search to prevent state races
        mat_weight = kwargs.get('material_weight', 0.2)
        disc = kwargs.get('discount', 0.90)
        p_vals = kwargs.get('piece_values', None)
        
        mcts_runner = MCTS(self.model, self.device, cpuct=cpuct, material_weight=mat_weight, discount=disc, piece_values=p_vals)
        self.mcts = mcts_runner # Save reference for checkmate extraction later

        # 0. Forced Mate Check 
        # Adaptive Depth: 4 normally, 6 if very few pieces (Mate in 3)
        mate_depth = 4
        if len(board.piece_map()) <= 6:
            mate_depth = 6
            
        mate_move = self.find_mate_in_n(board, depth=mate_depth)
        if mate_move:
            print(f"Force-Mate found (depth={mate_depth}): {mate_move}")
            return mate_move

        # 0.5 Syzygy Tablebase Probe (Perfect Play)
        if self.tablebase:
            try:
                # Only check if pieces match tablebase size (usually <= 5, maybe <= 6 if large TB)
                if len(board.piece_map()) <= 5: # Assuming standard 3-4-5 TB
                    best_tb_move = None
                    best_tb_score = (-999, -99999) # maximize (-wdl, dtz)

                    found_tb_move = False
                    # Check all legal moves
                    for move in board.legal_moves:
                        board.push(move)
                        try:
                            # Probe opponent's perspective
                            wdl = self.tablebase.probe_wdl(board)
                            dtz = self.tablebase.probe_dtz(board)
                            
                            # DEBUG: Syzygy probe result
                            # print(f"DEBUG: Syzygy probe result for {board.fen()}: wdl={wdl}, dtz={dtz}")
                            
                            score = (-wdl, dtz)
                            if score > best_tb_score:
                                best_tb_score = score
                                best_tb_move = move
                                
                            found_tb_move = True
                        except chess.syzygy.MissingTableError:
                            pass
                        finally:
                            board.pop()
                    
                    if found_tb_move and best_tb_move:
                        print(f"Syzygy Move: {best_tb_move} (WDL={best_tb_score[0]}, DTZ={best_tb_score[1]})")
                        return best_tb_move
            except Exception as e:
                print(f"Syzygy Error: {e}")

        # 1. Search (MCTS)
        mcts_move = mcts_runner.search(board, num_simulations=sims)
        
        # 2. Winning Draw Rejection & Blunder Guard
        # We look at candidate moves from MCTS and filter out draws/blunders
        root = self.mcts.root
        if root and root.Q > 0.1: # We think we are winning
            candidates = sorted(root.children.items(), key=lambda x: x[1].N, reverse=True)
            for move, node in candidates:
                if node.N == 0: continue
                
                board.push(move)
                is_draw = board.can_claim_draw()
                is_blunder = self._is_immediate_blunder(board)
                board.pop()
                
                if not is_draw and not is_blunder:
                    return move
                    
            # Fallback if all moves are filtered (this shouldn't happen unless we're forced into draw)
            pass

        return mcts_move

    def find_mate_in_n(self, board, depth):
        """
        Simple DFS to find a forced mate within 'depth' plies.
        Returns the winning Move or None.
        """
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            
            # If not immediate mate, check if this leads to forced mate
            if depth > 1:
                 # Opponent's turn: He will try to AVOID mate.
                 # If ALL his moves lead to our mate, then it's a forced mate.
                 opponent_survives = False
                 if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                     opponent_survives = True
                 else:
                     # Check all opponent moves
                     for opp_move in board.legal_moves:
                         board.push(opp_move)
                         # Can we mate after this?
                         # We need to find ONE move that mates
                         can_mate_response = self._can_mate(board, depth - 2)
                         board.pop()
                         
                         if not can_mate_response:
                             opponent_survives = True
                             break
                 
                 if not opponent_survives:
                     board.pop()
                     return move

            board.pop()
        return None

    def _can_mate(self, board, depth):
        """Helper for find_mate_in_n (Recursive)"""
        # Our turn. We need ONE move that leads to mate.
        if depth <= 0:
            return False
            
        for move in board.legal_moves:
             board.push(move)
             if board.is_checkmate():
                 board.pop()
                 return True
             
             # Opponent's turn logic (All moves must fail)
             if depth > 1:
                 opponent_survives = False
                 if board.is_game_over(): # Draw?
                     opponent_survives = True
                 else:
                     all_opp_moves_lead_to_mate = True
                     for opp_move in board.legal_moves:
                         board.push(opp_move)
                         still_can_mate = self._can_mate(board, depth - 2)
                         board.pop()
                         if not still_can_mate:
                             all_opp_moves_lead_to_mate = False
                             break
                     if not all_opp_moves_lead_to_mate:
                         opponent_survives = True
                 
                 if not opponent_survives:
                     board.pop()
                     return True
             
             board.pop()
        return False

    def _is_immediate_blunder(self, board: chess.Board) -> bool:
        """Check if opponent has an immediate mate-in-1."""
        for opp_move in board.legal_moves:
            board.push(opp_move)
            if board.is_checkmate():
                board.pop()
                return True
            board.pop()
        return False

    def evaluate(self, board: chess.Board) -> float:
        """Return the value-head evaluation in ``[-1, 1]`` for the side to move."""
        return self._nn_evaluate(board)

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

    # -----------------------------------------------------------------
    # Opening Book
    # -----------------------------------------------------------------

    def _try_book(self, board: chess.Board) -> Optional[chess.Move]:
        if self._book_path is None:
            return None
        try:
            with chess.polyglot.open_reader(str(self._book_path)) as reader:
                entry = reader.weighted_choice(board)
                return entry.move
        except (IndexError, KeyError, Exception):
            return None

    # -----------------------------------------------------------------
    # Raw NN inference (no search)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _nn_move(self, board: chess.Board) -> chess.Move:
        """Select a move using raw policy argmax (no search)."""
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, _value = self.model(state)
        else:
            policy_logits, _value = self.model(state)

        logits = policy_logits.squeeze(0).cpu().numpy()

        legal_indices: list[int] = []
        for move in board.legal_moves:
            legal_indices.append(_move_to_policy_index(move, board.turn))

        mask = np.full(4096, -1e9, dtype=np.float32)
        for idx in legal_indices:
            mask[idx] = 0.0
        masked_logits = logits + mask

        best_idx = int(np.argmax(masked_logits))
        return _policy_index_to_move(best_idx, board.turn)

    @torch.no_grad()
    def _nn_evaluate(self, board: chess.Board) -> float:
        """Value-head evaluation in [-1, 1] for side to move."""
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                _policy, value = self.model(state)
        else:
            _policy, value = self.model(state)
        return float(value.item())

    @torch.no_grad()
    def _nn_policy_and_value(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Single forward pass returning both policy logits and value."""
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, value = self.model(state)
        else:
            policy_logits, value = self.model(state)
        return policy_logits.squeeze(0).cpu().numpy(), float(value.item())

    # -----------------------------------------------------------------
    # MCTS (via imported module) - Legacy Alpha-Beta removed
    # -----------------------------------------------------------------



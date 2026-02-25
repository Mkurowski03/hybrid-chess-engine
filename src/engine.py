import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch

# Inject project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.board_encoder import encode_board, move_to_policy_index
from src.model import ChessNet, build_model
from src.mcts import MCTS

# Configure logging
logger = logging.getLogger(__name__)


def policy_index_to_move(index: int, turn: chess.Color) -> chess.Move:
    """Decodes a policy index (0-4095) into a chess.Move."""
    from_sq, to_sq = divmod(index, 64)
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return chess.Move(from_sq, to_sq)


class ChessEngine:
    """
    Neural MCTS Chess Engine.
    
    Combines a Dual-Headed ResNet (Policy/Value) with Monte Carlo Tree Search,
    Opening Books (Polyglot), and Endgame Tablebases (Syzygy).
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        book_path: Optional[Union[str, Path]] = None,
        syzygy_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        model_cfg: Optional[ModelConfig] = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        self.model = build_model(model_cfg)
        if checkpoint_path:
            self._load_checkpoint(Path(checkpoint_path))
        
        self.model.to(self.device)
        self.model.eval()

        # 2. Load Resources
        self.book_path = Path(book_path) if book_path else None
        self.tablebase = self._init_tablebase(syzygy_path)
        
        # 3. Initialize MCTS (Stateless runner)
        self.mcts = MCTS(self.model, device=self.device, batch_size=8)
        
        logger.info(f"Engine initialized on {self.device}")

    def _load_checkpoint(self, path: Path):
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}, using random weights.")
            return
        
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint: {path.name}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def _init_tablebase(self, path: Union[str, Path, None]):
        if not path:
            return None
        try:
            tb = chess.syzygy.open_tablebase(str(path))
            logger.info(f"Syzygy tablebases active: {path}")
            return tb
        except Exception as e:
            logger.warning(f"Syzygy init failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Core Decision Logic
    # -------------------------------------------------------------------------

    def select_move(
        self, 
        board: chess.Board, 
        depth: Optional[int] = None, 
        simulations: int = 800, 
        **kwargs
    ) -> chess.Move:
        """
        Determines the best move using the following priority pipeline:
        1. Opening Book
        2. Forced Mate Solver (DFS)
        3. Syzygy Tablebases (Perfect Endgame)
        4. Neural MCTS
        """
        # 1. Opening Book
        if self.book_path:
            book_move = self._get_book_move(board, kwargs.get("book_strategy", "weighted_random"))
            if book_move:
                return book_move

        # 2. Raw Policy (Zero-shot)
        if depth == 0:
            return self._get_raw_policy_move(board)

        # 3. Forced Mate Guard (Avoids "Promotion Blindness")
        # We search deeper if pieces are few (endgame).
        mate_depth = 6 if len(board.piece_map()) <= 6 else 4
        forced_mate = self._solve_forced_mate(board, depth=mate_depth)
        if forced_mate:
            logger.info(f"Mate Guard: Found forced mate in {mate_depth} plies.")
            return forced_mate

        # 4. Syzygy Tablebase
        # Standard TBs are 3-4-5 pieces.
        if self.tablebase and len(board.piece_map()) <= 5:
            tb_move = self._probe_syzygy(board)
            if tb_move:
                return tb_move

        # 5. MCTS Search configuration
        # Adaptive simulations based on game phase
        eff_sims = simulations
        
        # Endgame Turbo: If pieces <= 12, we can afford deeper search
        if len(board.piece_map()) <= 12:
            eff_sims = int(simulations * 3.0)
            
        # Hard cap for safety
        eff_sims = min(eff_sims, 100_000)

        # 6. Execute MCTS
        return self._run_mcts(board, eff_sims, **kwargs)

    def _run_mcts(self, board: chess.Board, sims: int, **kwargs) -> chess.Move:
        # Re-initialize MCTS to ensure no state leakage between moves
        self.mcts = MCTS(
            self.model, 
            self.device, 
            cpuct=kwargs.get('cpuct', 2.0),
            material_weight=kwargs.get('material_weight', 0.2),
            discount=kwargs.get('discount', 0.90),
            piece_values=kwargs.get('piece_values', None)
        )
        
        best_move = self.mcts.search(board, num_simulations=sims)

        # Post-Search Safety: Check for Draw by Repetition in winning positions
        root = self.mcts.root
        if root and root.Q > 0.1: # If we think we are winning
            best_move = self._filter_winning_draws(board, root)

        return best_move

    def _filter_winning_draws(self, board: chess.Board, root_node) -> chess.Move:
        """
        Prevents the engine from playing a move that claims a 3-fold repetition draw
        when the engine evaluates the position as winning (> 0.1).
        """
        # Sort children by visit count
        candidates = sorted(root_node.children.items(), key=lambda x: x[1].N, reverse=True)
        
        for move, node in candidates:
            if node.N == 0: continue
            
            board.push(move)
            is_draw = board.can_claim_draw() or board.is_stalemate()
            is_blunder = self._is_immediate_blunder(board)
            board.pop()
            
            if not is_draw and not is_blunder:
                return move
                
        # If all moves are bad/draws, return the original best
        return candidates[0][0]

    # -------------------------------------------------------------------------
    # Helpers: Syzygy, Book, Mate
    # -------------------------------------------------------------------------

    def _get_book_move(self, board: chess.Board, strategy: str) -> Optional[chess.Move]:
        if not self.book_path: return None
        try:
            with chess.polyglot.open_reader(str(self.book_path)) as reader:
                if strategy == "best":
                    # Deterministic best move
                    entry = reader.find(board)
                    return entry.move
                else:
                    # Weighted random
                    entry = reader.weighted_choice(board)
                    return entry.move
        except (IndexError, KeyError):
            return None
        except Exception:
            return None

    def _probe_syzygy(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Probes WDL and DTZ tables. 
        Prioritizes: Win > Draw > Loss. 
        Tie-breaker: Minimize DTZ (for win), Maximize DTZ (for loss).
        """
        best_move = None
        # Score format: (WDL, -DTZ) for wins, (WDL, DTZ) for defense
        best_score = (-2, -9999) 

        for move in board.legal_moves:
            board.push(move)
            try:
                # Probe from opponent's perspective, so invert WDL
                # WDL: 2=Win, 1=Win(50), 0=Draw, -1=Loss(50), -2=Loss
                wdl = -self.tablebase.probe_wdl(board)
                dtz = self.tablebase.probe_dtz(board)
                
                # Logic to strictly rank moves:
                # 1. WDL (Win is best)
                # 2. DTZ (Shortest win is best)
                
                # Normalize score for comparison
                # We want to MAXIMIZE this score tuple
                if wdl > 0:
                    # Winning: prefer smaller DTZ (faster win) -> negate DTZ
                    current_score = (wdl, -dtz)
                elif wdl < 0:
                    # Losing: prefer larger DTZ (delay loss) -> positive DTZ
                    current_score = (wdl, dtz)
                else:
                    # Draw
                    current_score = (wdl, 0)

                if current_score > best_score:
                    best_score = current_score
                    best_move = move

            except chess.syzygy.MissingTableError:
                pass
            finally:
                board.pop()
        
        if best_move:
            logger.info(f"Syzygy Probe: Selected {best_move} (Score={best_score})")
            
        return best_move

    def _solve_forced_mate(self, board: chess.Board, depth: int) -> Optional[chess.Move]:
        """DFS to find a forced mate sequence within `depth` plies."""
        # Simple iterative deepening wrapper could be added here, 
        # but for small depths straight DFS is fine.
        return self._dfs_mate(board, depth, is_root=True)

    def _dfs_mate(self, board: chess.Board, depth: int, is_root: bool = False) -> Optional[chess.Move]:
        if depth <= 0: return None
        
        # 1. Our Turn: Find ONE move that guarantees mate
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            
            # If no immediate mate, verify if this move forces a mate deeper
            if depth > 1:
                # After we move, it's opponent's turn. 
                # We need to ensure ALL of their replies lead to OUR mate.
                opponent_escapes = False
                
                if board.is_game_over(): # Stalemate/Draw
                    opponent_escapes = True
                else:
                    for reply in board.legal_moves:
                        board.push(reply)
                        can_still_mate = self._dfs_mate(board, depth - 2)
                        board.pop()
                        
                        if not can_still_mate:
                            opponent_escapes = True
                            break
                
                board.pop()
                if not opponent_escapes:
                    return move # This move forces mate!
            else:
                board.pop()

        return None

    def _is_immediate_blunder(self, board: chess.Board) -> bool:
        """Checks if the opponent has an immediate Mate-in-1 response."""
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return True
            board.pop()
        return False

    # -------------------------------------------------------------------------
    # Neural Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def _get_raw_policy_move(self, board: chess.Board) -> chess.Move:
        """Selects move solely based on Policy Head probabilities."""
        state = encode_board(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        policy_logits, _ = self.model(tensor)
        
        # Mask illegal moves
        logits = policy_logits.squeeze(0).cpu().numpy()
        mask = np.full(logits.shape, -float('inf'))
        
        legal_indices = [
            move_to_policy_index(m, board.turn) for m in board.legal_moves
        ]
        mask[legal_indices] = 0
        
        best_idx = np.argmax(logits + mask)
        return policy_index_to_move(int(best_idx), board.turn)

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> float:
        """Returns Value Head evaluation [-1, 1]."""
        state = encode_board(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        _, value = self.model(tensor)
        return float(value.item())

    @torch.inference_mode()
    def get_top_moves_data(self, board: chess.Board, top_n: int = 5) -> List[dict]:
        """Returns analytic data for the top N moves by raw policy."""
        state = encode_board(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        policy_logits, value = self.model(tensor)
        logits = policy_logits.squeeze(0).cpu().numpy()
        
        # Softmax over legal moves
        moves = []
        for m in board.legal_moves:
            idx = move_to_policy_index(m, board.turn)
            moves.append((m, logits[idx]))
            
        if not moves: return []
        
        # Numerical stability softmax
        scores = np.array([x[1] for x in moves])
        scores -= scores.max()
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # Sort and Format
        ranked = sorted(zip(moves, probs), key=lambda x: x[1], reverse=True)
        
        results = []
        for ((move, _), prob) in ranked[:top_n]:
            results.append({
                "uci": move.uci(),
                "san": board.san(move),
                "prob": float(prob),
                "value": float(value.item())
            })
            
        return results
"""
Hybrid Chess Engine (Rust Core + Python Inference).

This module bridges the high-performance Rust MCTS implementation with 
PyTorch/ONNX neural inference. It handles:
1. Game state management
2. Time allocation strategy
3. Knowledge injection (Books, Syzygy)
4. Batch inference scheduling
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch

# Inject project root for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig, SIMPLIFICATION_FACTOR, MIN_PRUNING_SIMS, SMART_PRUNING_FACTOR
from src.board_encoder import encode_board, move_to_policy_index
from src.model import ChessNet, build_model
from src.neural_backend import NeuralBackend, PyTorchBackend, ONNXBackend

# Attempt to load the compiled Rust extension
try:
    import chess_engine_core
except ImportError:
    logging.critical("CRITICAL: 'chess_engine_core' extension not found.")
    logging.critical("Run 'maturin develop --release' to build the Rust core.")
    sys.exit(1)


# Configure logging
logger = logging.getLogger(__name__)


class HybridEngine:
    """
    Orchestrates the Hybrid Architecture:
    - Rust: Heavy MCTS tree traversal, move generation, and leaf selection.
    - Python: Neural Network inference (GPU), Time Management, and heuristics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cuda",
        model_cfg: Optional[ModelConfig] = None,
        book_path: Optional[str | Path] = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg or ModelConfig()
        
        # 1. Initialize Inference Backend
        self.backend = self._init_backend(checkpoint_path)

        # 2. Initialize Knowledge Bases
        self.book_path = Path(book_path) if book_path else None
        self.tablebase = self._init_tablebase()
        
        logger.info(f"HybridEngine initialized on {self.device}")

    def _init_backend(self, checkpoint_path: Optional[str | Path]) -> NeuralBackend:
        """Sets up PyTorch or ONNX backend based on file extension."""
        if not checkpoint_path:
            # Fallback for testing without weights
            logger.warning("No checkpoint provided. Initializing random PyTorch model.")
            model = build_model(self.model_cfg).to(self.device)
            return PyTorchBackend(model, self.device)

        path = Path(checkpoint_path)
        if path.suffix == ".onnx":
            logger.info(f"Backend: ONNX Runtime ({path.name})")
            return ONNXBackend(path)
        else:
            logger.info(f"Backend: PyTorch Native ({path.name})")
            model = build_model(self.model_cfg)
            
            # Load weights safely
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            # Support multiple checkpoint formats:
            #   - raw state_dict (from extract_weights.py or direct save)
            #   - train.py uses 'model_state_dict'
            #   - train_advanced.py uses 'model'
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict)
            
            # Optimization settings
            if self.device.type == "cuda":
                model.half()
                torch.backends.cudnn.benchmark = True
                
            model.to(self.device)
            model.eval()
            return PyTorchBackend(model, self.device)

    def _init_tablebase(self) -> Optional[chess.syzygy.Tablebase]:
        """Loads Syzygy tablebases if configured."""
        if not self.model_cfg.syzygy_path:
            return None
            
        tb_path = Path(self.model_cfg.syzygy_path)
        if tb_path.exists() and any(tb_path.iterdir()):
            try:
                tb = chess.syzygy.open_tablebase(str(tb_path))
                logger.info(f"Syzygy active: {tb_path}")
                return tb
            except Exception as e:
                logger.error(f"Syzygy load failed: {e}")
        return None

    # -------------------------------------------------------------------------
    # Core Decision Logic
    # -------------------------------------------------------------------------

    def select_move(
        self, 
        board: chess.Board, 
        sims: int = 600,
        batch_size: int = 512,
        wtime: Optional[int] = None,
        btime: Optional[int] = None,
        winc: int = 0,
        binc: int = 0,
        book_strategy: str = "best",
        search_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> chess.Move:
        """
        Main entry point for move selection.
        Pipeline: Mate Guard -> Book -> Syzygy -> MCTS.
        """
        # 1. Instant Mate Guard (1-ply)
        mate = self._check_forced_mate(board)
        if mate: return mate

        # 2. Opening Book
        book_move = self._probe_book(board, book_strategy)
        if book_move: return book_move

        # 3. Syzygy Root Probe
        tb_move = self._probe_syzygy(board)
        if tb_move: return tb_move

        # 4. Time Management
        target_sims, time_limit_ms = self._allocate_time(
            board, sims, wtime, btime, winc, binc
        )

        # 5. Rust MCTS Execution
        return self._run_mcts_loop(
            board, 
            target_sims, 
            batch_size, 
            time_limit_ms, 
            search_context, 
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Pipeline Components
    # -------------------------------------------------------------------------

    def _check_forced_mate(self, board: chess.Board) -> Optional[chess.Move]:
        """Checks for an immediate Mate-in-1."""
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        return None

    def _probe_book(self, board: chess.Board, strategy: str) -> Optional[chess.Move]:
        """Queries the Polyglot opening book."""
        if not self.book_path or not self.book_path.exists():
            return None
            
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                if strategy == "best":
                    # Iterate to find max weight entry manually if needed, 
                    # or use reader.find(board) which usually returns the first/best match.
                    # Here we replicate the logic of finding the absolute max weight.
                    best_entry = max(reader.find_all(board), key=lambda e: e.weight, default=None)
                    if best_entry:
                        logger.info(f"[BOOK] Best move: {best_entry.move.uci()} (W: {best_entry.weight})")
                        return best_entry.move
                else:
                    entry = reader.weighted_choice(board)
                    if entry:
                        logger.info(f"[BOOK] Weighted move: {entry.move.uci()} (W: {entry.weight})")
                        return entry.move
        except Exception:
            return None
        return None

    def _probe_syzygy(self, board: chess.Board) -> Optional[chess.Move]:
        """Probes 5-piece Syzygy tablebases for perfect play."""
        if self.tablebase is None or len(board.piece_map()) > 5:
            return None

        try:
            # WDL: Win(+2), Win50(+1), Draw(0), Loss50(-1), Loss(-2)
            wdl = self.tablebase.probe_wdl(board)
            if wdl is None: return None

            # Optimization: Just find any move that preserves the WDL optimality.
            # A full implementation would optimize DTZ (Distance to Zero).
            best_move = None
            
            # Simple heuristic: Prefer winning > drawing. 
            # If winning, minimize DTZ. If losing, maximize DTZ.
            best_dtz = 99999 if wdl > 0 else -99999
            
            for move in board.legal_moves:
                board.push(move)
                try:
                    res_wdl = -self.tablebase.probe_wdl(board) # Negate because opponent's turn
                    res_dtz = self.tablebase.probe_dtz(board)
                    
                    # If this move preserves the optimal result
                    if res_wdl == wdl:
                        if wdl > 0:  # Winning — minimize DTZ (fastest win)
                            if res_dtz < best_dtz:
                                best_dtz = res_dtz
                                best_move = move
                        elif wdl < 0:  # Losing — just preserve WDL, pick first legal
                            best_move = move
                            break
                        else:
                            best_move = move
                            break
                except Exception:
                    logger.exception(f"Syzygy child probe failed for {move.uci()}")
                finally:
                    board.pop()
            
            if best_move:
                logger.info(f"[SYZYGY] Perfect move: {best_move.uci()} (WDL: {wdl})")
                return best_move

        except Exception as e:
            logger.warning(f"Syzygy probe error: {e}")
            
        return None

    def _allocate_time(
        self, board: chess.Board, sims: int, wtime: Optional[int], btime: Optional[int], winc: int, binc: int
    ) -> Tuple[int, Optional[float]]:
        """Calculates dynamic simulation cap and time limit based on clock state."""
        my_time = wtime if board.turn == chess.WHITE else btime
        my_inc = winc if board.turn == chess.WHITE else binc

        if my_time is None:
            return sims, None

        # "Deep Thinker" Strategy: Aggressive usage (~16% of clock)
        alloc_ms = (my_time / 6.0) + (my_inc * 0.75)
        safe_ms = max(100, my_time - 1000) # Always leave 1s buffer

        # Hard limit based on safe time
        limit_ms = min(alloc_ms, safe_ms)

        # Dynamic Simulation Cap based on Estimated NPS
        EST_NPS = 6500
        max_sims_time = int((limit_ms / 1000.0) * EST_NPS)
        
        # Panic Mode: If < 15s remaining, cap strict
        if my_time < 15000:
            target_sims = min(sims, 5000, max_sims_time)
        else:
            target_sims = min(sims, max_sims_time)
            
        target_sims = max(target_sims, 100) # Minimum floor
        
        return target_sims, limit_ms

    def _run_mcts_loop(
        self,
        board: chess.Board,
        sims: int,
        batch_size: int,
        time_limit_ms: Optional[float],
        context: Optional[Dict],
        **kwargs
    ) -> chess.Move:
        """Executes the tight loop between Rust MCTS and Python Inference."""
        
        # Initialize Rust Core
        tb_path = str(self.model_cfg.syzygy_path) if (self.tablebase and self.model_cfg.syzygy_path) else None
        
        try:
            rust_mcts = chess_engine_core.RustMCTS(
                board.fen(),
                kwargs.get("cpuct", 1.25),
                kwargs.get("discount", 0.90),
                tb_path,
                SIMPLIFICATION_FACTOR
            )
        except TypeError:
            # Fallback for older compiled .pyd without simplification_factor
            rust_mcts = chess_engine_core.RustMCTS(
                board.fen(),
                kwargs.get("cpuct", 1.25),
                kwargs.get("discount", 0.90),
                tb_path,
            )

        current_sims = 0
        start_time = time.time()
        last_log = start_time
        
        # --- The Hot Loop ---
        while current_sims < sims:
            # 1. Rust: Traverse tree and collect leaves
            tensors, node_ids = rust_mcts.select_leaves(batch_size)
            
            if not node_ids:
                break # Search exhausted
            
            batch_len = len(node_ids)

            # 2. Python/GPU: Batch Inference
            # Stack into (B, 18, 8, 8)
            states = np.stack(tensors) 
            policy, value = self.backend.predict(states)
            
            # 3. Rust: Backpropagation
            # We convert to list to cross FFI boundary safely
            rust_mcts.backpropagate(node_ids, value.tolist(), policy.tolist())
            
            current_sims += batch_len

            # 4. Periodic Checks (Every ~4k nodes to save CPU)
            if current_sims % 4096 < batch_len:
                if self._check_interrupts(context, start_time, time_limit_ms):
                    break
                
                # Smart Pruning
                if current_sims > MIN_PRUNING_SIMS:
                    v1, v2 = rust_mcts.top_two_visits()
                    if v1 > SMART_PRUNING_FACTOR * max(v2, 1):
                        logger.info(f"[PRUNING] Dominant move {v1} vs {v2}")
                        break

                # UCI Logging
                if time.time() - last_log > 1.0:
                    self._log_uci_info(rust_mcts, current_sims, start_time)
                    last_log = time.time()

        # Finalize
        best_uci = rust_mcts.best_move()
        self._log_final_stats(best_uci, current_sims, start_time)
        
        return chess.Move.from_uci(best_uci)

    def _check_interrupts(self, context: Optional[Dict], start_time: float, limit_ms: Optional[float]) -> bool:
        """Returns True if search should stop."""
        if context and context.get("stop_flag"):
            return True
            
        if limit_ms:
            elapsed = (time.time() - start_time) * 1000
            if elapsed > limit_ms:
                return True
        return False

    def _log_uci_info(self, mcts, sims, start):
        elapsed = int((time.time() - start) * 1000)
        nps = int(sims / ((time.time() - start) + 1e-6))
        pv = mcts.best_move()
        print(f"info depth {sims // 100} time {elapsed} nodes {sims} nps {nps} pv {pv}")
        sys.stdout.flush()

    def _log_final_stats(self, move, sims, start):
        elapsed = time.time() - start
        nps = int(sims / (elapsed + 1e-6))
        logger.info(f"[SEARCH] Best: {move} | Sims: {sims} | Time: {elapsed:.2f}s | NPS: {nps}")

    # -------------------------------------------------------------------------
    # Inference Helpers
    # -------------------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> float:
        """Zero-shot value head evaluation."""
        state = encode_board(board)[np.newaxis, ...] # Add batch dim
        _, value = self.backend.predict(state)
        return float(value[0])

    @torch.no_grad()
    def top_moves(self, board: chess.Board, n: int = 5) -> List[Dict]:
        """Zero-shot policy analysis."""
        state = encode_board(board)[np.newaxis, ...]
        policy, _ = self.backend.predict(state)
        
        logits = policy[0]
        legal_moves = []
        
        for move in board.legal_moves:
            idx = move_to_policy_index(move, board.turn)
            legal_moves.append((move, logits[idx]))
            
        # Softmax
        if not legal_moves: return []
        
        scores = np.array([x[1] for x in legal_moves])
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        
        ranked = sorted(zip(legal_moves, probs), key=lambda x: -x[1])[:n]
        
        return [
            {"move": m[0].uci(), "san": board.san(m[0]), "prob": round(float(p), 4)} 
            for m, p in ranked
        ]
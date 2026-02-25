#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) Server for ChessNet-3070.

This script connects the HybridEngine to chess GUIs (Arena, Banksia, CuteChess).
It handles threading, command parsing, and time management.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import chess

# Inject project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig, SEARCH_BATCH_SIZE
from src.hybrid_engine import HybridEngine

# Configure logging to file (never stdout, as that breaks UCI)
logging.basicConfig(
    filename='engine_debug.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UCI")

# Custom piece values for the material heuristic (optional override)
CUSTOM_PIECE_VALUES = {
    chess.PAWN: 1, 
    chess.KNIGHT: 3, 
    chess.BISHOP: 3.25, 
    chess.ROOK: 5, 
    chess.QUEEN: 9.5
}


class UCIServer:
    """
    Handles the UCI protocol loop and manages the search thread.
    """

    def __init__(self, engine: HybridEngine, args: argparse.Namespace):
        self.engine = engine
        self.args = args
        self.board = chess.Board()
        
        # Threading state
        self.search_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Shared context for the engine (stop flags, ponder state)
        self.search_context: Dict[str, Any] = {
            "stop_flag": False,
            "pondering": False,
            "ponderhit_time": 0.0
        }

    def run(self):
        """Main input loop."""
        logger.info("ChessNet-3070 UCI Server Started")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break # EOF
                
                line = line.strip()
                if not line:
                    continue

                self._process_command(line)
                
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("Unexpected error in UCI loop")

    def _process_command(self, line: str):
        """Parses and dispatches UCI commands."""
        tokens = line.split()
        if not tokens:
            return
            
        cmd = tokens[0]
        
        if cmd == "uci":
            self._send("id name ChessNet-3070")
            self._send("id author HybridBeast Team")
            # Advertise options here if implemented (e.g. Hash size)
            self._send("uciok")
            
        elif cmd == "isready":
            # Load lazy resources if needed
            self._send("readyok")
            
        elif cmd == "ucinewgame":
            self._stop_search()
            self.board.reset()
            self.engine.clear_tt() if hasattr(self.engine, 'clear_tt') else None
            
        elif cmd == "position":
            self._handle_position(tokens)
            
        elif cmd == "go":
            self._handle_go(tokens)
            
        elif cmd == "stop":
            self._stop_search()
            
        elif cmd == "ponderhit":
            self._handle_ponderhit()
            
        elif cmd == "quit":
            self._stop_search()
            sys.exit(0)

    def _handle_position(self, tokens: List[str]):
        """Parses: position [fen <fenstring> | startpos] moves <move1> ..."""
        try:
            moves_start = -1
            if "moves" in tokens:
                moves_start = tokens.index("moves")

            # 1. Setup Base Board
            if tokens[1] == "startpos":
                self.board.reset()
            elif tokens[1] == "fen":
                # FEN can contain spaces, so we grab everything up to 'moves'
                if moves_start == -1:
                    fen_str = " ".join(tokens[2:])
                else:
                    fen_str = " ".join(tokens[2:moves_start])
                
                self.board = chess.Board(fen_str)

            # 2. Apply Moves
            if moves_start != -1:
                for move_uci in tokens[moves_start + 1:]:
                    self.board.push(chess.Move.from_uci(move_uci))
                    
        except Exception:
            logger.error(f"Invalid position command: {' '.join(tokens)}")

    def _handle_go(self, tokens: List[str]):
        """Parses: go wtime 60000 btime 60000 ..."""
        # Stop existing search first
        self._stop_search()
        
        # Parse params
        params = {
            "wtime": None, "btime": None, 
            "winc": 0, "binc": 0,
            "ponder": False
        }
        
        iterator = iter(tokens[1:])
        try:
            for token in iterator:
                if token in params and token != "ponder":
                    params[token] = int(next(iterator))
                elif token == "ponder":
                    params["ponder"] = True
        except StopIteration:
            pass

        # Reset flags
        self.search_context["stop_flag"] = False
        self.search_context["pondering"] = params["ponder"]
        self.search_context["ponderhit_time"] = 0.0

        # Launch Search Thread
        self.search_thread = threading.Thread(
            target=self._search_worker,
            args=(self.board.copy(), params),
            daemon=True,
            name="SearchThread"
        )
        self.search_thread.start()

    def _handle_ponderhit(self):
        """Transition from pondering to normal search."""
        if self.search_context["pondering"]:
            logger.info("Ponderhit received - switching to live timer.")
            self.search_context["pondering"] = False
            self.search_context["ponderhit_time"] = time.time()

    def _stop_search(self):
        """ signals the search thread to stop and waits for join."""
        if self.search_thread and self.search_thread.is_alive():
            logger.info("Stopping search...")
            self.search_context["stop_flag"] = True
            self.search_thread.join(timeout=1.0)
            self.search_thread = None

    def _search_worker(self, board: chess.Board, params: Dict):
        """Threaded worker to run the heavy engine search."""
        try:
            logger.info(f"Starting search: {params}")
            
            best_move = self.engine.select_move(
                board,
                sims=self.args.sims,
                cpuct=self.args.cpuct,
                discount=self.args.discount,
                material_weight=self.args.material,
                batch_size=self.args.batch_size,
                wtime=params["wtime"],
                btime=params["btime"],
                winc=params["winc"],
                binc=params["binc"],
                search_context=self.search_context,
                piece_values=CUSTOM_PIECE_VALUES
            )
            
            # Send result to GUI
            # Note: If we were pondering and got stopped without a hit, we shouldn't send 'bestmove' usually,
            # but standard UCI engines often do send the best move found so far.
            self._send(f"bestmove {best_move.uci()}")
            
        except Exception:
            logger.exception("Search worker crashed")
            # Fallback to avoid hanging the GUI
            if board.legal_moves:
                fallback = list(board.legal_moves)[0]
                self._send(f"bestmove {fallback.uci()}")

    def _send(self, msg: str):
        """Sends a message to the GUI."""
        print(msg, flush=True)
        logger.debug(f">> {msg}")


def main():
    parser = argparse.ArgumentParser(description="ChessNet-3070 UCI Interface")
    parser.add_argument("--model", type=Path, default="checkpoints/baseline/chessnet_epoch9.pt")
    parser.add_argument("--book", type=Path, default="books/opening_book.bin")
    parser.add_argument("--sims", type=int, default=80000, help="Max MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=SEARCH_BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--cpuct", type=float, default=1.25)
    parser.add_argument("--discount", type=float, default=0.90)
    parser.add_argument("--material", type=float, default=0.15)
    args = parser.parse_args()

    # Load Engine
    try:
        model_cfg = ModelConfig()
        
        # Validate paths
        book_path = args.book if args.book.exists() else None
        if not args.model.exists():
            logger.critical(f"Model checkpoint not found: {args.model}")
            sys.exit(1)

        engine = HybridEngine(
            checkpoint_path=args.model,
            device="cuda",
            model_cfg=model_cfg,
            book_path=book_path
        )
        
    except Exception:
        logger.exception("Failed to initialize engine")
        sys.exit(1)

    # Start UCI Loop
    server = UCIServer(engine, args)
    server.run()


if __name__ == "__main__":
    main()
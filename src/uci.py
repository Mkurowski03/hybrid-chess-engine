import argparse
import logging
import sys
from pathlib import Path

import chess

# Allow direct imports when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.hybrid_engine import HybridEngine

logging.basicConfig(
    filename='live_engine.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

PIECE_VALUES = {"1": 1, "2": 3, "3": 3, "4": 5, "5": 9}

def main() -> None:
    """Run the UCI protocol."""
    parser = argparse.ArgumentParser(description="ChessNet-3070 UCI Engine")
    parser.add_argument("--model", type=str, default="checkpoints/baseline/chessnet_epoch9.pt", help="Path to checkpoint")
    parser.add_argument("--sims", type=int, default=600, help="Number of MCTS simulations")
    parser.add_argument("--cpuct", type=float, default=1.25, help="PUCT exploration constant")
    parser.add_argument("--discount", type=float, default=0.90, help="Checkmate discount factor")
    parser.add_argument("--material", type=float, default=0.15, help="Material weight")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for GPU MCTS")
    parser.add_argument("--book", type=str, default="books/opening_book.bin", help="Path to Polyglot opening book")
    args = parser.parse_args()

    # Load engine once at startup
    try:
        checkpoint = args.model
        book_path = args.book if Path(args.book).is_file() else None
        # Optionally, override device if needed
        model_cfg = ModelConfig()
        engine = HybridEngine(checkpoint, device="cuda", model_cfg=model_cfg, book_path=book_path)
    except Exception as e:
        msg = f"Failed to load checkpoint '{checkpoint}': {e}"
        sys.stderr.write(f"Error loading engine: {msg}\n")
        sys.exit(1)

    board = chess.Board()

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue
                
            tokens = line.split()
            command = tokens[0]

            if command == "uci":
                print("id name ChessNet-3070", flush=True)
                print("id author You", flush=True)
                print("uciok", flush=True)
                
            elif command == "isready":
                print("readyok", flush=True)
                
            elif command == "ucinewgame":
                board = chess.Board()
                
            elif command == "position":
                if len(tokens) > 1 and tokens[1] == "fen":
                    # Reconstruct the FEN
                    # "position fen <fen> [moves ...]"
                    try:
                        moves_idx = tokens.index("moves")
                        fen = " ".join(tokens[2:moves_idx])
                        board = chess.Board(fen)
                        # Apply moves
                        for move_uci in tokens[moves_idx + 1:]:
                            board.push(chess.Move.from_uci(move_uci))
                    except ValueError:
                        # No "moves" found
                        fen = " ".join(tokens[2:])
                        board = chess.Board(fen)
                        
                elif len(tokens) > 1 and tokens[1] == "startpos":
                    board = chess.Board()
                    if len(tokens) > 2 and tokens[2] == "moves":
                        for move_uci in tokens[3:]:
                            board.push(chess.Move.from_uci(move_uci))
                            
            elif command == "go":
                # Start MCTS search using HybridEngine
                wtime, btime, winc, binc = None, None, 0, 0
                for i in range(1, len(tokens)):
                    try:
                        if tokens[i] == 'wtime': wtime = int(tokens[i+1])
                        elif tokens[i] == 'btime': btime = int(tokens[i+1])
                        elif tokens[i] == 'winc': winc = int(tokens[i+1])
                        elif tokens[i] == 'binc': binc = int(tokens[i+1])
                    except (IndexError, ValueError):
                        pass

                logging.info(f"[START] Time allocated: wtime={wtime}, btime={btime}, winc={winc}, binc={binc}")

                best_move = engine.select_move(
                    board, 
                    sims=args.sims, 
                    cpuct=args.cpuct,
                    material_weight=args.material,
                    discount=args.discount,
                    batch_size=args.batch_size,
                    wtime=wtime,
                    btime=btime,
                    winc=winc,
                    binc=binc
                )
                print(f"bestmove {best_move.uci()}", flush=True)
                
            elif command == "quit":
                break
                
        except EOFError:
            break
        except Exception as e:
            sys.stderr.write(f"Error handling command '{line}': {e}\n")

if __name__ == "__main__":
    main()

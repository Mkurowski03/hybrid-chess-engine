import sys
import argparse
import chess
from engine import ChessEngine

PIECE_VALUES = {"1": 1, "2": 3, "3": 3, "4": 5, "5": 9}

def main():
    parser = argparse.ArgumentParser(description="ChessNet-3070 UCI Engine")
    parser.add_argument("--model", type=str, default="checkpoints/baseline/chessnet_epoch9.pt", help="Path to checkpoint")
    parser.add_argument("--sims", type=int, default=600, help="Number of MCTS simulations")
    parser.add_argument("--cpuct", type=float, default=1.25, help="PUCT exploration constant")
    parser.add_argument("--discount", type=float, default=0.90, help="Checkmate discount factor")
    parser.add_argument("--material", type=float, default=0.15, help="Material weight")
    args = parser.parse_args()

    # Load engine once at startup
    try:
        engine = ChessEngine(args.model)
    except Exception as e:
        sys.stderr.write(f"Error loading engine: {e}\n")
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
                print("id name ChessNet-3070")
                print("id author You")
                print("uciok")
                sys.stdout.flush()
                
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
                
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
                # Start MCTS search using parsed config
                kwargs = {
                    'cpuct': args.cpuct,
                    'material_weight': args.material,
                    'discount': args.discount,
                    'piece_values': {int(k): float(v) for k, v in PIECE_VALUES.items()}
                }
                
                best_move = engine.select_move(board, simulations=args.sims, **kwargs)
                print(f"bestmove {best_move.uci()}")
                sys.stdout.flush()
                
            elif command == "quit":
                break
                
        except EOFError:
            break
        except Exception as e:
            sys.stderr.write(f"Error handling command '{line}': {e}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ChessNet-3070 Web Server.

Exposes the HybridEngine via REST API for the web frontend.
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import chess
import chess.pgn
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# Inject project root for imports (works from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_engine import HybridEngine
from config import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# Global Engine Instance
# We use a container pattern if we needed thread safety, but for a simple 
# play server, a global reference is standard.
ENGINE: Optional[HybridEngine] = None
DEFAULT_BOOK = Path("books/opening_book.bin")


def load_engine_from_checkpoint(path: Optional[str], book_path: Optional[str] = None):
    """
    Initializes the engine, automatically loading architecture config 
    if a config.json exists in the checkpoint directory.
    """
    global ENGINE
    
    if not path:
        logger.warning("Initializing engine with random weights (No checkpoint provided)")
        ENGINE = HybridEngine(checkpoint_path=None, book_path=book_path)
        return

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Auto-detect configuration from training metadata
    model_cfg = ModelConfig()
    config_json = ckpt_path.parent / "config.json"
    
    if config_json.exists():
        try:
            with open(config_json) as f:
                saved_cfg = json.load(f)
                model_params = saved_cfg.get("model", {})
                
                # Update config with saved params
                if "num_filters" in model_params:
                    model_cfg.num_filters = model_params["num_filters"]
                if "num_residual_blocks" in model_params:
                    model_cfg.num_residual_blocks = model_params["num_residual_blocks"]
            logger.info(f"Loaded architecture config from {config_json.name}")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")

    logger.info(f"Loading checkpoint: {ckpt_path.name}")
    ENGINE = HybridEngine(checkpoint_path=ckpt_path, model_cfg=model_cfg, book_path=book_path)


def parse_board_input(data: Dict[str, Any]) -> chess.Board:
    """Helper to extract a board state from FEN or PGN."""
    fen = data.get("fen")
    pgn = data.get("pgn")

    if pgn:
        try:
            pgn_io = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_io)
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            return board
        except Exception:
            pass # Fallback to FEN

    if fen:
        return chess.Board(fen)
        
    raise ValueError("No valid FEN or PGN provided")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/move", methods=["POST"])
def api_move():
    """Calculates the best move."""
    if not ENGINE:
        return jsonify({"error": "Engine not initialized"}), 500

    try:
        board = parse_board_input(request.get_json())
    except ValueError:
        return jsonify({"error": "Invalid FEN/PGN"}), 400

    if board.is_game_over():
        return jsonify({"game_over": True, "result": board.result()})

    data = request.get_json()
    
    # Engine parameters
    params = {
        "sims": data.get("sims", 800),
        "cpuct": data.get("cpuct", 1.25),
        "material_weight": data.get("material", 0.15),
        "discount": data.get("discount", 0.90),
        "wtime": data.get("wtime"),
        "btime": data.get("btime"),
        "winc": 2000, # Assume standard increments for web play
        "binc": 2000
    }

    start = float(data.get("wtime") or 0)
    move = ENGINE.select_move(board, **params)
    
    # Evaluation Logic
    raw_eval = ENGINE.evaluate(board)
    eval_white = raw_eval if board.turn == chess.WHITE else -raw_eval
    
    # Retrieve PV if available
    pv = getattr(ENGINE, 'last_pv', [])

    return jsonify({
        "move": move.uci(),
        "san": board.san(move),
        "evaluation": round(eval_white, 2),
        "pv": pv,
        "game_over": False
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Returns static evaluation of a position."""
    if not ENGINE:
        return jsonify({"error": "Engine not initialized"}), 500

    try:
        board = parse_board_input(request.get_json())
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    raw_eval = ENGINE.evaluate(board)
    eval_white = raw_eval if board.turn == chess.WHITE else -raw_eval

    return jsonify({"evaluation": round(eval_white, 3)})


@app.route("/api/top_moves", methods=["POST"])
def api_top_moves():
    """Analyzes the top N candidate moves (Policy Head)."""
    if not ENGINE:
        return jsonify({"error": "Engine not initialized"}), 500

    try:
        data = request.get_json()
        board = parse_board_input(data)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    n = data.get("n", 5)
    moves = ENGINE.top_moves(board, n=n)
    
    raw_eval = ENGINE.evaluate(board)
    eval_white = raw_eval if board.turn == chess.WHITE else -raw_eval

    return jsonify({
        "moves": moves,
        "evaluation": round(eval_white, 3)
    })


@app.route("/api/checkpoints", methods=["GET"])
def api_list_checkpoints():
    """Scans the checkpoints directory."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return jsonify([])

    results = []
    for f in checkpoints_dir.rglob("*.pt"):
        results.append({
            "name": f"{f.parent.name}/{f.name}",
            "path": str(f)
        })
    # Also look for ONNX models
    for f in checkpoints_dir.rglob("*.onnx"):
        results.append({
            "name": f"{f.parent.name}/{f.name} (ONNX)",
            "path": str(f)
        })
        
    return jsonify(sorted(results, key=lambda x: x['name']))


@app.route("/api/load_checkpoint", methods=["POST"])
def api_load_checkpoint():
    """Hot-swap the engine model."""
    data = request.get_json()
    path = data.get("path")
    
    try:
        load_engine_from_checkpoint(path, book_path=str(DEFAULT_BOOK))
        return jsonify({"status": "success", "loaded": path})
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="ChessNet Web API")
    parser.add_argument("--checkpoint", type=str, help="Path to initial model checkpoint")
    parser.add_argument("--book", type=str, default=str(DEFAULT_BOOK), help="Path to opening book")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Initial Load
    book_file = args.book if Path(args.book).exists() else None
    
    # Try to find a checkpoint if none provided
    ckpt = args.checkpoint
    if not ckpt:
        candidates = sorted(Path("checkpoints").rglob("*.pt"))
        if candidates:
            ckpt = str(candidates[-1])
            logger.info(f"Auto-selected latest checkpoint: {ckpt}")

    load_engine_from_checkpoint(ckpt, book_path=book_file)

    logger.info(f"Server running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
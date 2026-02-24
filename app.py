#!/usr/bin/env python3
"""Flask server for playing chess against ChessNet-3070.

Usage
-----
    python app.py --checkpoint checkpoints/baseline/best.pt
    python app.py --checkpoint checkpoints/baseline/chessnet_epoch0.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import chess
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.hybrid_engine import HybridEngine
from config import ModelConfig

app = Flask(__name__, static_folder="static")
CORS(app)

# Load Engine
CHECKPOINT_PATH = "checkpoints/baseline/chessnet_epoch9.pt"
BOOK_PATH = "books/opening_book.bin"

if not Path(BOOK_PATH).exists():
    BOOK_PATH = None

engine = HybridEngine(checkpoint_path=CHECKPOINT_PATH, book_path=BOOK_PATH)


def _find_checkpoints() -> list[dict]:
    """Discover all available checkpoints."""
    results = []
    cp_dir = Path("checkpoints")
    if not cp_dir.exists():
        return results
    for exp_dir in sorted(cp_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for pt_file in sorted(exp_dir.glob("*.pt")):
            results.append({
                "name": f"{exp_dir.name}/{pt_file.name}",
                "path": str(pt_file),
            })
    return results


@app.route("/")
def index() -> Response:
    """Serve the index page."""
    return send_from_directory("static", "index.html")


@app.route("/api/move", methods=["POST"])
def api_move() -> tuple[Response, int] | Response:
    """Receive a FEN, return the engine's best move + evaluation."""
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "Missing 'fen'"}), 400

    try:
        pgn = data.get("pgn")
        if pgn:
            import io
            pgn_io = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_io)
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
        else:
            board = chess.Board(fen)
    except Exception:
        # Fallback to FEN if PGN fails
        try:
             board = chess.Board(fen)
        except ValueError:
             return jsonify({"error": "Invalid FEN/PGN"}), 400

    if board.is_game_over():
        return jsonify({
            "game_over": True,
            "result": board.result(),
        })

    depth = data.get("depth")
    sims = data.get("sims", 2200)
    cpuct = data.get("cpuct", 1.25)
    material = data.get("material", 0.15)
    discount = data.get("discount", 0.99)
    
    wtime = data.get("wtime", 300000) # Give it 5 virtual minutes to think
    btime = data.get("btime", 300000)
    
    move = engine.select_move(
        board, 
        sims=sims, 
        cpuct=cpuct, 
        material_weight=material, 
        discount=discount,
        batch_size=64,
        wtime=wtime,
        btime=btime,
        winc=2000,
        binc=2000
    )
    evaluation = engine.evaluate(board)

    # Evaluation from white's perspective for the eval bar
    eval_white = evaluation if board.turn == chess.WHITE else -evaluation

    return jsonify({
        "move": move.uci(),
        "san": board.san(move),
        "evaluation": round(eval_white, 3),
        "game_over": False,
    })


@app.route("/api/top_moves", methods=["POST"])
def api_top_moves() -> tuple[Response, int] | Response:
    """Return top-N candidate moves with probabilities."""
    data = request.get_json()
    fen = data.get("fen")
    n = data.get("n", 5)
    sims = data.get("sims", 600)
    cpuct = data.get("cpuct", 1.25)
    material = data.get("material", 0.15)
    discount = data.get("discount", 0.99)
    
    if not fen:
        return jsonify({"error": "Missing 'fen'"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    moves = engine.top_moves(
        board, 
        n=n,
        sims=sims,
        cpuct=cpuct,
        material_weight=material,
        discount=discount
    )
    evaluation = engine.evaluate(board)
    eval_white = evaluation if board.turn == chess.WHITE else -evaluation

    return jsonify({
        "moves": moves,
        "evaluation": round(eval_white, 3),
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate() -> tuple[Response, int] | Response:
    """Return evaluation for a position."""
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "Missing 'fen'"}), 400

    board = chess.Board(fen)
    evaluation = engine.evaluate(board)
    eval_white = evaluation if board.turn == chess.WHITE else -evaluation

    return jsonify({"evaluation": round(eval_white, 3)})


@app.route("/api/checkpoints")
def api_checkpoints() -> Response:
    """List available model checkpoints."""
    return jsonify(_find_checkpoints())


@app.route("/api/load_checkpoint", methods=["POST"])
def api_load_checkpoint() -> tuple[Response, int] | Response:
    """Switch to a different checkpoint."""
    global engine
    data = request.get_json()
    path = data.get("path")
    if not path or not Path(path).exists():
        return jsonify({"error": "Checkpoint not found"}), 404

    # Read model config if available
    exp_dir = Path(path).parent
    model_cfg = ModelConfig()
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_section = cfg.get("model", {})
        if "num_filters" in model_section:
            model_cfg.num_filters = model_section["num_filters"]
        if "num_residual_blocks" in model_section:
            model_cfg.num_residual_blocks = model_section["num_residual_blocks"]

    engine = HybridEngine(checkpoint_path=path, model_cfg=model_cfg, book_path=BOOK_PATH)
    return jsonify({"status": "loaded", "path": path})


def main() -> None:
    """Run the Flask server."""
    global engine

    parser = argparse.ArgumentParser(description="ChessNet-3070 Play Server")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--book", type=str, default="books/opening_book.bin",
                        help="Path to Polyglot opening book .bin file")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    # Load engine
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Auto-find latest checkpoint
        candidates = sorted(Path("checkpoints").rglob("*.pt"))
        if candidates:
            checkpoint = str(candidates[-1])
            logging.info(f"Auto-selected: {checkpoint}")

    if checkpoint:
        # Read model config if available
        exp_dir = Path(checkpoint).parent
        model_cfg = ModelConfig()
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            model_section = cfg.get("model", {})
            if "num_filters" in model_section:
                model_cfg.num_filters = model_section["num_filters"]
            if "num_residual_blocks" in model_section:
                model_cfg.num_residual_blocks = model_section["num_residual_blocks"]
        engine = HybridEngine(checkpoint_path=checkpoint, model_cfg=model_cfg, book_path=args.book)
        logging.info(f"Model loaded: {checkpoint}")
    else:
        engine = HybridEngine(checkpoint_path=None, book_path=args.book)
        logging.warning("No checkpoint found — using random weights!")

    logging.info(f"Open http://{args.host}:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

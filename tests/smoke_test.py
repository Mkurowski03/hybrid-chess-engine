"""Smoke tests for ChessNet-3070 core components.

Run with:  python -m pytest tests/smoke_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

# Make project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.board_encoder import encode_board
from src.model import ChessNet, build_model, count_params
from src.engine import ChessEngine


# ── Board Encoder ──────────────────────────────────────────────────────────


class TestBoardEncoder:
    """Tests for ``src.board_encoder.encode_board``."""

    def test_shape_and_dtype(self) -> None:
        board = chess.Board()
        enc = encode_board(board)
        assert enc.shape == (18, 8, 8), f"Expected (18,8,8), got {enc.shape}"
        assert enc.dtype == np.float32

    def test_starting_position_white_pawns(self) -> None:
        """Channel 0 should have White's pawns on rank-1 (row index 1)."""
        board = chess.Board()
        enc = encode_board(board)
        # Channel 0 = our pawns (White to move → our = White)
        pawn_plane = enc[0]
        # Rank 2 in chess = row index 1 in the array (a2..h2)
        assert pawn_plane[1].sum() == 8.0, "Expected 8 pawns on rank 2"
        assert pawn_plane[0].sum() == 0.0, "No pawns on rank 1"

    def test_canonical_flip_for_black(self) -> None:
        """When it's Black's turn the board should be flipped."""
        board = chess.Board()
        board.push_san("e4")  # Now it's Black's turn.
        enc = encode_board(board)
        # Channel 0 = "our" pawns → Black's pawns, but flipped to bottom.
        # After flip, Black pawns (originally rank 7) should appear on row 1.
        pawn_plane = enc[0]
        assert pawn_plane[1].sum() == 8.0, "After flip 8 Black pawns should be on row 1"

    def test_castling_rights_initial(self) -> None:
        board = chess.Board()
        enc = encode_board(board)
        # White to move: channels 13-16 should all be 1 (all rights available).
        for ch in range(13, 17):
            assert enc[ch, 0, 0] == 1.0, f"Channel {ch} should be 1.0 initially"

    def test_fifty_move_counter(self) -> None:
        board = chess.Board()
        board.halfmove_clock = 50
        enc = encode_board(board)
        expected = 50 / 100.0
        assert abs(enc[17, 0, 0] - expected) < 1e-6


# ── Model ──────────────────────────────────────────────────────────────────


class TestModel:
    """Tests for ``src.model.ChessNet``."""

    def test_forward_shape(self) -> None:
        model = build_model()
        x = torch.randn(1, 18, 8, 8)
        policy, value = model(x)
        assert policy.shape == (1, 4096), f"Policy shape: {policy.shape}"
        assert value.shape == (1, 1), f"Value shape: {value.shape}"

    def test_value_range(self) -> None:
        model = build_model()
        x = torch.randn(4, 18, 8, 8)
        _, value = model(x)
        assert value.min().item() >= -1.0 - 1e-6
        assert value.max().item() <= 1.0 + 1e-6

    def test_batch_forward(self) -> None:
        model = build_model()
        x = torch.randn(8, 18, 8, 8)
        p, v = model(x)
        assert p.shape == (8, 4096)
        assert v.shape == (8, 1)

    def test_param_count(self) -> None:
        model = build_model()
        n = count_params(model)
        assert n < 5_000_000, f"Too many params: {n:,}"
        print(f"  Model has {n:,} trainable parameters")


# ── Engine ─────────────────────────────────────────────────────────────────


class TestEngine:
    """Tests for ``src.engine.ChessEngine``."""

    def test_select_move_returns_legal_move(self) -> None:
        """From the starting position the engine must return a legal move."""
        engine = ChessEngine(checkpoint_path=None, book_path=None, device="cpu")
        board = chess.Board()
        move = engine.select_move(board)
        assert move in board.legal_moves, f"{move} is not legal"

    def test_evaluate_returns_float(self) -> None:
        engine = ChessEngine(checkpoint_path=None, book_path=None, device="cpu")
        board = chess.Board()
        val = engine.evaluate(board)
        assert -1.0 <= val <= 1.0, f"Value out of range: {val}"

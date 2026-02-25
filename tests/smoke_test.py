"""
Unit and Integration Tests for ChessNet-3070.

Usage:
    pytest tests/smoke_test.py -v
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig
from src.board_encoder import encode_board
from src.model import build_model, count_params
from src.engine import ChessEngine

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST")


class TestBoardEncoder:
    """Validates the FEN -> Tensor encoding logic."""

    def test_tensor_shape_and_dtype(self):
        board = chess.Board()
        enc = encode_board(board)
        
        assert enc.shape == (18, 8, 8), f"Expected (18,8,8), got {enc.shape}"
        assert enc.dtype == np.float32, f"Expected float32, got {enc.dtype}"

    def test_perspective_transform(self):
        """
        Ensures the board is always presented from the perspective of the active player.
        """
        # Case 1: White to move (Standard)
        board = chess.Board()
        enc_white = encode_board(board)
        # White pawns on rank 2 (index 1)
        assert enc_white[0, 1, :].sum() == 8.0, "White pawns missing on Rank 2"

        # Case 2: Black to move (Flipped)
        board.push_san("e4") # Now Black's turn
        enc_black = encode_board(board)
        
        # Channel 0 is "Our Pawns". For Black, "Our Pawns" are on Rank 7.
        # But the encoder flips the board vertically, so they should appear on Rank 2 (index 1).
        assert enc_black[0, 1, :].sum() == 8.0, "Black pawns not flipped correctly to Rank 2"

    def test_auxiliary_planes(self):
        """Checks Castling Rights, Repetition, and 50-move rule."""
        board = chess.Board()
        enc = encode_board(board)
        
        # Castling Rights (Channels 13-16) - All true at start
        assert enc[13:17].sum() == 4 * 64, "Initial castling rights invalid"

        # 50-move rule (Channel 17)
        board.halfmove_clock = 50
        enc_50 = encode_board(board)
        expected = 0.5 # 50 / 100
        assert np.allclose(enc_50[17], expected), "50-move counter plane incorrect"


class TestModelArchitecture:
    """Validates Neural Network structure and inference flow."""

    @pytest.fixture(scope="class")
    def model(self):
        return build_model(ModelConfig())

    def test_forward_pass(self, model):
        """Test random input through the network."""
        batch_size = 4
        x = torch.randn(batch_size, 18, 8, 8)
        
        policy, value = model(x)
        
        assert policy.shape == (batch_size, 4096), "Policy output shape mismatch"
        assert value.shape == (batch_size, 1), "Value output shape mismatch"
        
        # Value range check (Tanh output)
        assert value.min() >= -1.0, "Value < -1.0"
        assert value.max() <= 1.0, "Value > 1.0"

    def test_parameter_count(self, model):
        params = count_params(model)
        logger.info(f"Model Parameters: {params:,}")
        assert params > 0, "Model has no parameters"
        assert params < 10_000_000, "Model is unexpectedly large"


class TestChessEngine:
    """Integration tests for the Engine logic (MCTS + Inference)."""

    @pytest.fixture(scope="class")
    def engine(self):
        # Use CPU for tests to avoid CUDA overhead/errors in CI
        return ChessEngine(checkpoint_path=None, device="cpu")

    def test_legal_move_generation(self, engine):
        """Engine must return a valid move from startpos."""
        board = chess.Board()
        move = engine.select_move(board, simulations=10) # Low sims for speed
        assert move in board.legal_moves, f"Engine returned illegal move: {move}"

    def test_mate_in_one_detection(self, engine):
        """
        Engine should find immediate mate (handled by MCTS or Mate Guard).
        """
        # Fool's Mate pattern (White to move and mate)
        # Position: White Queen on h5, Black King on e8 unprotectable... 
        # Actually let's use a simpler forced mate: K+Q vs K
        fen = "8/8/8/8/8/5k2/4Q3/4K3 w - - 0 1" 
        # White Q e2, Black K f3. 
        # Setup specific mate in 1: 
        fen = "k7/P7/K7/8/8/8/8/8 w - - 0 1" # Stalemate trap? No.
        
        # Simple Back Rank Mate
        fen = "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1" # White Rook d1 to d8 is mate
        board = chess.Board(fen)
        
        move = engine.select_move(board, simulations=50)
        assert move.uci() == "d1d8", f"Failed to find back-rank mate. Got {move}"

    def test_evaluation_consistency(self, engine):
        """Evaluate should return a valid float."""
        board = chess.Board()
        val = engine.evaluate(board)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0
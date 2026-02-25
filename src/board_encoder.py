"""
Board Encoding Module for ChessNet-3070.

Converts a python-chess Board into an (18, 8, 8) float32 tensor
with perspective-relative piece planes and auxiliary channels.

Also provides the canonical ``move_to_policy_index`` mapping used by
MCTS, the pure-Python engine, and the Hybrid Rust/Python engine.
"""

from __future__ import annotations

import chess
import numpy as np

# Constants for channel ordering
# Channels 0-5: Our pieces
# Channels 6-11: Their pieces
PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING
]

def get_bitboard_plane(mask: int) -> np.ndarray:
    """
    Converts a bitmask integer into an 8x8 float32 plane.
    """
    plane = np.zeros(64, dtype=np.float32)
    if mask:
        # chess.SquareSet is the standard, readable way to iterate bits
        for sq in chess.SquareSet(mask):
            plane[sq] = 1.0
    return plane.reshape(8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes a chess board into an (18, 8, 8) float32 tensor.
    
    Format:
        Channels 0-5:   Our pieces (P, N, B, R, Q, K)
        Channels 6-11:  Opponent pieces
        Channel 12:     Repetition count (0.0, 0.5, 1.0)
        Channel 13-16:  Castling rights (K_us, Q_us, K_them, Q_them)
        Channel 17:     50-move rule (normalized)
        
    Orientation:
        The board is always oriented from the perspective of the side to move.
        If it's Black's turn, the board is flipped vertically.
    """
    # 1. Setup Perspective
    us = board.turn
    them = not us
    
    # We construct planes in the standard orientation first, then flip if necessary
    planes = []

    # 2. Piece Planes (0-11)
    for color in [us, them]:
        for piece_type in PIECE_ORDER:
            mask = board.pieces_mask(piece_type, color)
            plane = get_bitboard_plane(mask)
            planes.append(plane)

    # 3. Aux Planes
    
    # Repetition (Channel 12)
    # 3-fold is usually game over, so we care about 1 and 2 repetitions.
    # Normalizing: 0 -> 0.0, 1 -> 0.5, 2 -> 1.0
    if board.is_repetition(3):
        rep_val = 1.0
    elif board.is_repetition(2):
        rep_val = 0.5
    else:
        rep_val = 0.0
    
    planes.append(np.full((8, 8), rep_val, dtype=np.float32))

    # Castling Rights (Channels 13-16)
    # Order: Our King, Our Queen, Their King, Their Queen
    castling_values = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them)
    ]
    
    for right in castling_values:
        val = 1.0 if right else 0.0
        planes.append(np.full((8, 8), val, dtype=np.float32))

    # 50-Move Rule (Channel 17)
    halfmove_val = min(board.halfmove_clock / 100.0, 1.0)
    planes.append(np.full((8, 8), halfmove_val, dtype=np.float32))

    # 4. Stack and Flip
    tensor = np.stack(planes)  # Shape (18, 8, 8)

    # If it is Black's turn, we flip the board vertically (Rank 1 becomes Rank 8)
    # to maintain a "canonical" perspective for the neural network.
    if us == chess.BLACK:
        tensor = np.flip(tensor, axis=1)

    return tensor.astype(np.float32)


def encode_board_tensor(board: chess.Board):
    """
    Wrapper to return a PyTorch tensor (CPU) directly.
    """
    import torch
    np_array = encode_board(board)
    return torch.from_numpy(np_array)


# ---------------------------------------------------------------------------
# Move ↔ Policy Index Mapping
# ---------------------------------------------------------------------------

def move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
    """Maps a chess move to the flat policy index (0-4095).

    The encoding is ``from_sq * 64 + to_sq`` after mirroring squares
    for Black so the neural network always sees a canonical orientation.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    return from_sq * 64 + to_sq
"""Board encoder: chess.Board → (18, 8, 8) tensor via bitboard extraction.

Uses python-chess bitboard internals for speed — no per-square iteration.
Board is always rendered from the perspective of the side to move
(flip + colour-swap when it is Black's turn).
"""

from __future__ import annotations

import chess
import numpy as np


# Piece ordering used in channels 0-5 (ours) and 6-11 (theirs).
_PIECE_TYPES: list[int] = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

# Pre-computed lookup: bit-index → (row, col) for an 8×8 board.
# Bit 0 = a1 = (0,0), bit 63 = h8 = (7,7).
_BIT_TO_ROW = np.zeros(64, dtype=np.intp)
_BIT_TO_COL = np.zeros(64, dtype=np.intp)
for _sq in range(64):
    _BIT_TO_ROW[_sq] = _sq >> 3     # _sq // 8
    _BIT_TO_COL[_sq] = _sq & 0x07   # _sq % 8


def _bb_to_plane(bb: int) -> np.ndarray:
    """Convert a 64-bit bitboard to an (8, 8) binary numpy array.

    Uses ``numpy`` vectorised operations — no Python-level square loop.
    """
    if bb == 0:
        return np.zeros((8, 8), dtype=np.float32)

    # Extract set-bit indices.
    indices = np.array(_extract_bits(bb), dtype=np.intp)
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[_BIT_TO_ROW[indices], _BIT_TO_COL[indices]] = 1.0
    return plane


def _extract_bits(bb: int) -> list[int]:
    """Return a list of set-bit positions in *bb* (LSB-first)."""
    bits: list[int] = []
    while bb:
        lsb = bb & -bb          # isolate lowest set bit
        bits.append(lsb.bit_length() - 1)
        bb ^= lsb               # clear it
    return bits


def _flip_plane(plane: np.ndarray) -> np.ndarray:
    """Flip an 8×8 plane vertically (rank mirror)."""
    return plane[::-1, :]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a ``chess.Board`` as an ``(18, 8, 8)`` float32 array.

    The encoding is **canonical**: the board is always viewed from the
    perspective of the side to move.  When it is Black's turn the board
    is flipped vertically and colours are swapped so that channels 0-5
    always represent "our" pieces and 6-11 "their" pieces.

    Channels
    --------
    0-5   : Our pieces (P, N, B, R, Q, K).
    6-11  : Opponent's pieces (P, N, B, R, Q, K).
    12    : Repetition count (0, 1, or 2 — normalised by dividing by 2).
    13    : Our kingside castling right.
    14    : Our queenside castling right.
    15    : Their kingside castling right.
    16    : Their queenside castling right.
    17    : Fifty-move rule counter (normalised to [0, 1]).
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    us = board.turn           # True = WHITE, False = BLACK
    them = not us
    flip = us == chess.BLACK  # need to flip when Black is to move

    # ---- Piece planes (channels 0-11) ----
    for idx, pt in enumerate(_PIECE_TYPES):
        our_bb = board.pieces_mask(pt, us)
        their_bb = board.pieces_mask(pt, them)
        our_plane = _bb_to_plane(our_bb)
        their_plane = _bb_to_plane(their_bb)

        if flip:
            our_plane = _flip_plane(our_plane)
            their_plane = _flip_plane(their_plane)

        planes[idx] = our_plane
        planes[idx + 6] = their_plane

    # ---- Repetition plane (channel 12) ----
    # python-chess exposes `board.is_repetition(count)`.
    if board.is_repetition(3):
        rep = 2
    elif board.is_repetition(2):
        rep = 1
    else:
        rep = 0
    planes[12] = rep / 2.0

    # ---- Castling rights (channels 13-16) ----
    if us == chess.WHITE:
        our_k = bool(board.castling_rights & chess.BB_H1)
        our_q = bool(board.castling_rights & chess.BB_A1)
        their_k = bool(board.castling_rights & chess.BB_H8)
        their_q = bool(board.castling_rights & chess.BB_A8)
    else:
        our_k = bool(board.castling_rights & chess.BB_H8)
        our_q = bool(board.castling_rights & chess.BB_A8)
        their_k = bool(board.castling_rights & chess.BB_H1)
        their_q = bool(board.castling_rights & chess.BB_A1)

    planes[13] = float(our_k)
    planes[14] = float(our_q)
    planes[15] = float(their_k)
    planes[16] = float(their_q)

    # ---- Fifty-move counter (channel 17) ----
    planes[17] = min(board.halfmove_clock / 100.0, 1.0)

    return planes


def encode_board_tensor(board: chess.Board) -> "torch.Tensor":  # noqa: F821
    """Convenience wrapper returning a ``torch.Tensor`` on CPU."""
    import torch

    return torch.from_numpy(encode_board(board))

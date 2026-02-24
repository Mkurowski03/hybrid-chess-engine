"""
ChessNet-3070 — Central Configuration
======================================
All hyper-parameters live here so every script imports the same values.
Tuned for an NVIDIA RTX 3070 Ti (8 GB VRAM, 6144 CUDA cores).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Model ──────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    in_channels: int = 18             # 6 piece-types × 2 colours + extras
    num_filters: int = 128            # width of every residual block
    num_residual_blocks: int = 10     # depth of the tower
    policy_output_dim: int = 4096     # 64 × 64  (from-square, to-square)
    value_hidden_dim: int = 256       # FC layer in the value head
    syzygy_path: str = "tablebases/"  # Default Syzygy directory


# ── Data / ETL ─────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    train_file: str = "train.h5"
    min_elo: int = 2200               # both players must be ≥ this
    min_time: int = 180               # base time-control in seconds
    max_positions_per_game: int = 10  # random sample per game
    opening_skip_prob: float = 0.5    # P(skip) for first 10 moves
    num_workers: int = 10             # parallel PGN parsing (Ryzen 7 5700X)
    chunk_size: int = 100_000         # HDF5 write-chunk


# ── Training ───────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int = 1024
    num_epochs: int = 30
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    patience: int = 5            # early stopping patience (0 = disabled)
    save_best: bool = True       # save best val-loss model separately
    dataloader_workers: int = 4
    log_every_n_steps: int = 100
    checkpoint_dir: Path = Path("checkpoints")
    wandb_project: str = "chessnet-3070"
    compile_model: bool = False  # torch.compile (disabled — Triton incompatible with 3070 Ti)
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

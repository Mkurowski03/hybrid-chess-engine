"""
Central Configuration for ChessNet-3070.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Engine Heuristics ---
# Pruning: Stop search if best move visits > 3.5x second best
SMART_PRUNING_FACTOR = 3.5
MIN_PRUNING_SIMS = 20_000

# Evaluation boost for simplifying into winning endgames
SIMPLIFICATION_FACTOR = 0.2

# Inference batch size (Optimized for RTX 3070 Ti)
SEARCH_BATCH_SIZE = 512


@dataclass
class ModelConfig:
    """Network architecture hyperparameters."""
    in_channels: int = 18          # 6 pieces * 2 colors + aux planes
    num_filters: int = 128         # ResNet width
    num_residual_blocks: int = 10  # ResNet depth
    policy_output_dim: int = 4096  # 64 * 64 move encoding
    value_hidden_dim: int = 256
    syzygy_path: str = "tablebases/"


@dataclass
class DataConfig:
    """ETL pipeline settings."""
    data_dir: Path = Path("data")
    train_file: str = "train.h5"
    
    # Filtering criteria
    min_elo: int = 2200
    min_time: int = 180            # Seconds
    min_ply_count: int = 30        # Skip short games
    
    # Sampling
    positions_per_game: int = 10
    opening_cutoff_move: int = 10
    opening_keep_prob: float = 0.5
    
    # Parallel processing
    num_workers: int = 10          # Tuned for Ryzen 7 5700X
    chunk_size: int = 100_000      # HDF5 write buffer


@dataclass
class TrainConfig:
    """Training loop configuration."""
    batch_size: int = 1024         # Fits 8GB VRAM with FP16
    num_epochs: int = 30
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    
    # Checkpointing
    patience: int = 5
    save_best: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    
    # Runtime
    dataloader_workers: int = 4
    log_every_n_steps: int = 100
    compile_model: bool = False    # Disable if causing Triton/CUDA issues
    
    # Loss scaling
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
#!/usr/bin/env python3
"""
Production-Grade Training Script for ChessNet-3070.

Designed for long-running sessions (12-48h) with full crash recovery.
Features: OneCycleLR, early stopping, graceful SIGINT handling,
auto-resume from last checkpoint, and detailed ETA logging.

Usage:
    python src/train_advanced.py --data data/memmap --name experiment_1
    python src/train_advanced.py --data data/memmap --name experiment_1 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Inject project root for imports (works from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig, TrainConfig
from src.model import build_model, count_params

# Configure logging (stdout + file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_advanced.log", mode='a'),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class SignalHandler:
    """Handles SIGINT (Ctrl+C) to allow graceful mid-epoch shutdown."""

    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if self.interrupted:
            logger.warning("Force quitting...")
            sys.exit(1)
        self.interrupted = True
        logger.warning(
            "\nReceived SIGINT. Finishing current epoch and saving checkpoint...\n"
            "Press Ctrl+C again to force quit."
        )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    """
    High-performance dataset using memory-mapped Numpy arrays.
    Allows instant random access to 24M+ positions without loading RAM.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        meta_path = data_dir / "meta.json"

        if meta_path.exists():
            with open(meta_path) as f:
                self.length = json.load(f)["n_positions"]
        else:
            self.length = len(np.load(data_dir / "policies.npy", mmap_mode="r"))

        # Lazy-loaded handles (initialized in worker processes)
        self.states = None
        self.policies = None
        self.values = None

    def _init_memmaps(self):
        if self.states is None:
            self.states = np.load(self.data_dir / "states.npy", mmap_mode="r")
            self.policies = np.load(self.data_dir / "policies.npy", mmap_mode="r")
            self.values = np.load(self.data_dir / "values.npy", mmap_mode="r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_memmaps()
        state = torch.from_numpy(self.states[idx].copy())
        policy = torch.tensor(self.policies[idx], dtype=torch.long)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value


class HDF5Dataset(Dataset):
    """Fallback dataset for raw HDF5 files (slower than memmap)."""

    def __init__(self, path: Path):
        import h5py
        self.path = path
        with h5py.File(path, "r") as f:
            self.length = f["states"].shape[0]
        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import h5py
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        state = torch.from_numpy(self.file["states"][idx])
        policy = torch.tensor(self.file["policies"][idx], dtype=torch.long)
        value = torch.tensor(self.file["values"][idx], dtype=torch.float32)
        return state, policy, value


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    best_loss: float,
):
    """Saves the full training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_loss': best_loss,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    sig: SignalHandler,
) -> Dict[str, float]:
    """Runs one training epoch. Returns averaged metrics."""
    model.train()
    total_loss = 0.0
    p_loss_sum = 0.0
    v_loss_sum = 0.0
    correct = 0
    n_samples = 0

    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()

    pbar = tqdm(loader, desc=f"Train Ep {epoch}", dynamic_ncols=True, leave=False)

    for i, (states, policies, values) in enumerate(pbar):
        if sig.interrupted:
            break

        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred_p, pred_v = model(states)
            loss_p = p_criterion(pred_p, policies)
            loss_v = v_criterion(pred_v, values)
            loss = (loss_p * cfg.policy_loss_weight) + (loss_v * cfg.value_loss_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = states.size(0)
        n_samples += bs
        total_loss += loss.item() * bs
        p_loss_sum += loss_p.item() * bs
        v_loss_sum += loss_v.item() * bs
        correct += (pred_p.argmax(dim=1) == policies).sum().item()

        if i % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}")

    return {
        "loss": total_loss / n_samples,
        "p_loss": p_loss_sum / n_samples,
        "v_loss": v_loss_sum / n_samples,
        "acc": 100.0 * correct / n_samples,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Runs one validation pass. Returns averaged metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()

    for states, policies, values in tqdm(loader, desc="Validating", leave=False):
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        with autocast("cuda"):
            pred_p, pred_v = model(states)
            loss_p = p_criterion(pred_p, policies)
            loss_v = v_criterion(pred_v, values)
            loss = (loss_p * cfg.policy_loss_weight) + (loss_v * cfg.value_loss_weight)

        bs = states.size(0)
        n_samples += bs
        total_loss += loss.item() * bs
        correct += (pred_p.argmax(dim=1) == policies).sum().item()

    return {
        "loss": total_loss / n_samples,
        "acc": 100.0 * correct / n_samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChessNet Advanced Training Pipeline")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to memmap directory or .h5 file")
    parser.add_argument("--name", type=str, default="baseline",
                        help="Experiment name (determines checkpoint subdirectory)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if one exists")

    # Overrides (fall back to TrainConfig defaults if not specified)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # ── Configuration ─────────────────────────────────────────────────────
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.max_lr is not None:
        train_cfg.learning_rate = args.max_lr
    if args.patience is not None:
        train_cfg.patience = args.patience
    train_cfg.dataloader_workers = args.workers

    exp_dir = train_cfg.checkpoint_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Persist active config for reproducibility
    with open(exp_dir / "config.json", "w") as f:
        json.dump(
            {"model": asdict(model_cfg), "train": asdict(train_cfg)},
            f, indent=2, default=str,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Experiment '{args.name}' on {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    if args.data.is_dir() and (args.data / "states.npy").exists():
        logger.info(f"Loading MemmapDataset from {args.data}")
        dataset = MemmapDataset(args.data)
    elif args.data.suffix in ('.h5', '.hdf5'):
        logger.info(f"Loading HDF5Dataset from {args.data}")
        dataset = HDF5Dataset(args.data)
    else:
        logger.error(f"Invalid data path: {args.data}. "
                     "Provide a memmap directory or .h5 file.")
        sys.exit(1)

    total_len = len(dataset)
    val_len = int(total_len * train_cfg.val_split)
    train_len = total_len - val_len

    gen = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, val_len], generator=gen,
    )
    logger.info(f"Data: {total_len:,} samples ({train_len:,} train, {val_len:,} val)")

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ── Model & Optimizer ─────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    logger.info(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = GradScaler()

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=train_cfg.num_epochs,
        pct_start=0.3,
        div_factor=25.0,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    last_ckpt = exp_dir / "checkpoint_last.pt"

    if args.resume and last_ckpt.exists():
        logger.info(f"Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        logger.info(f"Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}, "
                     f"patience={patience_counter}/{train_cfg.patience}")
    elif args.resume:
        logger.warning(f"--resume specified but {last_ckpt} not found. Training from scratch.")

    # ── Training Loop ─────────────────────────────────────────────────────
    sig = SignalHandler()
    epoch_times = []

    for epoch in range(start_epoch, train_cfg.num_epochs):
        epoch_start = time.time()

        # Train
        t = train_epoch(model, train_loader, optimizer, scheduler,
                        scaler, device, epoch + 1, train_cfg, sig)

        # Validate
        v = validate(model, val_loader, device, train_cfg)

        # Timing & ETA
        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = int(avg_epoch * (train_cfg.num_epochs - epoch - 1))

        # Log epoch summary
        is_best = v['loss'] < best_loss
        marker = " << new best" if is_best else ""
        logger.info(
            f"Ep {epoch+1}/{train_cfg.num_epochs} "
            f"[{timedelta(seconds=int(elapsed))}] "
            f"TL: {t['loss']:.4f} (P:{t['p_loss']:.4f} V:{t['v_loss']:.4f}) "
            f"TA: {t['acc']:.1f}% | "
            f"VL: {v['loss']:.4f} VA: {v['acc']:.1f}%"
            f"{marker} | "
            f"ETA: {timedelta(seconds=remaining)}"
        )

        # Early stopping logic
        if is_best:
            best_loss = v['loss']
            patience_counter = 0
        else:
            patience_counter += 1

        # Always save last checkpoint
        save_checkpoint(last_ckpt, model, optimizer, scheduler,
                        scaler, epoch, best_loss)

        # Save best model
        if is_best:
            save_checkpoint(exp_dir / "best_model.pt", model, optimizer,
                            scheduler, scaler, epoch, best_loss)

        # Check early stopping
        if train_cfg.patience > 0 and patience_counter >= train_cfg.patience:
            logger.info(
                f"Early stopping triggered: val_loss did not improve "
                f"for {train_cfg.patience} consecutive epochs. "
                f"Best loss: {best_loss:.4f}"
            )
            break

        # Check SIGINT
        if sig.interrupted:
            logger.info("Training interrupted after epoch. Exiting.")
            break

    logger.info(f"Training complete. Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
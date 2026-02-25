#!/usr/bin/env python3
"""
Production-Grade Training Loop for ChessNet-3070.

Features:
- Robust signal handling for graceful shutdown.
- Checkpoint auto-resume.
- Mixed Precision (FP16) training.
- Memory-mapped data loading for handling datasets larger than RAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Inject project root for imports (works from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig, TrainConfig
from src.model import ChessNet, build_model, count_params

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


class SignalHandler:
    """Handles SIGINT (Ctrl+C) to allow for graceful shutdown."""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        if self.interrupted:
            logger.warning("Force quitting...")
            sys.exit(1)
        
        self.interrupted = True
        logger.warning("\nReceived SIGINT. Finishing current epoch and saving checkpoint...")
        logger.warning("Press Ctrl+C again to force quit.")


class MemmapDataset(Dataset):
    """
    High-performance dataset using memory-mapped Numpy arrays.
    Allows instant access to massive datasets without loading them into RAM.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.meta_path = data_dir / "meta.json"
        
        # Determine dataset length
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.length = json.load(f)["n_positions"]
        else:
            # Fallback: check file size directly
            self.length = len(np.load(data_dir / "policies.npy", mmap_mode="r"))

        # Lazy loading handles
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
        # Copy ensures we get a writable tensor and avoid negative stride issues
        state = torch.from_numpy(self.states[idx].copy())
        policy = torch.tensor(self.policies[idx], dtype=torch.long)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value


def save_checkpoint(
    path: Path, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Any, 
    scaler: GradScaler, 
    epoch: int, 
    metrics: Dict[str, float]
):
    """Saves the training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    signal_handler: SignalHandler
) -> Dict[str, float]:
    
    model.train()
    total_loss = 0.0
    correct_policy = 0
    total_samples = 0
    
    policy_crit = nn.CrossEntropyLoss()
    value_crit = nn.MSELoss()

    pbar = tqdm(loader, desc=f"Train Ep {epoch}", dynamic_ncols=True, leave=False)

    for i, (states, policies, values) in enumerate(pbar):
        if signal_handler.interrupted:
            break

        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred_policies, pred_values = model(states)
            loss_p = policy_crit(pred_policies, policies)
            loss_v = value_crit(pred_values, values)
            loss = (loss_p * cfg.policy_loss_weight) + (loss_v * cfg.value_loss_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        batch_size = states.size(0)
        total_loss += loss.item() * batch_size
        correct_policy += (pred_policies.argmax(dim=1) == policies).sum().item()
        total_samples += batch_size

        # Update Progress Bar
        if i % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_policy / total_samples
    
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    cfg: TrainConfig
) -> Dict[str, float]:
    
    model.eval()
    total_loss = 0.0
    correct_policy = 0
    total_samples = 0
    
    policy_crit = nn.CrossEntropyLoss()
    value_crit = nn.MSELoss()

    for states, policies, values in tqdm(loader, desc="Validating", leave=False):
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        with autocast("cuda"):
            pred_policies, pred_values = model(states)
            loss_p = policy_crit(pred_policies, policies)
            loss_v = value_crit(pred_values, values)
            loss = (loss_p * cfg.policy_loss_weight) + (loss_v * cfg.value_loss_weight)

        batch_size = states.size(0)
        total_loss += loss.item() * batch_size
        correct_policy += (pred_policies.argmax(dim=1) == policies).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_policy / total_samples
    
    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="ChessNet Training")
    parser.add_argument("--data", type=Path, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    
    # Overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    args = parser.parse_args()

    # 1. Configuration
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    # Apply CLI overrides
    if args.epochs: train_cfg.num_epochs = args.epochs
    if args.batch_size: train_cfg.batch_size = args.batch_size
    if args.lr: train_cfg.learning_rate = args.lr

    exp_dir = train_cfg.checkpoint_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save active config
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model": asdict(model_cfg), "train": asdict(train_cfg)}, f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting experiment '{args.name}' on {device}")

    # 2. Data Loading
    if not (args.data / "states.npy").exists():
        logger.error(f"Data not found in {args.data}. Run convert_to_memmap.py first.")
        sys.exit(1)

    dataset = MemmapDataset(args.data)
    total_len = len(dataset)
    val_len = int(total_len * train_cfg.val_split)
    train_len = total_len - val_len
    
    # Deterministic split
    gen = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len], generator=gen)

    logger.info(f"Data: {total_len:,} samples ({train_len:,} train, {val_len:,} val)")

    train_loader = DataLoader(
        train_set, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        num_workers=train_cfg.dataloader_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=train_cfg.batch_size, 
        shuffle=False, 
        num_workers=train_cfg.dataloader_workers,
        pin_memory=True
    )

    # 3. Model & Optimization
    model = build_model(model_cfg).to(device)
    logger.info(f"Model Parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay
    )
    scaler = GradScaler()
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=train_cfg.num_epochs,
        pct_start=0.3,
        div_factor=25.0
    )

    # 4. Resume Logic
    start_epoch = 0
    best_loss = float('inf')
    last_ckpt = exp_dir / "last.pt"
    
    if args.resume and last_ckpt.exists():
        logger.info(f"Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['metrics'].get('best_loss', float('inf'))

    # 5. Training Loop
    signal_handler = SignalHandler()
    
    for epoch in range(start_epoch, train_cfg.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            device, epoch, train_cfg, signal_handler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, train_cfg)
        
        duration = time.time() - epoch_start
        
        # Logging
        is_best = val_metrics['loss'] < best_loss
        if is_best:
            best_loss = val_metrics['loss']
            
        logger.info(
            f"Ep {epoch+1}/{train_cfg.num_epochs} [{timedelta(seconds=int(duration))}] "
            f"TL: {train_metrics['loss']:.4f} TA: {train_metrics['accuracy']:.1f}% | "
            f"VL: {val_metrics['loss']:.4f} VA: {val_metrics['accuracy']:.1f}% "
            f"{'★' if is_best else ''}"
        )

        # Checkpointing
        metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_loss': best_loss
        }
        
        # Always save last
        save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler, epoch, metrics)
        
        # Save best
        if is_best and train_cfg.save_best:
            save_checkpoint(exp_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, metrics)

        # Early exit on SIGINT
        if signal_handler.interrupted:
            logger.info("Training interrupted. Exiting.")
            sys.exit(0)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import json
import logging
import math
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Inject project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.model import build_model, count_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class MemmapDataset(Dataset):
    """
    High-performance dataset using memory-mapped Numpy arrays.
    Allows instant access to 24M+ positions without loading RAM.
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
        # Copy ensures we get a writable tensor, avoiding negative stride issues
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


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logger.info(f"Checkpoint saved: {filename}")


def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    
    loss_meter = 0.0
    policy_loss_meter = 0.0
    value_loss_meter = 0.0
    correct = 0
    total = 0
    
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()

    pbar = tqdm(loader, desc=f"Train Ep {epoch}", dynamic_ncols=True)
    
    for states, policies, values in pbar:
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred_policies, pred_values = model(states)
            
            loss_p = p_criterion(pred_policies, policies)
            loss_v = v_criterion(pred_values, values)
            loss = loss_p + loss_v

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        bs = states.size(0)
        total += bs
        correct += (pred_policies.argmax(dim=1) == policies).sum().item()
        
        loss_val = loss.item()
        loss_meter += loss_val
        policy_loss_meter += loss_p.item()
        value_loss_meter += loss_v.item()

        # Update progress bar
        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    metrics = {
        "loss": loss_meter / len(loader),
        "acc": 100 * correct / total,
        "p_loss": policy_loss_meter / len(loader),
        "v_loss": value_loss_meter / len(loader)
    }
    return metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    p_criterion = nn.CrossEntropyLoss()
    v_criterion = nn.MSELoss()

    for states, policies, values in tqdm(loader, desc="Validating", leave=False):
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        with autocast("cuda"):
            pred_policies, pred_values = model(states)
            loss = p_criterion(pred_policies, policies) + v_criterion(pred_values, values)

        total_loss += loss.item()
        total += states.size(0)
        correct += (pred_policies.argmax(dim=1) == policies).sum().item()

    return {
        "loss": total_loss / len(loader),
        "acc": 100 * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description="ChessNet Training Pipeline")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory (memmap) or .h5 file")
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-lr", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Setup directories
    checkpoint_dir = Path("checkpoints") / args.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device} | Experiment: {args.name}")

    # Initialize Dataset
    if args.data.is_dir() and (args.data / "states.npy").exists():
        logger.info(f"Loading MemmapDataset from {args.data}")
        dataset = MemmapDataset(args.data)
    elif args.data.suffix in ['.h5', '.hdf5']:
        logger.info(f"Loading HDF5Dataset from {args.data}")
        dataset = HDF5Dataset(args.data)
    else:
        logger.error("Invalid data path. Ensure convert_to_memmap.py was run.")
        sys.exit(1)

    # Split Train/Val (90/10)
    total_len = len(dataset)
    val_len = int(0.10 * total_len)
    train_len = total_len - val_len
    
    # Deterministic split
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0)
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )

    logger.info(f"Train: {train_len:,} | Val: {val_len:,}")

    # Model Setup
    config = ModelConfig()
    model = build_model(config).to(device)
    logger.info(f"Model Parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-4)
    scaler = GradScaler()
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr, 
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0
    )

    # State tracking
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume Logic
    last_ckpt_path = checkpoint_dir / "checkpoint_last.pt"
    if args.resume and last_ckpt_path.exists():
        logger.info(f"Resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

    # Training Loop
    try:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            
            # Train
            t_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch + 1)
            
            # Validate
            v_metrics = validate(model, val_loader, device)
            
            duration = str(timedelta(seconds=int(time.time() - start_time)))
            
            logger.info(
                f"Ep {epoch+1}/{args.epochs} [{duration}] "
                f"Train Loss: {t_metrics['loss']:.4f} (Acc: {t_metrics['acc']:.1f}%) | "
                f"Val Loss: {v_metrics['loss']:.4f} (Acc: {v_metrics['acc']:.1f}%)"
            )

            # Checkpointing
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_loss': best_loss
            }
            
            # Always save last
            save_checkpoint(state, last_ckpt_path)

            # Save best
            if v_metrics['loss'] < best_loss:
                best_loss = v_metrics['loss']
                state['best_loss'] = best_loss
                save_checkpoint(state, checkpoint_dir / "best_model.pt")
                logger.info(f"New best model! (Loss: {best_loss:.4f})")

    except KeyboardInterrupt:
        logger.warning("Training interrupted! Saving emergency checkpoint...")
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_loss': best_loss
        }
        save_checkpoint(state, checkpoint_dir / "checkpoint_interrupted.pt")
        sys.exit(0)


if __name__ == "__main__":
    main()
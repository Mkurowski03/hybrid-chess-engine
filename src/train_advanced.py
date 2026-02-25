#!/usr/bin/env python3
"""Production-Grade Training Script for ChessNet-3070.

Designed for long-running sessions (12-48h) with maximum resilience.

Usage
-----
    # Fresh training:
    python src/train_advanced.py --data data/ --name baseline_v2

    # Resume after interruption:
    python src/train_advanced.py --data data/ --name baseline_v2 --resume

Features
--------
- **OneCycleLR** scheduler (max_lr=0.01, epochs=30).
- **Early Stopping** (patience=5).
- **Robust Checkpointing**: best_model.pt, checkpoint_last.pt, checkpoint_interrupted.pt.
- **Auto-Resume**: --resume flag loads everything from checkpoint_last.pt.
- **Ctrl+C Safety**: Saves checkpoint_interrupted.pt on KeyboardInterrupt.
- **FP16 Mixed Precision** (torch.cuda.amp.GradScaler).
- **High-Throughput DataLoader** (num_workers=8, pin_memory=True).
- **Detailed Logging**: epoch progress, moving average loss, LR, ETA.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig, TrainConfig
from src.model import ChessNet, build_model, count_params


# ---------------------------------------------------------------------------
# Dataset (Memmap + HDF5 Fallback)
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    """Memory-mapped numpy dataset for instant random-access I/O."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._len = json.load(f)["n_positions"]
        else:
            arr = np.load(data_dir / "policies.npy", mmap_mode="r")
            self._len = arr.shape[0]
            del arr
        self._states = None
        self._policies = None
        self._values = None

    def _ensure_open(self) -> None:
        if self._states is None:
            self._states = np.load(self._data_dir / "states.npy", mmap_mode="r")
            self._policies = np.load(self._data_dir / "policies.npy", mmap_mode="r")
            self._values = np.load(self._data_dir / "values.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        self._ensure_open()
        state = torch.from_numpy(self._states[idx].copy())
        policy = torch.tensor(self._policies[idx], dtype=torch.long)
        value = torch.tensor(self._values[idx], dtype=torch.float32)
        return state, policy, value


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_val_loss: float,
    patience_counter: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }, path)


def _format_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Training / Validation Epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: ChessNet,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    global_step: int,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """Run one epoch (train or val). Returns metrics dict."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    phase = "Train" if is_train else "  Val"

    total_loss = 0.0
    total_ploss = 0.0
    total_vloss = 0.0
    correct = 0
    total_samples = 0
    n_batches = 0
    # Moving average for display
    ma_loss = 0.0
    ma_alpha = 0.05  # Exponential moving average weight

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    pbar = tqdm(
        loader,
        desc=f"  {phase} {epoch + 1}/{num_epochs}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for states, policies, values in pbar:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                policy_logits, value_pred = model(states)
                p_loss = policy_criterion(policy_logits, policies)
                v_loss = value_criterion(value_pred, values)
                loss = policy_weight * p_loss + value_weight * v_loss

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            total_ploss += p_loss.item()
            total_vloss += v_loss.item()
            n_batches += 1
            total_samples += states.size(0)
            correct += (policy_logits.argmax(dim=1) == policies).sum().item()

            if is_train:
                global_step += 1

            # Exponential moving average for display
            if n_batches == 1:
                ma_loss = batch_loss
            else:
                ma_loss = (1 - ma_alpha) * ma_loss + ma_alpha * batch_loss

            acc = correct / total_samples * 100
            postfix = f"ma_loss={ma_loss:.4f} acc={acc:.1f}%"
            if is_train and scheduler is not None:
                postfix += f" lr={scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix_str(postfix, refresh=False)

    pbar.close()

    avg_loss = total_loss / max(n_batches, 1)
    avg_ploss = total_ploss / max(n_batches, 1)
    avg_vloss = total_vloss / max(n_batches, 1)
    accuracy = correct / max(total_samples, 1) * 100

    return {
        "loss": avg_loss,
        "policy_loss": avg_ploss,
        "value_loss": avg_vloss,
        "accuracy": accuracy,
        "global_step": global_step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ChessNet-3070 Advanced Training")
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory with states.npy/policies.npy/values.npy (or .h5)")
    parser.add_argument("--name", type=str, default="advanced",
                        help="Experiment name (checkpoints → checkpoints/<name>/)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint_last.pt")
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--max-lr", type=float, default=0.01, help="OneCycleLR max_lr (default: 0.01)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")
    # Model overrides
    parser.add_argument("--filters", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    args = parser.parse_args()

    # ── Config ──
    model_cfg = ModelConfig()
    if args.filters is not None:
        model_cfg.num_filters = args.filters
    if args.blocks is not None:
        model_cfg.num_residual_blocks = args.blocks

    exp_dir = Path("checkpoints") / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ──
    print()
    print("=" * 65)
    print(f"  ♟  ChessNet-3070 Advanced Trainer — [{args.name}]")
    print("=" * 65)
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM            : {vram:.1f} GB")

    # ── Model ──
    model = build_model(model_cfg).to(device)
    n_params = count_params(model)
    print(f"  Architecture    : {model_cfg.num_residual_blocks} blocks × {model_cfg.num_filters} filters")
    print(f"  Parameters      : {n_params:,}")

    # ── Data ──
    if (args.data / "states.npy").exists():
        dataset = MemmapDataset(args.data)
        print(f"  Data format     : memmap (fast)")
    else:
        h5_candidates = list(args.data.glob("*.h5")) + list(args.data.glob("*.hdf5"))
        if h5_candidates:
            import h5py

            class HDF5Dataset(Dataset):
                def __init__(self, path: Path) -> None:
                    self._path = path
                    with h5py.File(path, "r") as f:
                        self._len = f["states"].shape[0]
                    self._file = None

                def __len__(self) -> int:
                    return self._len

                def __getitem__(self, idx):
                    if self._file is None:
                        self._file = h5py.File(self._path, "r")
                    state = torch.from_numpy(self._file["states"][idx].astype(np.float32))
                    policy = torch.tensor(self._file["policies"][idx], dtype=torch.long)
                    value = torch.tensor(self._file["values"][idx], dtype=torch.float32)
                    return state, policy, value

            dataset = HDF5Dataset(h5_candidates[0])
            print(f"  Data format     : HDF5 (slow — run convert_to_memmap.py!)")
        else:
            print(f"\n❌ No data found in {args.data}!")
            return

    n_total = len(dataset)
    val_split = 0.1
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_subset = Subset(dataset, indices[:n_train])
    val_subset = Subset(dataset, indices[n_train:])

    print(f"  Positions       : {n_total:,}")
    print(f"  Train / Val     : {n_train:,} / {n_val:,}")
    print(f"  Batch size      : {args.batch_size}")

    num_workers = args.workers
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps/epoch     : {steps_per_epoch}")
    print(f"  Total steps     : {total_steps:,}")

    # ── Optimizer / Scheduler / Scaler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr / 25,  # div_factor=25 is OneCycle default
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # ── Auto-Resume ──
    last_ckpt = exp_dir / "checkpoint_last.pt"
    if args.resume or last_ckpt.exists():
        resume_path = last_ckpt if last_ckpt.exists() else None
        if resume_path is not None and resume_path.exists():
            print(f"  Resuming from   : {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"  Resumed epoch   : {start_epoch} (best_val_loss={best_val_loss:.4f})")
        else:
            print(f"  Resume          : no checkpoint found, starting fresh")
    else:
        print(f"  Starting fresh")

    print(f"  Epochs          : {start_epoch} → {args.epochs}")
    print(f"  Max LR          : {args.max_lr}")
    print(f"  Early stopping  : patience={args.patience}")
    print(f"  Workers         : {num_workers}")

    # Save experiment config
    exp_config = {
        "model": {k: str(v) if isinstance(v, Path) else v for k, v in vars(model_cfg).items()},
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "max_lr": args.max_lr, "patience": args.patience,
            "workers": num_workers,
        },
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("=" * 65)
    print("  Ctrl+C → saves checkpoint_interrupted.pt")
    print("  --resume → loads checkpoint_last.pt")
    print("=" * 65)
    print()

    # ── Training Loop (KeyboardInterrupt-safe) ──
    training_start = time.perf_counter()
    epoch_times = []

    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.perf_counter()

            # ── Train ──
            train_metrics = _run_epoch(
                model, train_loader, device,
                epoch, args.epochs, global_step,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            )
            global_step = train_metrics["global_step"]

            # ── Validate ──
            val_metrics = _run_epoch(
                model, val_loader, device,
                epoch, args.epochs, global_step,
            )

            epoch_time = time.perf_counter() - t0
            epoch_times.append(epoch_time)
            total_elapsed = time.perf_counter() - training_start
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_remaining = args.epochs - epoch - 1
            eta_total = avg_epoch_time * epochs_remaining

            # ── Epoch Summary ──
            improved = val_metrics["loss"] < best_val_loss
            marker = " ★ NEW BEST" if improved else ""
            now = datetime.now().strftime('%H:%M:%S')

            print(f"\n{'─' * 65}")
            print(f"  [{now}] Epoch {epoch + 1}/{args.epochs}  │  "
                  f"T.loss={train_metrics['loss']:.4f}  V.loss={val_metrics['loss']:.4f}{marker}")
            print(f"  T.acc={train_metrics['accuracy']:.1f}%  V.acc={val_metrics['accuracy']:.1f}%  │  "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")
            print(f"  Epoch: {_format_time(epoch_time)}  │  "
                  f"Elapsed: {_format_time(total_elapsed)}  │  "
                  f"ETA: {_format_time(eta_total)}")
            print(f"{'─' * 65}")

            # ── Always save checkpoint_last.pt ──
            _save_checkpoint(
                exp_dir / "checkpoint_last.pt",
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_val_loss if not improved else val_metrics["loss"],
                patience_counter if not improved else 0,
            )
            print(f"  💾 checkpoint_last.pt saved")

            # ── Best model ──
            if improved:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                _save_checkpoint(
                    exp_dir / "best_model.pt",
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_val_loss, 0,
                )
                print(f"  🏆 best_model.pt saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{args.patience})")

            # ── Early stopping ──
            if patience_counter >= args.patience:
                print(f"\n⛔ Early stopping! Val loss didn't improve for {args.patience} epochs.")
                print(f"   Best model: {exp_dir / 'best_model.pt'} (val_loss={best_val_loss:.4f})")
                break

        else:
            total_time = time.perf_counter() - training_start
            print(f"\n✅ Training complete! Total time: {_format_time(total_time)}")
            print(f"   Best val_loss: {best_val_loss:.4f}")
            print(f"   Best model:    {exp_dir / 'best_model.pt'}")

    except KeyboardInterrupt:
        print(f"\n\n⏸  KeyboardInterrupt received! Saving emergency checkpoint...")
        _save_checkpoint(
            exp_dir / "checkpoint_interrupted.pt",
            epoch, global_step, model, optimizer, scheduler, scaler,
            best_val_loss, patience_counter,
        )
        print(f"  💾 Saved → {exp_dir / 'checkpoint_interrupted.pt'}")
        print(f"\n  To resume: python src/train_advanced.py --data {args.data} --name {args.name} --resume\n")


if __name__ == "__main__":
    main()

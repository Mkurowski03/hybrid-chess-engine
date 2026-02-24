#!/usr/bin/env python3
"""Training loop for ChessNet-3070.

Usage
-----
    # Convert HDF5 to memmap first (one-time, ~2 min):
    python scripts/convert_to_memmap.py data/train.h5 data/

    # Train:
    python train.py --data data/ --name baseline
    python train.py --data data/ --name small --filters 64 --blocks 6 --epochs 15
    python train.py --data data/ --name baseline --resume   # resume from checkpoint

Features
--------
- **Numpy memmap** dataset — instant random-access I/O (no gzip overhead).
- **Named experiments** — each ``--name`` gets its own checkpoint dir.
- **CLI model overrides** — ``--filters``, ``--blocks`` to test architectures.
- **Train/val split** (90/10) with val loss tracking per epoch.
- **Early stopping** — stops if val loss doesn't improve for N epochs.
- **CosineAnnealingWarmRestarts** scheduler (flexible for extended training).
- **Mixed-precision** (torch.amp) + **AdamW**.
- **tqdm progress bars** with ETA per epoch and overall.
- **Graceful pause/resume** — Ctrl+C saves checkpoint, ``--resume`` restores.
- **WandB logging** (optional).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from src.model import ChessNet, build_model, count_params

# ---------------------------------------------------------------------------
# Graceful interruption
# ---------------------------------------------------------------------------

_INTERRUPTED = False


def _signal_handler(signum: int, frame: object) -> None:
    global _INTERRUPTED
    if _INTERRUPTED:
        print("\n⚠  Force quit.")
        sys.exit(1)
    _INTERRUPTED = True
    print("\n\n⏸  Ctrl+C — finishing current batch, saving checkpoint …")
    print("   (press Ctrl+C again to force quit)\n")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MemmapDataset(Dataset):
    """Memory-mapped numpy dataset — instant random-access I/O.

    Opens memmap lazily on first access so DataLoader workers don't
    need to pickle the 100+ GB array references during spawn.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        # Read length from metadata to avoid opening large files
        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                self._len = json.load(f)["n_positions"]
        else:
            # Fallback: peek at policies (small file)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_open()
        state = torch.from_numpy(self._states[idx].copy())
        policy = torch.tensor(self._policies[idx], dtype=torch.long)
        value = torch.tensor(self._values[idx], dtype=torch.float32)
        return state, policy, value


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------


def _run_epoch(
    model: ChessNet,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    num_epochs: int,
    global_step: int,
    # Training-only args (None for validation)
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    wandb_run: object | None = None,
) -> dict:
    """Run one epoch (train or val). Returns metrics dict."""
    global _INTERRUPTED
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    phase = "Train" if is_train else "Val"

    total_loss = 0.0
    total_ploss = 0.0
    total_vloss = 0.0
    correct = 0
    total_samples = 0
    n_batches = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    pbar = tqdm(
        loader,
        desc=f"  {phase} {epoch + 1}/{num_epochs}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for states, policies, values in pbar:
            if _INTERRUPTED:
                pbar.close()
                break

            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                policy_logits, value_pred = model(states)
                p_loss = policy_criterion(policy_logits, policies)
                v_loss = value_criterion(value_pred, values)
                loss = cfg.policy_loss_weight * p_loss + cfg.value_loss_weight * v_loss

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            total_ploss += p_loss.item()
            total_vloss += v_loss.item()
            n_batches += 1
            total_samples += states.size(0)
            correct += (policy_logits.argmax(dim=1) == policies).sum().item()

            if is_train:
                global_step += 1

            acc = correct / total_samples * 100
            postfix = f"loss={loss.item():.3f} acc={acc:.1f}%"
            if is_train and scheduler is not None:
                postfix += f" lr={scheduler.get_last_lr()[0]:.1e}"
            pbar.set_postfix_str(postfix, refresh=False)

            # WandB
            if is_train and wandb_run and global_step % cfg.log_every_n_steps == 0:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/policy_loss": p_loss.item(),
                    "train/value_loss": v_loss.item(),
                    "train/accuracy": acc,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

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
        "interrupted": _INTERRUPTED,
    }


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


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("chessnet_epoch*.pt"))
    return checkpoints[-1] if checkpoints else None


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(description="Train ChessNet-3070")
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory with states.npy/policies.npy/values.npy")
    parser.add_argument("--name", type=str, default="default",
                        help="Experiment name (checkpoints go to checkpoints/<name>/)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume training (auto-find or pass checkpoint path)")
    parser.add_argument("--no-wandb", action="store_true")
    # Model overrides
    parser.add_argument("--filters", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    model_cfg = ModelConfig()

    # Apply CLI overrides
    if args.filters is not None:
        model_cfg.num_filters = args.filters
    if args.blocks is not None:
        model_cfg.num_residual_blocks = args.blocks
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
        cfg.max_lr = args.lr
    if args.patience is not None:
        cfg.patience = args.patience
    if args.no_wandb:
        cfg.use_wandb = False

    # Experiment directory
    exp_dir = cfg.checkpoint_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ──
    print()
    print("=" * 60)
    print(f"  ♟  ChessNet-3070 Training — [{args.name}]")
    print("=" * 60)
    print(f"  Device       : {device}")

    # ── Model ──
    model = build_model(model_cfg).to(device)
    n_params = count_params(model)
    print(f"  Architecture : {model_cfg.num_residual_blocks} blocks × {model_cfg.num_filters} filters")
    print(f"  Parameters   : {n_params:,}")

    if cfg.compile_model and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]
        print("  torch.compile: enabled")

    # ── Data ──
    # Check for memmap files, fall back to HDF5
    if (args.data / "states.npy").exists():
        dataset = MemmapDataset(args.data)
        print(f"  Data format  : memmap (fast)")
    else:
        # Fallback: check for HDF5
        h5_candidates = list(args.data.glob("*.h5")) + list(args.data.glob("*.hdf5"))
        if h5_candidates:
            # Import HDF5 fallback
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
            print(f"  Data format  : HDF5 (slow — run convert_to_memmap.py!)")
        else:
            print(f"\n❌ No data found in {args.data}!")
            print(f"   Expected: states.npy or *.h5 file")
            return

    n_total = len(dataset)
    print(f"  Positions    : {n_total:,}")

    # ── Train/Val split ──
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"  Train split  : {n_train:,} ({100 - cfg.val_split * 100:.0f}%)")
    print(f"  Val split    : {n_val:,} ({cfg.val_split * 100:.0f}%)")
    print(f"  Batch size   : {cfg.batch_size}")

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.dataloader_workers, pin_memory=True,
        drop_last=True, persistent_workers=cfg.dataloader_workers > 0,
        prefetch_factor=4 if cfg.dataloader_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.dataloader_workers, pin_memory=True,
        drop_last=False, persistent_workers=cfg.dataloader_workers > 0,
        prefetch_factor=4 if cfg.dataloader_workers > 0 else None,
    )

    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    print(f"  Steps/epoch  : {steps_per_epoch}")

    # ── Optimizer / Scheduler / Scaler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=2, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # ── Resume ──
    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = _find_latest_checkpoint(exp_dir)
            if resume_path is None:
                print("  Resume       : no checkpoint found, starting fresh")
            else:
                print(f"  Resume       : {resume_path.name}")
        else:
            resume_path = Path(args.resume)
            print(f"  Resume       : {resume_path.name}")

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Resumed at   : epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    print(f"  Epochs       : {start_epoch} → {cfg.num_epochs}")
    if cfg.patience > 0:
        print(f"  Early stop   : patience={cfg.patience}")

    # Save experiment config
    exp_config = {"model": vars(model_cfg), "train": vars(cfg)}
    # Convert Path objects for JSON serialization
    for section in exp_config.values():
        for k, v in section.items():
            if isinstance(v, Path):
                section[k] = str(v)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("=" * 60)
    print("  Ctrl+C to pause  •  --resume to continue")
    print("=" * 60)
    print()

    # ── WandB ──
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=args.name,
                config=exp_config,
                resume="allow",
            )
        except Exception as exc:
            print(f"⚠  WandB init failed ({exc}), continuing without logging.")

    # ── Training Loop ──
    training_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.perf_counter()

        # ── Train ──
        train_metrics = _run_epoch(
            model, train_loader, device, cfg,
            epoch, cfg.num_epochs, global_step,
            optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, wandb_run=wandb_run,
        )
        global_step = train_metrics["global_step"]

        if train_metrics["interrupted"]:
            _save_checkpoint(
                exp_dir / f"chessnet_epoch{epoch}.pt",
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_val_loss, patience_counter,
            )
            print(f"\n  💾 Checkpoint saved → {exp_dir / f'chessnet_epoch{epoch}.pt'}")
            print(f"\n⏸  Training paused at epoch {epoch + 1}. Resume with:")
            print(f"   python train.py --data {args.data} --name {args.name} --resume\n")
            break

        # ── Validate ──
        val_metrics = _run_epoch(
            model, val_loader, device, cfg,
            epoch, cfg.num_epochs, global_step,
        )

        if val_metrics["interrupted"]:
            _save_checkpoint(
                exp_dir / f"chessnet_epoch{epoch}.pt",
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_val_loss, patience_counter,
            )
            print(f"\n  💾 Checkpoint saved → {exp_dir / f'chessnet_epoch{epoch}.pt'}")
            print(f"\n⏸  Training paused. Resume with:")
            print(f"   python train.py --data {args.data} --name {args.name} --resume\n")
            break

        elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - training_start
        epochs_done = epoch - start_epoch + 1
        epochs_remaining = cfg.num_epochs - epoch - 1
        eta = (total_elapsed / epochs_done) * epochs_remaining

        # ── Epoch Summary ──
        improved = val_metrics["loss"] < best_val_loss
        marker = " ★" if improved else ""

        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n{'─' * 70}")
        print(f"  [{now}] Epoch {epoch + 1}/{cfg.num_epochs}  │  "
              f"train_loss={train_metrics['loss']:.4f}  val_loss={val_metrics['loss']:.4f}{marker}  │  "
              f"train_acc={train_metrics['accuracy']:.1f}%  val_acc={val_metrics['accuracy']:.1f}%")
        print(f"  {_format_time(elapsed)}/epoch  │  ETA: {_format_time(eta)}")
        print(f"{'─' * 70}")

        # WandB epoch metrics
        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": train_metrics["loss"],
                "epoch/val_loss": val_metrics["loss"],
                "epoch/train_accuracy": train_metrics["accuracy"],
                "epoch/val_accuracy": val_metrics["accuracy"],
            }, step=global_step)

        # ── Checkpoint ──
        _save_checkpoint(
            exp_dir / f"chessnet_epoch{epoch}.pt",
            epoch, global_step, model, optimizer, scheduler, scaler,
            best_val_loss, patience_counter,
        )
        print(f"  💾 Checkpoint → {exp_dir / f'chessnet_epoch{epoch}.pt'}")

        # ── Best model ──
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            if cfg.save_best:
                _save_checkpoint(
                    exp_dir / "best.pt",
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_val_loss, patience_counter,
                )
                print(f"  🏆 New best val_loss={best_val_loss:.4f} → {exp_dir / 'best.pt'}")
        else:
            patience_counter += 1
            if cfg.patience > 0:
                print(f"  ⏳ No improvement ({patience_counter}/{cfg.patience})")

        # ── Early stopping ──
        if cfg.patience > 0 and patience_counter >= cfg.patience:
            print(f"\n⛔ Early stopping! Val loss didn't improve for {cfg.patience} epochs.")
            print(f"   Best model: {exp_dir / 'best.pt'}")
            break

    else:
        total_time = time.perf_counter() - training_start
        print(f"\n✅ Training complete! Total time: {_format_time(total_time)}")
        print(f"   Best val_loss: {best_val_loss:.4f}")
        print(f"   Best model:    {exp_dir / 'best.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

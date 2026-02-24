#!/usr/bin/env python3
"""ETL: Lumbras Gigabase .7z archives  →  HDF5 dataset (states, policies, values).

Usage
-----
    python scripts/process_data.py *.7z -o data/train.h5

    # Resume after Ctrl+C (skips already-processed archives):
    python scripts/process_data.py *.7z -o data/train.h5

Key optimisation
----------------
Workers return **lightweight tuples** (FEN string, move-UCI, result) instead of
numpy arrays. The main process does the board encoding in bulk. This avoids
blowing up Windows IPC pipes with large numpy payloads.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import h5py
import numpy as np
import py7zr
from tqdm import tqdm

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataConfig
from src.board_encoder import encode_board

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
    print("\n\n⏸  Ctrl+C — saving progress after current batch …")
    print("   (press Ctrl+C again to force quit)\n")


# ---------------------------------------------------------------------------
# Global config for worker processes (set via initializer)
# ---------------------------------------------------------------------------

_WORKER_CFG: DataConfig | None = None


def _init_worker(cfg: DataConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


# ---------------------------------------------------------------------------
# Per-game processing (runs in worker processes)
# Returns LIGHTWEIGHT data: list of (fen, move_uci, value) strings
# ---------------------------------------------------------------------------


def _process_single_game(pgn_text: str) -> list[tuple[str, str, float]]:
    """Parse one PGN, return lightweight (FEN, move_uci, value) tuples.

    Board encoding happens in the main process to avoid sending
    large numpy arrays through the multiprocessing pipe.
    """
    try:
        return _process_single_game_inner(pgn_text)
    except Exception:
        return []


def _process_single_game_inner(pgn_text: str) -> list[tuple[str, str, float]]:
    cfg = _WORKER_CFG
    assert cfg is not None

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []

    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return []
    if white_elo < cfg.min_elo or black_elo < cfg.min_elo:
        return []

    result_str = headers.get("Result", "*")
    if result_str not in ("1-0", "0-1", "1/2-1/2"):
        return []

    # Walk through the game
    board = game.board()
    candidates: list[tuple[str, str, int]] = []  # (fen, move_uci, fullmove)
    node = game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        candidates.append((board.fen(), move.uci(), board.fullmove_number))
        board.push(move)
        node = next_node

    if len(candidates) < cfg.min_ply_count:
        return []

    # Determine value
    if result_str == "1-0":
        white_val = 1.0
    elif result_str == "0-1":
        white_val = -1.0
    else:
        white_val = 0.0

    # Sample positions
    random.shuffle(candidates)
    sampled: list[tuple[str, str, float]] = []
    for fen, move_uci, fullmove in candidates:
        if len(sampled) >= cfg.positions_per_game:
            break
        if fullmove <= cfg.opening_cutoff_move and random.random() > cfg.opening_keep_prob:
            continue
        # Value from perspective of side to move
        is_white = " w " in fen
        value = white_val if is_white else -white_val
        sampled.append((fen, move_uci, value))

    return sampled


# ---------------------------------------------------------------------------
# Main-process encoding: FEN+UCI → numpy arrays
# ---------------------------------------------------------------------------


def _move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq


def _encode_batch(
    samples: list[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a batch of (FEN, move_uci, value) in the main process."""
    states = []
    policies = []
    values = []
    for fen, move_uci, value in samples:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        states.append(encode_board(board))
        policies.append(_move_to_policy_index(move, board.turn))
        values.append(value)
    return (
        np.stack(states, axis=0),
        np.array(policies, dtype=np.int64),
        np.array(values, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 7z  →  PGN game strings
# ---------------------------------------------------------------------------


def _iter_games_from_pgn_text(pgn_text: str) -> list[str]:
    games: list[str] = []
    buf: list[str] = []
    in_moves = False
    for line in pgn_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            if in_moves and buf:
                games.append("".join(buf))
                buf = []
                in_moves = False
            buf.append(line)
        elif stripped == "":
            buf.append(line)
        else:
            in_moves = True
            buf.append(line)
    if buf:
        games.append("".join(buf))
    return games


def _extract_games_from_7z(archive_path: Path) -> list[str]:
    all_games: list[str] = []
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive.extractall(path=tmpdir)
            for root, _dirs, files in os.walk(tmpdir):
                for fname in files:
                    if fname.lower().endswith(".pgn"):
                        pgn_path = Path(root) / fname
                        text = pgn_path.read_text(encoding="utf-8", errors="replace")
                        all_games.extend(_iter_games_from_pgn_text(text))
    return all_games


def _collect_7z_files(inputs: list[Path]) -> list[Path]:
    archives: list[Path] = []
    for p in inputs:
        if p.is_dir():
            archives.extend(sorted(p.glob("*.7z")))
        elif p.suffix == ".7z" and p.is_file():
            archives.append(p)
    return archives


# ---------------------------------------------------------------------------
# Resume tracking
# ---------------------------------------------------------------------------

def _progress_file(output: Path) -> Path:
    return output.with_suffix(".progress.json")


def _load_progress(output: Path) -> dict:
    pf = _progress_file(output)
    if pf.exists():
        with open(pf, "r") as f:
            return json.load(f)
    return {"completed_archives": [], "total_positions": 0}


def _save_progress(output: Path, progress: dict) -> None:
    pf = _progress_file(output)
    with open(pf, "w") as f:
        json.dump(progress, f, indent=2)


def _append_to_hdf5(
    output: Path,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    n_new = states.shape[0]
    if output.exists():
        with h5py.File(output, "a") as hf:
            old_n = hf["states"].shape[0]
            new_n = old_n + n_new
            hf["states"].resize(new_n, axis=0)
            hf["policies"].resize(new_n, axis=0)
            hf["values"].resize(new_n, axis=0)
            hf["states"][old_n:new_n] = states
            hf["policies"][old_n:new_n] = policies
            hf["values"][old_n:new_n] = values
        return new_n
    else:
        chunk_sz = 4096
        with h5py.File(output, "w") as hf:
            hf.create_dataset(
                "states", data=states,
                maxshape=(None, 18, 8, 8),
                chunks=(min(chunk_sz, n_new), 18, 8, 8),
                compression="gzip", compression_opts=4,
            )
            hf.create_dataset(
                "policies", data=policies,
                maxshape=(None,),
                chunks=(min(chunk_sz, n_new),),
                compression="gzip", compression_opts=4,
            )
            hf.create_dataset(
                "values", data=values,
                maxshape=(None,),
                chunks=(min(chunk_sz, n_new),),
                compression="gzip", compression_opts=4,
            )
        return n_new


# ---------------------------------------------------------------------------
# Process one archive end-to-end (with chunked encoding/writing)
# ---------------------------------------------------------------------------

def _process_archive(
    archive: Path,
    cfg: DataConfig,
    output: Path,
    progress: dict,
) -> int:
    global _INTERRUPTED

    # ── Extract ──
    tqdm.write(f"\n📦 Extracting {archive.name} …")
    t0 = time.perf_counter()
    game_strings = _extract_games_from_7z(archive)
    elapsed = time.perf_counter() - t0
    tqdm.write(f"   {len(game_strings):,} raw games (extracted in {elapsed:.1f}s)")

    if not game_strings or _INTERRUPTED:
        return 0

    # ── Parsing & Chunked Processing ──
    # We buffer samples and flush to HDF5 periodically to prevent OOM
    # on large archives (which can produce >10M positions).
    BATCH_SIZE = 50_000
    pending_samples: list[tuple[str, str, float]] = []
    total_arch_positions = 0
    games_passed = 0

    chunksize = max(1, min(500, len(game_strings) // (cfg.num_workers * 20)))

    # Phase 1: Parallel Parse -> Buffered Encode & Write
    with Pool(
        processes=cfg.num_workers,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(_process_single_game, game_strings, chunksize=chunksize),
            total=len(game_strings),
            desc=f"Processing {archive.stem}",
            unit="game",
            dynamic_ncols=True,
            miniters=1000,
        )

        for samples in pbar:
            if _INTERRUPTED:
                pool.terminate()
                pbar.close()
                break

            if samples:
                games_passed += 1
                pending_samples.extend(samples)

            # Flush buffer if full
            if len(pending_samples) >= BATCH_SIZE:
                _flush_batch(pending_samples, output)
                total_arch_positions += len(pending_samples)
                pending_samples = []  # clear buffer

            pbar.set_postfix_str(
                f"kept={games_passed:,}  saved={total_arch_positions:,}  buff={len(pending_samples):,}",
                refresh=False,
            )

        pbar.close()

    if _INTERRUPTED:
        return 0

    # Flush remaining
    if pending_samples:
        _flush_batch(pending_samples, output)
        total_arch_positions += len(pending_samples)

    # Done
    if total_arch_positions == 0:
        tqdm.write(f"   ⚠ No positions passed filters for {archive.name}.")
    else:
        tqdm.write(f"   ✓ {archive.name}: +{total_arch_positions:,} positions")

    # Only update progress.json AFTER full archive is successfully done.
    # Note: HDF5 has been appended to incrementally. If we crash now,
    # the JSON won't include this archive, and next run will TRUNCATE
    # the HDF5 back to the old length, effectively rolling back this archive.
    progress["completed_archives"].append(archive.name)
    progress["total_positions"] += total_arch_positions
    _save_progress(output, progress)

    return total_arch_positions


def _flush_batch(
    samples: list[tuple[str, str, float]],
    output: Path,
) -> int:
    """Encode and append a batch of samples."""
    if not samples:
        return 0
    states, policies, values = _encode_batch(samples)
    return _append_to_hdf5(output, states, policies, values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    global _INTERRUPTED

    parser = argparse.ArgumentParser(
        description="Convert Lumbras Gigabase .7z archives → HDF5 dataset",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more .7z files or directories containing .7z files",
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("data/train.h5"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore previous progress")
    args = parser.parse_args()

    cfg = DataConfig()
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.min_elo is not None:
        cfg.min_elo = args.min_elo
    if args.min_ply is not None:
        cfg.min_ply_count = args.min_ply

    # ── Banner ──
    print()
    print("=" * 60)
    print("  ♟  ChessNet-3070  ETL Pipeline (Memory Optimized)")
    print("=" * 60)
    print(f"  Workers    : {cfg.num_workers}")
    print(f"  Min Elo    : {cfg.min_elo}")
    print(f"  Min plies  : {cfg.min_ply_count}")
    print(f"  Output     : {args.output}")

    # ── Discover ──
    archives = _collect_7z_files(args.inputs)
    if not archives:
        print("\n[ETL] No .7z files found. Exiting.")
        return

    # ── Resume / Fresh ──
    if args.fresh:
        if args.output.exists():
            args.output.unlink()
        pf = _progress_file(args.output)
        if pf.exists():
            pf.unlink()

    progress = _load_progress(args.output)
    
    # TRUNCATE HDF5 to match the last successful progress checkpoint.
    # This rolls back any partial writes from a crashed run.
    expected_positions = progress["total_positions"]
    if args.output.exists():
        with h5py.File(args.output, "a") as hf:
            current_positions = hf["states"].shape[0]
            if current_positions > expected_positions:
                print(f"  ↺ Rolling back HDF5: {current_positions:,} → {expected_positions:,} positions")
                hf["states"].resize(expected_positions, axis=0)
                hf["policies"].resize(expected_positions, axis=0)
                hf["values"].resize(expected_positions, axis=0)
            elif current_positions < expected_positions:
                 # Should theoretically not happen unless user messed with files manually
                 print(f"  ⚠ Warning: HDF5 has fewer positions ({current_positions}) than progress.json ({expected_positions}). Trusting HDF5.")
                 progress["total_positions"] = current_positions
                 expected_positions = current_positions

    completed = set(progress["completed_archives"])
    remaining = [a for a in archives if a.name not in completed]

    if completed:
        print(f"  Resuming   : {len(completed)} archives done "
              f"({expected_positions:,} positions)")

    print(f"  Remaining  : {len(remaining)} archive(s)")
    print("=" * 60)
    print("  Ctrl+C to pause  •  re-run to resume")
    print("=" * 60)
    print()

    if not remaining:
        print("✅ All archives already processed!")
        print(f"   {progress['total_positions']:,} positions in {args.output}")
        print(f"\n   Next: python train.py --data {args.output}\n")
        return

    # ── Process ──
    for i, archive in enumerate(remaining, 1):
        if _INTERRUPTED:
            break
        print(f"\n{'━' * 60}")
        print(f"  Archive {i}/{len(remaining)}: {archive.name} "
              f"({archive.stat().st_size / 1e6:.1f} MB)")
        print(f"{'━' * 60}")
        _process_archive(archive, cfg, args.output, progress)
        if _INTERRUPTED:
            break

    # ── Summary ──
    progress = _load_progress(args.output)
    done = len(progress["completed_archives"])

    if _INTERRUPTED:
        print(f"\n⏸  Paused! {done}/{len(archives)} archives, "
              f"{progress['total_positions']:,} positions saved.")
        print(f"   Resume: python scripts/process_data.py "
              f"{' '.join(str(a) for a in args.inputs)} -o {args.output}\n")
    else:
        file_mb = args.output.stat().st_size / 1e6
        print(f"\n✅ ETL complete!")
        print(f"   {progress['total_positions']:,} positions → {args.output} ({file_mb:.1f} MB)")
        print(f"\n   Next: python train.py --data {args.output}\n")


if __name__ == "__main__":
    main()

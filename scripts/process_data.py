#!/usr/bin/env python3
import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional

import chess
import chess.pgn
import h5py
import numpy as np
import py7zr
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataConfig
from src.board_encoder import encode_board

# Configure logging to work nicely with TQDM
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[TqdmLoggingHandler()]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
STOP_SIGNAL = False

def signal_handler(signum, frame):
    global STOP_SIGNAL
    if STOP_SIGNAL:
        logger.warning("Force quitting...")
        sys.exit(1)
    STOP_SIGNAL = True
    logger.warning("\nStop signal received! Finishing current batch then saving...")

# Global worker config (read-only)
_WORKER_CONFIG: Optional[DataConfig] = None

def worker_init(cfg: DataConfig):
    """Initialize worker process with shared configuration."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = cfg

# ---------------------------------------------------------------------------
# Worker Logic: PGN Parsing -> Lightweight Tuples
# ---------------------------------------------------------------------------

def process_single_game(pgn_text: str) -> List[Tuple[str, str, float]]:
    """
    Parses a single PGN string and extracts training positions.
    Returns lightweight strings (FEN, UCI) to minimize IPC serialization overhead.
    """
    cfg = _WORKER_CONFIG
    if not cfg:
        return []

    try:
        # Fast parse using io.StringIO
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []

        # 1. Header Filtering
        headers = game.headers
        try:
            w_elo = int(headers.get("WhiteElo", "0"))
            b_elo = int(headers.get("BlackElo", "0"))
        except ValueError:
            return []

        if w_elo < cfg.min_elo or b_elo < cfg.min_elo:
            return []

        result = headers.get("Result")
        if result not in ["1-0", "0-1", "1/2-1/2"]:
            return []

        # 2. Replay Game
        board = game.board()
        candidates = [] # (fen, move_uci, ply)
        node = game
        
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            candidates.append((board.fen(), move.uci(), board.fullmove_number))
            board.push(move)
            node = next_node

        if len(candidates) < cfg.min_ply_count:
            return []

        # 3. Value Assignment
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0

        # 4. Sampling
        # Filter opening moves if configured
        valid_candidates = [
            c for c in candidates 
            if c[2] > cfg.opening_cutoff_move or random.random() < cfg.opening_keep_prob
        ]
        
        # Random sample to avoid correlation
        if len(valid_candidates) > cfg.positions_per_game:
            valid_candidates = random.sample(valid_candidates, cfg.positions_per_game)

        # 5. Format Output
        output_samples = []
        for fen, move_uci, _ in valid_candidates:
            # Value is always relative to the active player
            is_white_turn = " w " in fen
            relative_value = game_value if is_white_turn else -game_value
            output_samples.append((fen, move_uci, relative_value))

        return output_samples

    except Exception:
        # Fail silently on bad PGNs to keep pipeline moving
        return []

# ---------------------------------------------------------------------------
# Main Process: Heavy Encoding & HDF5 I/O
# ---------------------------------------------------------------------------

def move_to_policy_index(move_uci: str, fen: str) -> int:
    """Maps a move to the 0-4095 policy index."""
    move = chess.Move.from_uci(move_uci)
    board = chess.Board(fen) # Lightweight re-parse just for color check is fine here
    
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Flip logic for Black to ensure perspective-independent training
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
        
    return from_sq * 64 + to_sq

def encode_and_save_batch(samples: List[Tuple[str, str, float]], h5_path: Path) -> int:
    """
    Encodes FENs to numpy tensors and appends to HDF5.
    Returns number of saved positions.
    """
    if not samples:
        return 0

    # Batch encode
    states, policies, values = [], [], []
    
    for fen, move_uci, val in samples:
        board = chess.Board(fen)
        states.append(encode_board(board))
        policies.append(move_to_policy_index(move_uci, fen))
        values.append(val)

    # Convert to numpy
    # states shape: (N, 18, 8, 8)
    np_states = np.stack(states).astype(np.float32)
    np_policies = np.array(policies, dtype=np.int64)
    np_values = np.array(values, dtype=np.float32)

    # Append to HDF5
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if h5_path.exists() else "w"
    
    with h5py.File(h5_path, mode) as f:
        # Create datasets if new
        if "states" not in f:
            f.create_dataset("states", data=np_states, maxshape=(None, 18, 8, 8), chunks=(4096, 18, 8, 8), compression="gzip")
            f.create_dataset("policies", data=np_policies, maxshape=(None,), chunks=(4096,), compression="gzip")
            f.create_dataset("values", data=np_values, maxshape=(None,), chunks=(4096,), compression="gzip")
            return len(samples)
        
        # Resize and append
        current_size = f["states"].shape[0]
        new_size = current_size + len(samples)
        
        f["states"].resize(new_size, axis=0)
        f["policies"].resize(new_size, axis=0)
        f["values"].resize(new_size, axis=0)
        
        f["states"][current_size:] = np_states
        f["policies"][current_size:] = np_policies
        f["values"][current_size:] = np_values

    return len(samples)

def parse_pgn_stream(text: str) -> List[str]:
    """Splits a large PGN block into individual game strings."""
    games = []
    buffer = []
    in_game = False
    
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("[Event"):
            if in_game and buffer:
                games.append("".join(buffer))
                buffer = []
            in_game = True
        
        if in_game:
            buffer.append(line)
            
    if buffer:
        games.append("".join(buffer))
    return games

def process_archive(archive_path: Path, h5_path: Path, config: DataConfig) -> int:
    """Extracts, processes, and saves an entire 7z archive."""
    global STOP_SIGNAL
    
    logger.info(f"Extracting: {archive_path.name}...")
    
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            with tempfile.TemporaryDirectory() as temp_dir:
                z.extractall(path=temp_dir)
                
                # Gather all PGNs
                pgn_files = list(Path(temp_dir).rglob("*.pgn"))
                if not pgn_files:
                    logger.warning(f"No PGN files found in {archive_path.name}")
                    return 0
                
                # Read all games into memory (strings)
                # Note: If single PGN files are massive (>2GB), stream read might be needed.
                # For Gigabase, they are usually split.
                all_games = []
                for pgn_file in pgn_files:
                    text = pgn_file.read_text(encoding="utf-8", errors="replace")
                    all_games.extend(parse_pgn_stream(text))
                
                logger.info(f"Found {len(all_games):,} games. Starting parallel processing...")
                
                # Setup Pool
                batch_buffer = []
                total_saved = 0
                chunk_size = 500 # Games per worker task
                
                with mp.Pool(processes=config.num_workers, initializer=worker_init, initargs=(config,)) as pool:
                    iterator = pool.imap_unordered(process_single_game, all_games, chunksize=chunk_size)
                    
                    for game_samples in tqdm(iterator, total=len(all_games), desc="Processing", unit="game"):
                        if STOP_SIGNAL:
                            logger.warning("Terminating worker pool...")
                            pool.terminate()
                            break
                        
                        if game_samples:
                            batch_buffer.extend(game_samples)
                        
                        # Flush to HDF5 when buffer fills
                        if len(batch_buffer) >= 50_000:
                            n = encode_and_save_batch(batch_buffer, h5_path)
                            total_saved += n
                            batch_buffer = []
                            
                    # Final flush
                    if batch_buffer and not STOP_SIGNAL:
                        total_saved += encode_and_save_batch(batch_buffer, h5_path)
                
                return total_saved

    except Exception as e:
        logger.error(f"Failed to process archive {archive_path.name}: {e}")
        return 0

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="ChessNet ETL Pipeline")
    parser.add_argument("archives", nargs="+", type=Path, help="Input .7z files")
    parser.add_argument("--output", "-o", type=Path, default="data/train.h5", help="Output HDF5 file")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()
    
    config = DataConfig()
    config.num_workers = args.workers
    
    # Progress Tracking
    progress_file = args.output.with_suffix(".progress.json")
    progress = {"completed": [], "total_positions": 0}
    
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
        logger.info(f"Resuming... {len(progress['completed'])} archives already done.")
        
        # Consistency Check: Rollback HDF5 if it has more data than tracking says (crash recovery)
        if args.output.exists():
            with h5py.File(args.output, "a") as f:
                real_count = f["states"].shape[0]
                if real_count > progress["total_positions"]:
                    logger.warning(f"Data inconsistency detected! Rolling back HDF5 from {real_count} to {progress['total_positions']}")
                    f["states"].resize(progress["total_positions"], axis=0)
                    f["policies"].resize(progress["total_positions"], axis=0)
                    f["values"].resize(progress["total_positions"], axis=0)

    # Filter input list
    to_process = [p for p in args.archives if p.name not in progress["completed"]]
    logger.info(f"Archives to process: {len(to_process)}")

    for archive in to_process:
        if STOP_SIGNAL:
            break
            
        count = process_archive(archive, args.output, config)
        
        if STOP_SIGNAL:
            logger.info("Graceful stop triggered. Progress for current archive NOT saved to ensure consistency.")
            break
            
        # Update progress
        progress["completed"].append(archive.name)
        progress["total_positions"] += count
        
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
            
        logger.info(f"Completed {archive.name}. Total positions: {progress['total_positions']:,}")

    logger.info("ETL Pipeline finished.")

if __name__ == "__main__":
    main()
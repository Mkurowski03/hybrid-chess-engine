#!/usr/bin/env python3
"""
Automated Engine Tournament Runner.

Orchestrates a round-robin tournament between the local HybridEngine
and various skill levels of Stockfish using 'cutechess-cli'.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CUTECHESS_DIR = PROJECT_ROOT / "cutechess-1.4.0-win64"

# Engine Settings
HERO_NAME = "Hybrid_Beast_V2"
HERO_CMD = PROJECT_ROOT / "run_hybrid_beast.bat"

# Tournament Settings
TIME_CONTROL = "180+2.0"  # 3 minutes + 2s increment
ROUNDS = 2                # Pairs of games (white/black)
CONCURRENCY = 3           # Number of parallel games
OUTPUT_FILE = "hybrid_beast_results.pgn"

# Opponent Ladder (Stockfish Elo ratings)
OPPONENT_LEVELS = [2100, 2300, 2500]
# -----------------------------------------------------------------------------


def find_binary(name: str, search_paths: List[Path]) -> Optional[Path]:
    """
    Locates an executable by checking system PATH and specific directories.
    """
    # 1. Check system PATH
    sys_path = shutil.which(name)
    if sys_path:
        return Path(sys_path)

    # 2. Check explicit folders
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    for path in search_paths:
        candidate = path / exe_name
        if candidate.exists():
            return candidate
        
        # Check without .exe extension just in case
        candidate_no_ext = path / name
        if candidate_no_ext.exists():
            return candidate_no_ext

    return None


def run_tournament():
    # 1. Locate Dependencies
    logger.info("Locating executables...")
    
    cutechess = find_binary("cutechess-cli", [PROJECT_ROOT, CUTECHESS_DIR])
    if not cutechess:
        logger.critical("cutechess-cli not found. Please install it or place in project root.")
        sys.exit(1)

    stockfish = find_binary("stockfish", [PROJECT_ROOT, CUTECHESS_DIR])
    if not stockfish:
        logger.critical("Stockfish not found. Please download it to the project root.")
        sys.exit(1)

    if not HERO_CMD.exists():
        logger.critical(f"Hero engine script not found: {HERO_CMD}")
        sys.exit(1)

    logger.info(f"CuteChess: {cutechess}")
    logger.info(f"Stockfish: {stockfish}")
    logger.info(f"HybridEng: {HERO_CMD}")

    # 2. Build Command Arguments (List format is safer than Shell=True)
    # Common settings applied to all engines
    cmd = [
        str(cutechess),
        "-concurrency", str(CONCURRENCY),
        "-tournament", "round-robin",
        "-rounds", str(ROUNDS),
        "-repeat",          # Play each opening twice (flip colors)
        "-recover",         # Restart engine if it crashes
        "-pgnout", OUTPUT_FILE,
        "-each",            # Apply following args to all engines
        f"tc={TIME_CONTROL}",
        "proto=uci"
    ]

    # 3. Add Hero Engine
    cmd.extend([
        "-engine",
        f"name={HERO_NAME}",
        f"cmd={HERO_CMD}"
    ])

    # 4. Add Opponents
    for elo in OPPONENT_LEVELS:
        cmd.extend([
            "-engine",
            f"name=Stockfish-{elo}",
            f"cmd={stockfish}",
            "option.UCI_LimitStrength=true",
            f"option.UCI_Elo={elo}"
        ])

    # 5. Execute
    logger.info("-" * 50)
    logger.info(f"Starting Tournament: {HERO_NAME} vs Stockfish ({OPPONENT_LEVELS})")
    logger.info(f"Time Control: {TIME_CONTROL}")
    logger.info("-" * 50)

    try:
        # Popen allows us to stream stdout line by line
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            shell=False # Safer, avoids quoting hell
        ) as process:
            
            # Stream output
            for line in process.stdout:
                print(line.strip())

            return_code = process.wait()
            
            if return_code == 0:
                logger.info("Tournament finished successfully.")
            else:
                logger.error(f"Tournament failed with exit code {return_code}")

    except KeyboardInterrupt:
        logger.warning("Tournament interrupted by user.")
    except Exception as e:
        logger.exception("Unexpected error during execution")


if __name__ == "__main__":
    run_tournament()
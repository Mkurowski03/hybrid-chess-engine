#!/usr/bin/env python3
"""
Timeout Analysis Tool.

Scans a PGN file for games that ended due to time forfeiture ("on time").
Useful for identifying performance bottlenecks in engine matches (e.g., flagging).
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_timeout_matchups(pgn_path: Path) -> Set[Tuple[str, str]]:
    """
    Parses PGN to find internal engine matchups that ended in a timeout.
    
    Returns:
        Set of (EngineA, EngineB) tuples.
    """
    timeout_pairs = set()
    
    # Track current game state
    current_white = "Unknown"
    current_black = "Unknown"
    
    # Pre-compile regex for speed
    re_white = re.compile(r'\[White "(.*)"\]')
    re_black = re.compile(r'\[Black "(.*)"\]')
    
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # 1. Update Headers
                # We assume standard PGN formatting where headers precede the moves/result
                mw = re_white.search(line)
                if mw:
                    current_white = mw.group(1)
                    continue

                mb = re_black.search(line)
                if mb:
                    current_black = mb.group(1)
                    continue

                # 2. Check for Timeout Indicator
                # "on time" usually appears in the result comment: { White wins on time }
                if "on time" in line:
                    # Filter out reference engine (Stockfish) to focus on internal regressions
                    if "Stockfish" in current_white or "Stockfish" in current_black:
                        continue
                    
                    # Store as sorted tuple so (A vs B) and (B vs A) count as the same matchup
                    pair = tuple(sorted([current_white, current_black]))
                    timeout_pairs.add(pair)

    except FileNotFoundError:
        logger.error(f"File not found: {pgn_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading PGN: {e}")
        sys.exit(1)

    return timeout_pairs


def main():
    parser = argparse.ArgumentParser(description="Analyze PGN for engine timeouts.")
    parser.add_argument(
        "file", 
        nargs="?", 
        default=Path("gauntlet_results.pgn"), 
        type=Path,
        help="Path to the PGN file (default: gauntlet_results.pgn)"
    )
    args = parser.parse_args()

    if not args.file.exists():
        logger.error(f"Target file '{args.file}' does not exist.")
        return

    logger.info(f"Scanning {args.file} for timeouts...")
    timeouts = find_timeout_matchups(args.file)

    print("\n" + "=" * 40)
    print("MATCHUPS WITH TIMEOUTS")
    print("=" * 40)
    
    if timeouts:
        for p1, p2 in sorted(list(timeouts)):
            print(f" {p1} <---> {p2}")
    else:
        print("No internal timeouts detected.")
    
    print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
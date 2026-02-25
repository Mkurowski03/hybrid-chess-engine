#!/usr/bin/env python3
"""
PGN Standings Calculator.

Parses a PGN file to calculate a tournament cross-table/leaderboard.
Designed to be memory-efficient (streaming) and robust to file updates.
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_pgn_standings(file_path: Path) -> Tuple[Dict[str, float], int]:
    """
    Streams a PGN file line-by-line and calculates scores.
    
    Returns:
        Tuple of (scores_dict, games_count)
    """
    scores = defaultdict(float)
    games_count = 0
    
    # State tracking for the current game
    current_white = None
    current_black = None
    
    # Regex patterns (compiled for speed)
    # PGN tags format: [Key "Value"]
    re_white = re.compile(r'^\[White "(.*)"\]')
    re_black = re.compile(r'^\[Black "(.*)"\]')
    re_result = re.compile(r'^\[Result "(.*)"\]')

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                
                # Extract White
                m_white = re_white.match(line)
                if m_white:
                    current_white = m_white.group(1)
                    continue

                # Extract Black
                m_black = re_black.match(line)
                if m_black:
                    current_black = m_black.group(1)
                    continue

                # Extract Result (Trigger score update)
                m_result = re_result.match(line)
                if m_result:
                    result = m_result.group(1)
                    
                    # Validation: Ensure we have player names before scoring
                    if current_white and current_black:
                        games_count += 1
                        
                        # Initialize players in dict to ensure they appear even with 0 pts
                        if current_white not in scores: scores[current_white] = 0.0
                        if current_black not in scores: scores[current_black] = 0.0

                        if result == '1-0':
                            scores[current_white] += 1.0
                        elif result == '0-1':
                            scores[current_black] += 1.0
                        elif result == '1/2-1/2':
                            scores[current_white] += 0.5
                            scores[current_black] += 0.5
                    
                    # Reset state for next game
                    current_white = None
                    current_black = None

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading PGN: {e}")
        sys.exit(1)

    return scores, games_count


def print_standings(scores: Dict[str, float], total_games: int):
    """Formats and prints the leaderboard."""
    sorted_scores = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    print("\n" + "=" * 40)
    print(f"LIVE STANDINGS (Games: {total_games})")
    print("=" * 40)
    print(f"{'Rank':<5} {'Player':<20} {'Points':>6}")
    print("-" * 40)
    
    for rank, (player, score) in enumerate(sorted_scores, 1):
        print(f"{rank:<5} {player:<20} {score:>6.1f}")
    print("=" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PGN Standings Calculator")
    parser.add_argument("file", nargs="?", default="rapid_results.pgn", type=Path, help="Path to PGN file")
    args = parser.parse_args()

    if not args.file.exists():
        logger.error(f"Target file '{args.file}' does not exist.")
        sys.exit(1)

    logger.info(f"Processing {args.file}...")
    scores, count = parse_pgn_standings(args.file)
    
    if count == 0:
        logger.warning("No games found or file is empty.")
    else:
        print_standings(scores, count)


if __name__ == "__main__":
    main()
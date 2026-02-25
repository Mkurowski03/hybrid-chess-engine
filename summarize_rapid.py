#!/usr/bin/env python3
"""
PGN Tournament Statistics Parser.

Analyzes PGN files to generate leaderboards and head-to-head records.
Designed for memory efficiency using stream processing.
"""

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TournamentStats:
    """Aggregates scores and statistics for a tournament."""
    def __init__(self):
        self.total_games = 0
        self.scores: Dict[str, float] = defaultdict(float)
        # H2H format: (PlayerA, PlayerB) -> {PlayerA: wins, PlayerB: wins, 'draws': draws}
        self.h2h: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update(self, white: str, black: str, result: str):
        """Updates stats with a single game result."""
        self.total_games += 1
        
        # Ensure players exist in score dict even if they have 0 points
        if white not in self.scores: self.scores[white] = 0.0
        if black not in self.scores: self.scores[black] = 0.0

        # Sort pair for consistent H2H key
        pair = tuple(sorted([white, black]))
        
        if result == '1-0':
            self.scores[white] += 1.0
            self.h2h[pair][white] += 1
        elif result == '0-1':
            self.scores[black] += 1.0
            self.h2h[pair][black] += 1
        elif result == '1/2-1/2':
            self.scores[white] += 0.5
            self.scores[black] += 0.5
            self.h2h[pair]['draws'] += 1


def parse_pgn(file_path: Path) -> Optional[TournamentStats]:
    """
    Parses a PGN file line-by-line to extract tournament statistics.
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    stats = TournamentStats()
    
    # Pre-compile regex for performance
    re_white = re.compile(r'^\[White "(.*)"\]')
    re_black = re.compile(r'^\[Black "(.*)"\]')
    re_result = re.compile(r'^\[Result "(.*)"\]')

    current_white = None
    current_black = None

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                
                # State Machine: Accumulate headers -> Commit on Result
                m_white = re_white.match(line)
                if m_white:
                    current_white = m_white.group(1)
                    continue

                m_black = re_black.match(line)
                if m_black:
                    current_black = m_black.group(1)
                    continue

                m_result = re_result.match(line)
                if m_result:
                    if current_white and current_black:
                        stats.update(current_white, current_black, m_result.group(1))
                    
                    # Reset state
                    current_white = None
                    current_black = None

    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None

    return stats


def print_report(title: str, stats: Optional[TournamentStats]):
    """Formats and prints the tournament report."""
    print(f"\n### {title}")
    
    if not stats or stats.total_games == 0:
        print("No data found.\n")
        print("-" * 40)
        return

    print(f"**Total Games Completed:** {stats.total_games} / 90\n")
    
    print("**Live Standings:**")
    # Sort by score descending
    ranking = sorted(stats.scores.items(), key=lambda x: x[1], reverse=True)
    for player, score in ranking:
        print(f"- **{player}**: {score} pts")

    print("\n**Key Head-to-Head Matchups:**")
    for pair, data in stats.h2h.items():
        p1, p2 = pair
        # Default to 0 if key missing
        s1 = int(data.get(p1, 0))
        s2 = int(data.get(p2, 0))
        d = int(data.get('draws', 0))
        print(f"- {p1} ({s1}) vs {p2} ({s2}) | Draws: {d}")
    
    print("-" * 40)


def main():
    stats = parse_pgn(Path('rapid_results.pgn'))
    print_report("Rapid Gauntlet (3m + 2s @ 2200 Sims)", stats)


if __name__ == "__main__":
    main()
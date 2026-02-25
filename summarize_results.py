#!/usr/bin/env python3
"""
Tournament Results Analyzer.

Parses 'tournament_results.json' to generate a performance report
including Elo-adjusted metrics and detailed head-to-head records.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

RESULTS_FILE = Path('tournament_results.json')


def load_games(file_path: Path) -> List[Dict[str, Any]]:
    """Loads and filters valid games from the JSON log."""
    if not file_path.exists():
        logger.error(f"Results file not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Filter aborted/crashed games
        valid_games = [g for g in data if g.get('status') != 'aborted']
        logger.info(f"Loaded {len(valid_games)} valid games from {len(data)} entries.")
        return valid_games
        
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return []


def parse_rating(rating: Any) -> Optional[int]:
    """Cleanly parses ratings like '1500?' or 1600."""
    try:
        return int(str(rating).replace('?', ''))
    except (ValueError, TypeError):
        return None


def calculate_statistics(games: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Aggregates stats per profile."""
    # Structure: profile -> {wins, losses, ...}
    stats = defaultdict(lambda: {
        'wins': 0, 'losses': 0, 'draws': 0, 
        'total': 0, 'opp_ratings': [], 
        'weighted_score': 0.0,
        'matchups': defaultdict(lambda: {'w': 0, 'l': 0, 'd': 0, 'rating': 0})
    })

    for g in games:
        profile = g.get('profile', 'Unknown')
        outcome = g.get('outcome')
        opponent = g.get('opponent', 'Unknown')
        
        # Parse rating
        rating = parse_rating(g.get('opponent_rating'))
        if rating is None:
            continue

        if outcome in ['win', 'loss', 'draw']:
            s = stats[profile]
            s['total'] += 1
            s['opp_ratings'].append(rating)
            
            # Matchup tracking
            s['matchups'][opponent]['rating'] = rating

            # Scoring
            # Weighted Score: Experimental metric rewarding wins against higher rated opponents
            base_weight = rating / 1500.0

            if outcome == 'win':
                s['wins'] += 1
                s['weighted_score'] += base_weight
                s['matchups'][opponent]['w'] += 1
            elif outcome == 'loss':
                s['losses'] += 1
                s['matchups'][opponent]['l'] += 1
            elif outcome == 'draw':
                s['draws'] += 1
                s['weighted_score'] += base_weight * 0.5
                s['matchups'][opponent]['d'] += 1

    return stats


def calculate_tpr(avg_rating: float, score_pct: float) -> int:
    """
    Calculates Tournament Performance Rating (TPR) using the linear approximation method.
    Formula: TPR = Ra + dp
    where dp = 800 * score_pct - 400
    """
    if score_pct >= 1.0:
        return int(avg_rating + 400)
    elif score_pct <= 0.0:
        return int(avg_rating - 400)
    
    dp = 800 * score_pct - 400
    return int(avg_rating + dp)


def print_performance_table(stats: Dict[str, Dict]):
    """Prints the main leaderboard."""
    print("\n--- Profile Performance (Elo Adjusted) ---")
    header = f"{'Profile':<25} | {'Score':<6} | {'W.Score':<7} | {'Gms':<5} | {'Avg Opp':<9} | {'Est. TPR':<10}"
    print(header)
    print("-" * len(header))

    # Sort by raw score points (Wins + 0.5 * Draws)
    sorted_profiles = sorted(
        stats.items(), 
        key=lambda item: item[1]['wins'] + 0.5 * item[1]['draws'], 
        reverse=True
    )

    for profile, s in sorted_profiles:
        if s['total'] == 0:
            continue

        score = s['wins'] + 0.5 * s['draws']
        score_pct = score / s['total']
        avg_opp_elo = sum(s['opp_ratings']) / s['total']
        tpr = calculate_tpr(avg_opp_elo, score_pct)

        print(f"{profile:<25} | {score}/{s['total']:<4} | {s['weighted_score']:<7.2f} | {s['total']:<5} | {avg_opp_elo:<9.1f} | ~{tpr:<10}")


def print_detailed_matchups(stats: Dict[str, Dict]):
    """Prints head-to-head records."""
    print("\n--- Detailed Matchups (Ordered by Opponent Rating) ---")
    
    for profile, s in stats.items():
        if s['total'] == 0: 
            continue
            
        print(f"\n[{profile}]")
        
        # Sort opponents by their rating
        opponents = sorted(
            s['matchups'].items(), 
            key=lambda x: x[1]['rating']
        )
        
        for opp_name, m in opponents:
            record = f"{m['w']}W - {m['l']}L - {m['d']}D"
            print(f"  vs {opp_name:<20} ({m['rating']}): {record}")


def main():
    games = load_games(RESULTS_FILE)
    if not games:
        print("No valid games to analyze.")
        return

    stats = calculate_statistics(games)
    print_performance_table(stats)
    print_detailed_matchups(stats)


if __name__ == "__main__":
    main()
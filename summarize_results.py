import json
import pandas as pd
from collections import defaultdict

with open('tournament_results.json', 'r') as f:
    data = json.load(f)

# Filter out the aborted games from the crash earlier today
valid_games = [g for g in data if g.get('status') != 'aborted']

print(f"Total valid games played so far: {len(valid_games)}\n")

profile_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'total': 0, 'opp_ratings': [], 'weighted_score': 0.0})

for g in valid_games:
    profile = g.get('profile', 'Unknown')
    outcome = g.get('outcome', 'unknown')
    opp_rating = g.get('opponent_rating', 'Unknown')
    
    # Try to parse opponent rating
    try:
        opp_rating_int = int(str(opp_rating).replace('?', ''))
    except ValueError:
        continue # Skip if no valid rating
        
    if outcome in ['win', 'loss', 'draw']:
        stats = profile_stats[profile]
        stats['total'] += 1
        stats['opp_ratings'].append(opp_rating_int)
        if outcome == 'win':
            stats['wins'] += 1
            stats['weighted_score'] += opp_rating_int / 1500.0
        elif outcome == 'loss':
            stats['losses'] += 1
        elif outcome == 'draw':
            stats['draws'] += 1
            stats['weighted_score'] += (opp_rating_int / 1500.0) * 0.5

print("--- Profile Performance (Elo Adjusted) ---")
print(f"{'Profile':<25} | {'Score':<5} | {'W.Score':<7} | {'Games':<5} | {'Avg Opp Elo':<11} | {'Est. Perf (TPR)':<15}")
print("-" * 85)

for profile, stats in profile_stats.items():
    total = stats['total']
    if total == 0:
        continue
        
    wins = stats['wins']
    losses = stats['losses']
    draws = stats['draws']
    
    score = wins + 0.5 * draws
    score_pct = score / total
    
    avg_opp_elo = sum(stats['opp_ratings']) / total
    
    # Simple linear approximation for TPR (Tournament Performance Rating)
    # TPR = Average_Opp_Rating + 400 * (Wins - Losses) / Total
    # A more standard one is based on normal distribution, but the linear rule of thumb is fine for small samples
    # Let's use the +400/-400 rule:
    # Or, TPR = Average_Opp_Rating + 800 * (score_pct - 0.5)
    
    if score_pct == 1.0:
        tpr = avg_opp_elo + 400
    elif score_pct == 0.0:
        tpr = avg_opp_elo - 400
    else:
        # FIDE formula approximation: dp = 800 * p - 400
        dp = 800 * score_pct - 400
        tpr = avg_opp_elo + dp
        
    w_score = stats['weighted_score']
    print(f"{profile:<25} | {score}/{total:<3} | {w_score:<7.2f} | {total:<5} | {avg_opp_elo:<11.1f} | ~{tpr:.0f}")

print("\n--- Detailed Matchups (ordered by Opponent Elo) ---")
matchups = defaultdict(lambda: defaultdict(lambda: {'win': 0, 'loss': 0, 'draw': 0, 'opp_rating': 0}))

for g in valid_games:
    profile = g.get('profile', 'Unknown')
    opponent = g.get('opponent', 'Unknown')
    opp_rating = g.get('opponent_rating', 'Unknown')
    outcome = g.get('outcome', 'unknown')
    
    try:
        opp_rating_int = int(str(opp_rating).replace('?', ''))
    except ValueError:
        continue
        
    if outcome in ['win', 'loss', 'draw']:
        key = f"{opponent}"
        matchups[profile][key][outcome] += 1
        matchups[profile][key]['opp_rating'] = opp_rating_int

for profile, opp_stats in profile_stats.items():
    print(f"\n[{profile}]")
    # Sort opponents by rating
    sorted_opps = sorted(matchups[profile].items(), key=lambda x: x[1]['opp_rating'])
    for opp, stats in sorted_opps:
        print(f"  vs {opp} ({stats['opp_rating']}): {stats['win']}W - {stats['loss']}L - {stats['draw']}D")

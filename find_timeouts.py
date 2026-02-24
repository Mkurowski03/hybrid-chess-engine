import re

with open('gauntlet_results.pgn', 'r') as f:
    text = f.read()

games = text.split('\n\n\n')
timeout_matchups = set()

for game in games:
    if 'on time' in game:
        white_match = re.search(r'\[White "(.*)"\]', game)
        black_match = re.search(r'\[Black "(.*)"\]', game)
        
        if white_match and black_match:
            w = white_match.group(1)
            b = black_match.group(1)
            
            # We don't care about Stockfish timeouts, only our internal parameter testing
            if 'Stockfish' not in w and 'Stockfish' not in b:
                # Sort alphabetically to treat White vs Black and Black vs White as the same matchup pair
                matchup = tuple(sorted([w, b]))
                timeout_matchups.add(matchup)

print("--- Engine Matchups with Timeouts ---")
for m in timeout_matchups:
    print(f"{m[0]} <---> {m[1]}")
    
if not timeout_matchups:
    print("No internal engine matchups ended in timeouts.")

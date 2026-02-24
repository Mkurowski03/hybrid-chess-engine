import re
import os

def parse_pgn(filename):
    if not os.path.exists(filename):
        return None
        
    with open(filename, 'r') as f:
        text = f.read()

    whites = re.findall(r'\[White "(.*)"\]', text)
    blacks = re.findall(r'\[Black "(.*)"\]', text)
    results = re.findall(r'\[Result "(.*)"\]', text)

    scores = {}
    head_to_head = {}

    for w, b, r in zip(whites, blacks, results):
        if w not in scores: scores[w] = 0.0
        if b not in scores: scores[b] = 0.0
        
        pair = tuple(sorted([w, b]))
        if pair not in head_to_head:
            head_to_head[pair] = {w: 0.0, b: 0.0, 'draws': 0}
            
        if r == '1-0':
            scores[w] += 1.0
            head_to_head[pair][w] += 1
        elif r == '0-1':
            scores[b] += 1.0
            head_to_head[pair][b] += 1
        elif r == '1/2-1/2':
            scores[w] += 0.5
            scores[b] += 0.5
            head_to_head[pair]['draws'] += 1

    return {'total': len(results), 'scores': scores, 'h2h': head_to_head}

print("### 1. Initial 60-Game Gauntlet (3m + 2s)")
res1 = parse_pgn('gauntlet_results.pgn')
if res1:
    print(f"**Total Games:** {res1['total']}\n")
    print("**Standings:**")
    for k, v in sorted(res1['scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"- **{k}**: {v} pts")
    
    print("\n**Head-to-Head Breakdown:**")
    for pair, data in res1['h2h'].items():
        w, b = pair
        print(f"- {w} ({data[w]}) vs {b} ({data[b]}) | Draws: {data['draws']}")
else:
    print("No data found.\n")

print("\n---\n")

print("### 2. Tiebreaker Rematch (5m + 5s)")
res2 = parse_pgn('replay_timeout_results.pgn')
if res2:
    print(f"**Total Games:** {res2['total']}\n")
    print("**Standings:**")
    for k, v in sorted(res2['scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"- **{k}**: {v} pts")
        
    print("\n**Head-to-Head Breakdown:**")
    for pair, data in res2['h2h'].items():
        w, b = pair
        print(f"- {w} ({data[w]}) vs {b} ({data[b]}) | Draws: {data['draws']}")
else:
    print("No data found.")

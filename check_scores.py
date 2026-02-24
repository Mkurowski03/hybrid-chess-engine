import re

with open('rapid_results.pgn') as f:
    text = f.read()

whites = re.findall(r'\[White "(.*)"\]', text)
blacks = re.findall(r'\[Black "(.*)"\]', text)
results = re.findall(r'\[Result "(.*)"\]', text)

print(f"Total Games Completed: {len(results)}\n")

scores = {}
for w, b, r in zip(whites, blacks, results):
    if w not in scores: scores[w] = 0.0
    if b not in scores: scores[b] = 0.0
    
    if r == '1-0':
        scores[w] += 1.0
    elif r == '0-1':
        scores[b] += 1.0
    elif r == '1/2-1/2':
        scores[w] += 0.5
        scores[b] += 0.5

print("Live Standings:")
print("-" * 30)
for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{k.ljust(20)}: {v} pts")

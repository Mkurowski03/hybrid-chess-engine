import time
import requests

URL = "http://127.0.0.1:5000/api/move"
FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"  # Italian Game

def test_depth(depth):
    print(f"Testing Depth {depth}...")
    start = time.time()
    try:
        resp = requests.post(URL, json={"fen": FEN, "depth": depth}, timeout=120)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Move: {data.get('move')}")
            print(f"  Eval: {data.get('evaluation')}")
        else:
            print(f"  Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Warmup / Cache population
    test_depth(3)
    # Target test
    test_depth(4)

#!/usr/bin/env python3
"""
ChessNet API Latency & Correctness Tester.

A robust client to benchmark the inference server's response time and 
stability across different search parameters.
"""

import argparse
import logging
import sys
import time
from typing import Optional, Dict, Any, List

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("API_TEST")

# Defaults
DEFAULT_URL = "http://127.0.0.1:5000/api/move"
# Italian Game Position
DEFAULT_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def benchmark_request(
    session: requests.Session, 
    url: str, 
    payload: Dict[str, Any], 
    timeout: float = 120.0
) -> Optional[float]:
    """
    Sends a single request and logs performance.
    Returns the elapsed time in seconds, or None if failed.
    """
    depth = payload.get("depth", "?")
    logger.info(f"Sending request: Depth={depth} | FEN={payload['fen'][:20]}...")

    start_time = time.perf_counter()
    
    try:
        response = session.post(url, json=payload, timeout=timeout)
        duration = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            move = data.get("move")
            eval_score = data.get("evaluation")
            
            logger.info(f"[200 OK] Time: {duration:.4f}s | Move: {move} | Eval: {eval_score}")
            return duration
            
        else:
            logger.error(f"[{response.status_code}] Error: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        logger.critical(f"Connection Failed: Is the server running at {url}?")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Request Timed Out after {timeout}s")
        return None
    except Exception as e:
        logger.exception(f"Unexpected Exception: {e}")
        return None


def run_benchmark_suite(args: argparse.Namespace):
    """Executes the warmup and main benchmark loops."""
    
    # Use a session for connection pooling (keep-alive)
    with requests.Session() as session:
        
        # 1. Warmup / Cache Population
        # This forces the server to load the model into GPU memory and JIT compile any optimized paths
        logger.info("--- Starting Warmup Phase ---")
        warmup_payload = {"fen": args.fen, "depth": args.warmup_depth}
        if benchmark_request(session, args.url, warmup_payload) is None:
            logger.critical("Warmup failed. Aborting benchmark.")
            sys.exit(1)

        # 2. Main Benchmark Loop
        logger.info(f"\n--- Starting Benchmark Phase ({len(args.depths)} tests) ---")
        results = []
        
        for depth in args.depths:
            payload = {
                "fen": args.fen, 
                "depth": depth,
                # Pass extra engine params if needed
                "sims": depth * 1000  # Example heuristic: map depth to sims if server expects it
            }
            
            elapsed = benchmark_request(session, args.url, payload, timeout=args.timeout)
            if elapsed:
                results.append((depth, elapsed))
            
            # Small cooldown to let GPU cool/reset if needed
            time.sleep(0.5)

    # 3. Summary
    if results:
        print("\n" + "="*40)
        print(f" BENCHMARK SUMMARY")
        print("="*40)
        print(f"{'Depth':<10} | {'Latency (s)':<15}")
        print("-" * 30)
        for d, t in results:
            print(f"{d:<10} | {t:.4f}")
        print("="*40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ChessNet API Benchmark Tool")
    
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API Endpoint")
    parser.add_argument("--fen", type=str, default=DEFAULT_FEN, help="FEN string to analyze")
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8], help="List of depths to test")
    parser.add_argument("--warmup-depth", type=int, default=2, help="Depth for warmup request")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")

    args = parser.parse_args()
    
    run_benchmark_suite(args)


if __name__ == "__main__":
    main()
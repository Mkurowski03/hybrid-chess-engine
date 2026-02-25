# ChessNet-3070

![Rust](https://img.shields.io/badge/Core-Rust-orange?logo=rust)
![Python](https://img.shields.io/badge/Inference-Python_3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/ML-PyTorch_CUDA-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A hybrid Rust/Python chess engine. Rust handles the MCTS tree search, Python handles GPU inference. Trained on 24M+ grandmaster positions via imitation learning, optimized for consumer hardware (RTX 3070 Ti, 8GB VRAM).

**Estimated strength:** ~2200 Elo (Candidate Master). Defeated a 2450-rated Chess.com bot with 96.8% accuracy and zero blunders.

---

## Architecture

The engine is split across two runtimes connected by a zero-copy PyO3 bridge:

- **Rust** (`chess_engine_core`): MCTS tree traversal, PUCT selection, move generation via `shakmaty`, Syzygy tablebase probing, draw aversion logic. Nodes live in a flat `Vec<Node>` arena for cache-friendly access.
- **Python**: Batched neural inference (PyTorch or ONNX Runtime), time management, UCI protocol, web UI.

Board states are 18-channel 8x8 float32 tensors (6 piece types x 2 colors + castling rights + repetition + fifty-move counter), always encoded from the active player's perspective.

### Model

Dual-Headed ResNet (~1.2M parameters):
- **Backbone**: 10 residual blocks, 128 filters each
- **Policy head**: 4096-dim output (from_sq * 64 + to_sq)
- **Value head**: Scalar in [-1, 1]

### Performance

| Metric | Value |
|---|---|
| Nodes/sec (RTX 3070 Ti) | 6,500+ NPS |
| Max simulations/move | 80,000 |
| Training data | 24.3M positions from 3.7M games |
| Dataset | Lumbras Gigabase (Elo >= 2200, >= 30 moves) |

---

## Features

**Search**
- Monte Carlo Tree Search with PUCT exploration and Dirichlet noise
- Smart pruning: early stop when the best move leads by 3.5x visits
- Mate guard: 1-ply forced-mate check before neural evaluation
- Draw aversion: -5.0 penalty on repetition branches when winning

**Endgames**
- 5-piece Syzygy tablebases (WDL/DTZ) probed directly from Rust
- Simplification bias: captures are favored when winning, to reach tablebase range

**Inference**
- ONNX Runtime with TensorRT > CUDA > CPU provider fallback (~35% faster than raw PyTorch)
- Dynamic batching with zero-copy tensor handoff from Rust to Python

**Time Management**
- Adaptive allocation (up to 16% of remaining clock on critical moves)
- Panic mode: scales down simulations when clock drops below 5 seconds

---

## Quick Start

**Prerequisites:** Rust toolchain, Python 3.10+, CUDA-capable GPU.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Compile the Rust core
maturin develop --release

# (Optional) Export ONNX model for faster inference
python scripts/export_onnx.py --checkpoint checkpoints/baseline/chessnet_epoch9.pt

# Run the web UI
python app.py

# Or run as a UCI engine (for Arena, BanksiaGUI, cutechess-cli)
python src/uci.py --model checkpoints/baseline/chessnet_epoch9.pt
```

---

## Project Structure

```
chessbot/
  config.py              # All hyperparameters and constants
  app.py                 # Flask web server + REST API
  train.py               # Training script (basic)
  gauntlet.py            # Automated tournament runner (cutechess-cli)
  src/
    model.py             # Dual-Headed ResNet architecture
    board_encoder.py     # Board -> tensor encoding + move_to_policy_index
    mcts.py              # Pure-Python MCTS (fallback)
    engine.py            # Pure-Python engine (Alpha-Beta + MCTS)
    hybrid_engine.py     # Rust/Python hybrid engine (production)
    neural_backend.py    # PyTorch and ONNX inference backends
    uci.py               # UCI protocol server
    train_advanced.py    # Training script (resumable, early stopping)
    lichess_bot.py       # Lichess API integration
  scripts/
    process_data.py      # PGN -> HDF5 ETL pipeline
    convert_to_memmap.py # HDF5 -> numpy memmap conversion
    export_onnx.py       # PyTorch -> ONNX export
    download_syzygy.py   # Syzygy tablebase downloader
  tests/
    smoke_test.py        # Core unit + integration tests
    test_rust_encoding.py # Rust vs Python encoding parity
    test_rust_mcts.py    # Rust MCTS integration tests
  src_rust/               # Rust crate (chess_engine_core)
```

---

## The Full Story

This engine evolved from a Python script that couldn't find mate-in-1 (~1500 Elo) to a Master-level hybrid engine through a long series of architectural rewrites, pipeline crashes, and late-night debugging sessions.

**[Read the walkthrough](walkthrough.md)**

---

*Built by Mkurowski03*

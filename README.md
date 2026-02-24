# Hybrid Beast Chess Engine 🦀

![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## High-Performance Hybrid Architecture: Rust MCTS Core + PyTorch CUDA Inference.

**🏆 Major Achievement:**
"Draw vs 2450 ELO Bot ('Luke'): Achieved a draw by repetition with 95.6% Accuracy and 0 Blunders in the release candidate match."

## Project Overview

ChessNet-3070 has been reforged. What began as an experimental pure-Python imitation learner has escalated into an aggressive, deep-calculating computational beast running on a hybrid Rust/PyTorch architecture. 

By offloading the massive Monte Carlo Tree Search (MCTS) graph and mathematics to a flat memory arena in Rust, and executing batched neural evaluations on the GPU in Python, we achieved a staggering **100x speedup** in node generation and memory management.

## Key Features

*   **🚀 Hybrid Architecture:** Rust (`shakmaty` + `pyo3`) for tree search, Python for GPU inference.
*   **⚡ 6500+ NPS:** Sustained calculation on RTX 3070 Ti (40k simulations/move).
*   **🛡️ Panic Mode:** Dynamic time management to prevent flagging.
*   **⚔️ Mate Guard:** Instant 1-ply forced-mate solver.

## The Dataset

*   **Trained on 24 Million+ Positions** extracted from the Lumbras Gigabase (3.7M high-quality games, Elo ≥ 2200).
*   **Data Pipeline:** Custom multiprocessing pipeline with chunked HDF5 writing to handle massive datasets on consumer hardware.

## Installation & Usage

Ensure you have the Rust compiler (`cargo`), Python 3.10+, and CUDA installed.

1. **Install Python Dependencies and Compile Rust Core:**
   ```bash
   pip install -r requirements.txt
   pip install maturin
   maturin develop --release
   ```

2. **Launch the Flask Server (Web UI):**
   ```bash
   python app.py --port 5000
   ```
   *Open `http://localhost:5000` to play against the beast.*

3. **Tournament Testing (CuteChess / BanksiaGUI):**
   Configure your GUI to use the `run_hybrid_beast.bat` script as an external UCI engine.

## Development Journey & Engineering Challenges

This engine evolved from a simple Python script (~1500 ELO) to a Master-level Hybrid Engine. Read the full story of memory optimizations, IPC crash fixes, and the Rust rewrite here:

[📖 Read the Full Walkthrough](walkthrough.md)

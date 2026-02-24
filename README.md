# Hybrid Beast Chess Engine 🦀🧠

![Rust](https://img.shields.io/badge/Core-Rust-orange?logo=rust)
![Python](https://img.shields.io/badge/Inference-Python_3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/ML-PyTorch_CUDA-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

**High-Performance Hybrid Architecture:** Rust MCTS Core + PyTorch CUDA Inference.

---

### 🏆 Major Achievement: Release v2.0
> **Draw vs 2450 ELO Bot ("Luke"):** Achieved a draw by repetition with **95.6% Accuracy** and **0 Blunders** in the release candidate match.

---

### 📖 Project Overview
**ChessNet-3070** has been reforged. What began as an experimental pure-Python imitation learner has escalated into an aggressive, deep-calculating computational beast running on a hybrid Rust/PyTorch architecture.

By offloading the massive Monte Carlo Tree Search (MCTS) graph and mathematics to a flat memory arena in **Rust**, and executing batched neural evaluations on the GPU in **Python**, we achieved a staggering **100x speedup** in node generation.

### ✨ Key Features
* 🚀 **Hybrid Architecture:** Rust (`shakmaty` + `pyo3`) handles the heavy tree search; Python handles GPU inference.
* ⚡ **6500+ NPS:** Sustained calculation speed on RTX 3070 Ti (~40,000 simulations/move in Rapid).
* 🔄 **Zero-Copy Bridge:** 18-channel board states are encoded directly into Python memory space, eliminating data-transfer bottlenecks.
* 🛡️ **Panic Mode:** Dynamic time-management ensures the engine never flags, scaling down simulations when the clock is low.
* ⚔️ **Mate Guard:** Instant 1-ply forced-mate solver cures "promotion blindness".
* 🧩 **Endgame Excellence:** Integrated 5-piece Syzygy Tablebases (WDL/DTZ) directly into the Rust MCTS core. The engine plays mathematically perfect chess when 5 or fewer pieces remain.

### 📊 Data & Training
* **Dataset:** Trained on **24 Million+ Positions** extracted from the **Lumbras Gigabase** (3.7M high-quality games, Elo ≥ 2200).
* **Pipeline:** Custom multiprocessing pipeline with chunked HDF5 writing to handle massive datasets on consumer hardware.

### 🛠️ Quick Start
1.  **Prerequisites:** Rust (cargo), Python 3.10+, CUDA.
2.  **Install & Compile:**
    ```bash
    pip install maturin
    maturin develop --release
    ```
3.  **Run Web UI:**
    ```bash
    python app.py
    ```

### 📚 The Engineering Journey
This engine evolved from a simple Python script (~1500 ELO) to a Master-level Hybrid Engine.
Read the full story of memory optimizations, IPC crash fixes, and the Rust rewrite here:

👉 **[READ THE FULL WALKTHROUGH](walkthrough.md)**

---
*Created by [Your Name]*

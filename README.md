# Hybrid Beast Chess Engine 🦀

![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## High-Performance Hybrid Architecture: Rust MCTS Core + PyTorch CUDA Inference.

**🏆 Achievement: Draw vs 2450 ELO Bot (95.6% Accuracy) - Release v2.0**

ChessNet-3070 has been reforged. What began as an experimental pure-Python imitation learner has escalated into an aggressive, deep-calculating computational beast running on a hybrid Rust/PyTorch architecture. 

By offloading the massive Monte Carlo Tree Search (MCTS) graph and mathematics to a flat memory arena in Rust, and executing batched neural evaluations on the GPU in Python, we achieved a staggering **100x speedup** in node generation and memory management.

### Features
*   **Zero-Copy Bridge:** Rust (`shakmaty`) encodes 18-channel board states directly into Python memory space via `pyo3` and `numpy`, eliminating GPU data-transfer bottlenecks.
*   **6500+ NPS (Nodes Per Second):** Sustained calculation speed on an RTX 3070 Ti, allowing the "Hybrid Beast" profile to search **40,000 simulations per move**.
*   **Panic Mode:** Dynamic time-management protocol (`wtime`, `winc`) that aborts the Rust search loop immediately if the clock drops below a critical threshold, guaranteeing the engine never flags (loses on time).
*   **Mate Guard:** Instant 1-ply forced-mate solver executed before network inference to irrevocably cure standard MCTS "promotion blindness".

### Quick Start
Ensure you have the Rust compiler (`cargo`), Python 3.10+, and CUDA installed.

1. **Install Python Dependencies and Compile Rust Core:**
   ```bash
   pip install -r requirements.txt  # If applicable
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

### Final Verification
*   **Stockfish Matches:** Consistently defeats Stockfish 2100 in blitz. Equal footing against Stockfish 2300 in the mid-game.
*   **Grandmaster Validation:** Successfully drew against the simulated 2450 ELO "Luke" bot on Chess.com with **zero blunders** and **95.6% accuracy**.
*   **Estimated Strength:** Candidate Master (CM) Level — **~2200-2250 ELO**.

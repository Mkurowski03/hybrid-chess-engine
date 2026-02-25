#!/usr/bin/env python3
"""Export ChessNet PyTorch checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py [--checkpoint PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from src.model import ChessNet, build_model


def export_onnx(checkpoint_path: Path, output_path: Path) -> None:
    """Load a .pt checkpoint and export to ONNX."""
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")  # Export on CPU for portability
    model: ChessNet = build_model(ModelConfig())

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: (batch=1, channels=18, height=8, width=8)
    dummy_input = torch.randn(1, 18, 8, 8, device=device)

    print(f"Exporting to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=13,
        input_names=["board_state"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board_state": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    # Quick sanity check
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Export successful! File size: {file_size_mb:.1f} MB")
    print(f"   Input:  board_state (batch, 18, 8, 8)")
    print(f"   Output: policy (batch, 4096), value (batch, 1)")


def main():
    parser = argparse.ArgumentParser(description="Export ChessNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/baseline/chessnet_epoch9.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/baseline/chessnet.onnx"),
        help="Output .onnx path",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    export_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

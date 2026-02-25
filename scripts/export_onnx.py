#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import onnx
    from config import ModelConfig
    from src.model import ChessNet, build_model
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("Ensure you are running this from the project root or have dependencies installed.")
    sys.exit(1)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> None:
    """Safely loads weights from a .pt file into the model."""
    logger.info(f"Loading weights from: {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load on CPU to avoid CUDA OOM during simple export tasks
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Handle both full checkpoint dicts and raw state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strict=True ensures exact architecture match
    model.load_state_dict(state_dict, strict=True)


def export_to_onnx(model: torch.nn.Module, output_path: Path, device: torch.device) -> None:
    """Exports the PyTorch model to ONNX with dynamic batching support."""
    model.eval()

    # Create dummy input: 1 batch, 18 channels, 8x8 board
    # The values don't matter, only the shape and type
    dummy_input = torch.randn(1, 18, 8, 8, device=device, requires_grad=False)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting ONNX graph to: {output_path}")

    # Dynamic axes are critical for the Rust engine to send variable batch sizes
    dynamic_axes = {
        "board_state": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,  # Opset 14 is stable and widely supported
            do_constant_folding=True,
            input_names=["board_state"],
            output_names=["policy", "value"],
            dynamic_axes=dynamic_axes,
        )
        logger.info("Graph export completed.")
    except Exception as e:
        logger.error(f"Torch ONNX export failed: {e}")
        raise


def verify_onnx(output_path: Path) -> None:
    """Performs a sanity check on the exported ONNX model structure."""
    logger.info("Verifying ONNX model integrity...")
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Log file size for sanity check
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Verification successful! Model size: {size_mb:.2f} MB")
        
    except onnx.checker.ValidationError as e:
        logger.error(f"ONNX validation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export ChessNet PyTorch checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        required=True,
        help="Path to the source .pt checkpoint file"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=None,
        help="Path for the output .onnx file (defaults to same name as input)"
    )

    args = parser.parse_args()

    # Default output path handling
    if args.output is None:
        args.output = args.checkpoint.with_suffix(".onnx")

    device = torch.device("cpu")

    try:
        # 1. Initialize Model structure
        config = ModelConfig()
        model = build_model(config)
        model.to(device)

        # 2. Load Weights
        load_checkpoint(args.checkpoint, model, device)

        # 3. Export
        export_to_onnx(model, args.output, device)

        # 4. Verify
        verify_onnx(args.output)

    except Exception as e:
        logger.exception("Export process failed unexpectedly")
        sys.exit(1)


if __name__ == "__main__":
    main()
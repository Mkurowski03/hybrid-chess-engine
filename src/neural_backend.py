"""
Neural Inference Backends.

Provides an abstraction layer for switching between:
- PyTorch Native (Flexible, good for debugging)
- ONNX Runtime (High-performance, TensorRT/CUDA optimized)
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)


class NeuralBackend(ABC):
    """Abstract interface for ChessNet inference."""

    @abstractmethod
    def predict(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched inference.
        
        Args:
            states: (B, 18, 8, 8) float32 numpy array.
            
        Returns:
            policy: (B, 4096) float32 probabilities/logits.
            value: (B,) float32 scalar values [-1, 1].
        """
        pass


class PyTorchBackend(NeuralBackend):
    """
    Standard PyTorch inference.
    Supports FP16 (Half Precision) automatically on CUDA.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.use_half = (device.type == "cuda")

    def predict(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Zero-copy conversion if possible
        tensor = torch.from_numpy(states).to(self.device)
        
        if self.use_half:
            tensor = tensor.half()

        # InferenceMode is slightly faster than no_grad
        with torch.inference_mode():
            policy_logits, values = self.model(tensor)

        # Move back to CPU numpy
        # We perform float conversion here to ensure compatibility with Rust/Python glue code
        policy_np = policy_logits.float().cpu().numpy()
        value_np = values.float().cpu().numpy().flatten()
        
        return policy_np, value_np


class ONNXBackend(NeuralBackend):
    """
    High-Performance ONNX Runtime Backend.
    Prioritizes TensorRT > CUDA > CPU.
    """

    def __init__(self, model_path: Union[str, Path]):
        try:
            import onnxruntime as ort
        except ImportError:
            logger.critical("onnxruntime not installed. Run: pip install onnxruntime-gpu")
            sys.exit(1)

        model_path = str(model_path)
        available = ort.get_available_providers()
        
        # Priority list for Execution Providers
        # TensorRT is best for RTX 3070 Ti, followed by standard CUDA
        priorities = [
            'TensorrtExecutionProvider', 
            'CUDAExecutionProvider', 
            'CPUExecutionProvider'
        ]
        
        # Filter priorities based on availability
        providers = [p for p in priorities if p in available]
        
        logger.info(f"Initializing ONNX Session: {model_path}")
        logger.info(f"Target Providers: {providers}")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            logger.warning(f"Failed to load with requested providers: {e}")
            logger.warning("Falling back to CPUExecutionProvider.")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        logger.info(f"Active Provider: {self.session.get_providers()[0]}")

        # Introspect Input/Output names
        self.input_name = self.session.get_inputs()[0].name
        
        # We expect two outputs: Policy and Value
        # We blindly trust the export order: [Policy, Value]
        # Use export_onnx.py to ensure this order is fixed.
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ORT is strict about types and memory layout
        if states.dtype != np.float32:
            states = states.astype(np.float32)
            
        if not states.flags['C_CONTIGUOUS']:
            states = np.ascontiguousarray(states)

        # Run Inference
        # returns list of [policy_array, value_array]
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: states}
        )

        policy = outputs[0]
        value = outputs[1].flatten()

        return policy, value
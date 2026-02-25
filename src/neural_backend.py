"""Neural inference backends for ChessNet-3070.

Provides a unified ``NeuralBackend`` interface with two implementations:
  - ``PyTorchBackend``: raw PyTorch FP16 inference (legacy)
  - ``ONNXBackend``: ONNX Runtime with TensorRT/CUDA acceleration
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class NeuralBackend(ABC):
    """Abstract base class for neural inference backends."""

    @abstractmethod
    def predict(self, states_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of board states.

        Parameters
        ----------
        states_np : ndarray of shape ``(B, 18, 8, 8)``, dtype float32

        Returns
        -------
        policy : ndarray of shape ``(B, 4096)``, dtype float32
        value  : ndarray of shape ``(B,)``, dtype float32
        """
        ...


class PyTorchBackend(NeuralBackend):
    """Legacy PyTorch FP16 inference backend."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict(self, states_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor = torch.from_numpy(states_np).to(self.device)
        if self.device.type == "cuda":
            input_tensor = input_tensor.half()

        with torch.inference_mode():
            policy, value = self.model(input_tensor)

        policy_np = policy.float().cpu().numpy()  # (B, 4096)
        value_np = value.float().cpu().numpy().flatten()  # (B,)
        return policy_np, value_np


class ONNXBackend(NeuralBackend):
    """ONNX Runtime inference backend with TensorRT/CUDA acceleration."""

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        providers = []
        available = ort.get_available_providers()

        # Prefer TensorRT > CUDA > CPU
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        logging.info(f"[ONNX] Loading model: {onnx_path}")
        logging.info(f"[ONNX] Available providers: {available}")
        logging.info(f"[ONNX] Requested providers: {providers}")

        try:
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            # TensorRT EP may fail if TRT libraries are not installed
            fallback = [p for p in providers if p != "TensorrtExecutionProvider"]
            logging.warning(
                f"[ONNX] Failed with {providers}, falling back to {fallback}"
            )
            self.session = ort.InferenceSession(str(onnx_path), providers=fallback)

        active = self.session.get_providers()
        logging.info(f"[ONNX] Active providers: {active}")

        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logging.info(
            f"[ONNX] Input: {self.input_name}, Outputs: {self.output_names}"
        )

    def predict(self, states_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ONNX Runtime expects float32 contiguous arrays
        if states_np.dtype != np.float32:
            states_np = states_np.astype(np.float32)
        if not states_np.flags["C_CONTIGUOUS"]:
            states_np = np.ascontiguousarray(states_np)

        outputs = self.session.run(
            self.output_names, {self.input_name: states_np}
        )

        policy_np = outputs[0]   # (B, 4096)
        value_np = outputs[1].flatten()  # (B,)
        return policy_np, value_np

"""
Compact Dual-Headed ResNet for ChessNet-3070.

Architecture:
- Input: (B, 18, 8, 8) Board Tensor
- Backbone: Standard ResNet V1 (Conv -> BN -> ReLU)
- Heads:
    - Policy: Conv1x1 -> BN -> ReLU -> FC -> Logits (4096)
    - Value:  Conv1x1 -> BN -> ReLU -> FC -> ReLU -> FC -> Tanh (-1, 1)

Designed for RTX 3070 Ti (8GB VRAM) with mixed precision support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class ResidualBlock(nn.Module):
    """
    Standard ResNet 'BasicBlock' implementation.
    Structure: Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ Input) -> ReLU
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    Dual-Headed ResNet for Chess Policy and Value prediction.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.cfg = cfg
        num_filters = cfg.num_filters

        # ---------------------------------------------------------------------
        # Backbone (Feature Extractor)
        # ---------------------------------------------------------------------
        self.input_conv = nn.Conv2d(cfg.in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Stack residual blocks
        blocks = [ResidualBlock(num_filters) for _ in range(cfg.num_residual_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # ---------------------------------------------------------------------
        # Policy Head (Move Probabilities)
        # Output: 4096 logits (64 from_sq * 64 to_sq)
        # ---------------------------------------------------------------------
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        
        # Flattened size: 2 channels * 8 * 8 = 128
        self.policy_fc = nn.Linear(128, cfg.policy_output_dim)

        # ---------------------------------------------------------------------
        # Value Head (Win Probability)
        # Output: Scalar [-1, 1]
        # ---------------------------------------------------------------------
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        
        # Flattened size: 1 channel * 8 * 8 = 64
        self.value_fc1 = nn.Linear(64, cfg.value_hidden_dim)
        self.value_fc2 = nn.Linear(cfg.value_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, 18, 8, 8)
            
        Returns:
            policy_logits: (Batch, 4096) - Raw unnormalized logits
            value: (Batch, 1) - Tanh activated scalar [-1, 1]
        """
        # Backbone
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        x = self.res_blocks(x)

        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = torch.flatten(p, start_dim=1) # (B, 128)
        p = self.policy_fc(p)             # (B, 4096)

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = torch.flatten(v, start_dim=1) # (B, 64)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)                 # (B, 1)

        return p, v


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_model(cfg: ModelConfig | None = None) -> ChessNet:
    """Factory function to instantiate the model."""
    return ChessNet(cfg)


def count_params(model: nn.Module) -> int:
    """Utility to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
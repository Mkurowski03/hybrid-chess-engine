"""Compact Dual-Headed ResNet for chess (policy + value).

Designed to fit in 8 GB VRAM with batch-size ≥ 512 on an RTX 3070 Ti.
The forward method is ``torch.compile``-safe (no data-dependent control flow).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class ResidualBlock(nn.Module):
    """Pre-activation-style residual block with two 3×3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ChessNet(nn.Module):
    """Dual-headed CNN: shared ResNet backbone → policy head + value head."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        nf = cfg.num_filters

        # ---------- backbone ----------
        self.input_conv = nn.Conv2d(cfg.in_channels, nf, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(nf)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(nf) for _ in range(cfg.num_residual_blocks)]
        )

        # ---------- policy head ----------
        self.policy_conv = nn.Conv2d(nf, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, cfg.policy_output_dim)

        # ---------- value head ----------
        self.value_conv = nn.Conv2d(nf, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, cfg.value_hidden_dim)
        self.value_fc2 = nn.Linear(cfg.value_hidden_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, 18, 8, 8)``

        Returns
        -------
        policy : Tensor of shape ``(B, 4096)`` — raw logits.
        value  : Tensor of shape ``(B, 1)`` in ``[-1, 1]``.
        """
        # Backbone
        s = F.relu(self.input_bn(self.input_conv(x)))
        s = self.res_blocks(s)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


# --------------- helpers ---------------


def build_model(cfg: ModelConfig | None = None) -> ChessNet:
    """Build a ``ChessNet`` model from ``cfg``."""
    return ChessNet(cfg)


def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

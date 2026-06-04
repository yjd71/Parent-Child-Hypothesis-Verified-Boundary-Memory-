from __future__ import annotations

import torch.nn as nn


def make_gate_head(value_dim: int) -> nn.Module:
    gate_in_channels = int(value_dim) * 3 + 4
    gate_hidden = max(16, int(value_dim) * 2)
    return nn.Sequential(
        nn.Conv2d(gate_in_channels, gate_hidden, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
    )

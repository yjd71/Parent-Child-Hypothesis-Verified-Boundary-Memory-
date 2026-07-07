"""Parent-child hypothesis cross-attention (PC-HCA)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..common.utils import finite_or_zero, masked_softmax, normalize


class PCHCA(nn.Module):
    """Multi-head hypothesis attention with prior-bias logits.

    Args:
        q_state: ``[M,512]``
        h_tokens: ``[M,K,512]``
        prior_bias: ``[M,K]``
        route_context: ``[M,512]``
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, head_dim: int = 64, tau: float = 0.10) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner = self.num_heads * self.head_dim
        self.tau = float(tau)
        self.q_proj = nn.Linear(dim, self.inner)
        self.k_proj = nn.Linear(dim, self.inner)
        self.v_proj = nn.Linear(dim, self.inner)
        self.out_proj = nn.Linear(self.inner, dim)
        self.mod = nn.Sequential(
            nn.Linear(dim, dim * 3),
        )
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, q_state: torch.Tensor, h_tokens: torch.Tensor, prior_bias: torch.Tensor, route_context: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if q_state.numel() == 0:
            return q_state, q_state.new_empty(q_state.size(0), h_tokens.size(1) if h_tokens.dim() > 1 else 0)
        m, k, _ = h_tokens.shape
        q = self.q_proj(q_state).view(m, self.num_heads, self.head_dim)
        key = self.k_proj(h_tokens).view(m, k, self.num_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(h_tokens).view(m, k, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits / max(self.tau, 1e-6) + prior_bias.unsqueeze(1)
        attn = masked_softmax(logits, mask.unsqueeze(1) if mask is not None else None, dim=-1)
        out = (attn.unsqueeze(-1) * val).sum(dim=2).reshape(m, self.inner)
        out = self.out_proj(out)
        shift, scale, gate = self.mod(route_context).chunk(3, dim=-1)
        mod_out = out * (1.0 + scale.tanh()) + shift
        q_new = normalize(q_state + torch.sigmoid(gate) * mod_out, dim=-1)
        return finite_or_zero(q_new), attn.mean(dim=1)

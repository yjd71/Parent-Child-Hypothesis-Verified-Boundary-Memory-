"""p3 parent hypothesis retrieval from routed PC-HBM memory."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .utils import EPS, entropy_from_probs, gather_tokens, masked_softmax, normalize


class ParentRetriever(nn.Module):
    """Retrieve top-K parent hypotheses for p3 boundary tokens."""

    def __init__(self, p3_ch: int, dim: int = 512, topk: int = 64, tau: float = 0.07) -> None:
        super().__init__()
        self.dim = int(dim)
        self.topk = int(topk)
        self.tau = float(tau)
        self.proj_parent_q = nn.Conv2d(int(p3_ch), self.dim, kernel_size=1, bias=False)

    def encode_q_map(self, p3: torch.Tensor) -> torch.Tensor:
        return normalize(self.proj_parent_q(p3), dim=1)

    def forward(
        self,
        p3: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        parent_subbank: Dict[str, Any],
    ) -> Dict[str, Any]:
        q_map = self.encode_q_map(p3)
        q3 = gather_tokens(q_map, batch_ids, flat_indices)
        m = int(q3.size(0))
        keys = parent_subbank["p3_keys"].to(device=p3.device, dtype=p3.dtype)
        values = parent_subbank["p3_values"].to(device=p3.device, dtype=p3.dtype)
        geo = parent_subbank["p3_geometry"].to(device=p3.device, dtype=p3.dtype)
        child_ptr = parent_subbank["child_ptr"].to(device=p3.device)
        meta = parent_subbank.get("parent_meta", [])
        if m == 0 or keys.size(0) == 0:
            return self._empty(q3, self.topk)
        sim = normalize(q3, dim=-1) @ normalize(keys, dim=-1).transpose(0, 1)
        k = min(self.topk, keys.size(0))
        score, idx = torch.topk(sim, k=k, dim=1)
        if k < self.topk:
            pad = self.topk - k
            idx = torch.cat([idx, idx[:, -1:].expand(m, pad)], dim=1)
            score = torch.cat([score, score[:, -1:].expand(m, pad)], dim=1)
        top_keys = keys.index_select(0, idx.reshape(-1)).reshape(m, self.topk, self.dim)
        top_values = values.index_select(0, idx.reshape(-1)).reshape(m, self.topk, values.size(1))
        top_geo = geo.index_select(0, idx.reshape(-1)).reshape(m, self.topk, geo.size(1))
        top_child_ptrs = child_ptr.index_select(0, idx.reshape(-1)).reshape(m, self.topk)
        attn = torch.softmax(score / max(self.tau, EPS), dim=1)
        p3_group = (attn.unsqueeze(-1) * top_values[..., :4]).sum(dim=1)
        p3_group = p3_group / p3_group.sum(dim=1, keepdim=True).clamp_min(EPS)
        fg = (attn * top_values[..., 5]).sum(dim=1, keepdim=True)
        bg = (attn * top_values[..., 4]).sum(dim=1, keepdim=True)
        parent_entropy = entropy_from_probs(attn, dim=1)
        meta_top = []
        for row in idx.detach().cpu().tolist():
            meta_top.append([meta[int(i)] if int(i) < len(meta) else {} for i in row])
        return {
            "q3": q3,
            "q3_map": q_map,
            "top_parent_keys": top_keys,
            "top_parent_values": top_values,
            "top_parent_geo": top_geo,
            "top_child_ptrs": top_child_ptrs,
            "top_parent_scores": score,
            "A_parent": attn,
            "P3_group": p3_group,
            "S_fg_parent": fg,
            "S_bg_parent": bg,
            "M_parent": fg - bg,
            "parent_entropy": parent_entropy,
            "top_parent_meta": meta_top,
        }

    def _empty(self, q3: torch.Tensor, k: int) -> Dict[str, Any]:
        m = int(q3.size(0))
        return {
            "q3": q3,
            "q3_map": None,
            "top_parent_keys": q3.new_empty(m, k, self.dim),
            "top_parent_values": q3.new_empty(m, k, 8),
            "top_parent_geo": q3.new_empty(m, k, 6),
            "top_child_ptrs": torch.empty(m, k, device=q3.device, dtype=torch.long),
            "top_parent_scores": q3.new_empty(m, k),
            "A_parent": q3.new_empty(m, k),
            "P3_group": q3.new_empty(m, 4),
            "S_fg_parent": q3.new_empty(m, 1),
            "S_bg_parent": q3.new_empty(m, 1),
            "M_parent": q3.new_empty(m, 1),
            "parent_entropy": q3.new_empty(m),
            "top_parent_meta": [],
        }

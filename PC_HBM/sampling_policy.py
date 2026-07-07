"""Availability-aware sampling policy for labelled PC-HBM memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegionSamplingRule:
    max_count: int
    min_count: int
    ratio: float


DEFAULT_REGION_SAMPLING = {
    "fg_core": RegionSamplingRule(max_count=128, min_count=16, ratio=0.20),
    "fg_boundary": RegionSamplingRule(max_count=384, min_count=32, ratio=0.50),
    "bg_near": RegionSamplingRule(max_count=384, min_count=32, ratio=0.50),
    "bg_far": RegionSamplingRule(max_count=128, min_count=16, ratio=0.20),
}


def sample_region_indices(mask: torch.Tensor, score: torch.Tensor | None, region: str) -> torch.Tensor:
    """Sample available flat indices for one region mask.

    Args:
        mask: ``[H,W]`` boolean tensor.
        score: optional ``[H,W]`` reliability score.
        region: one of fg_core/fg_boundary/bg_near/bg_far.
    """

    rule = DEFAULT_REGION_SAMPLING[region]
    flat = mask.flatten().bool().nonzero(as_tuple=False).flatten()
    n = int(flat.numel())
    if n == 0:
        return flat
    k = min(rule.max_count, max(min(n, rule.min_count), int(round(n * rule.ratio))))
    if score is None:
        return flat[:k]
    rel = score.flatten().index_select(0, flat)
    _, order = torch.topk(rel, k=min(k, n), dim=0)
    return flat.index_select(0, order)

from __future__ import annotations

import torch


def sample_indices(token_indices: torch.Tensor, sample_count: int) -> torch.Tensor:
    sample_count = int(sample_count)
    if sample_count <= 0 or token_indices.numel() <= sample_count:
        return token_indices
    perm = torch.randperm(token_indices.numel(), device=token_indices.device)[:sample_count]
    return token_indices.index_select(0, perm)

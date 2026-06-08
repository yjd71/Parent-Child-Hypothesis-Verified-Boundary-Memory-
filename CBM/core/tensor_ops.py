from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def normalize_distribution(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp_min(eps)
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def entropy_uncertainty(evidence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = normalize_distribution(evidence, dim=1, eps=eps)
    entropy = -(probs * probs.log()).sum(dim=1)
    denom = torch.log(torch.tensor(float(evidence.size(1)), device=evidence.device, dtype=evidence.dtype))
    return (entropy / denom.clamp_min(eps)).clamp(0.0, 1.0)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = normalize_distribution(p, dim=1, eps=eps)
    q = normalize_distribution(q, dim=1, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m.clamp_min(eps)).log()).sum(dim=1)
    kl_qm = (q * (q / m.clamp_min(eps)).log()).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def unfold_neighbors(x: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor:
    bsz, channels, height, width = x.shape
    kernel_elems = kernel_size * kernel_size
    unfolded = F.unfold(x, kernel_size=kernel_size, padding=padding)
    return unfolded.reshape(bsz, channels, kernel_elems, height * width)


def scatter_tokens(
    out_map: torch.Tensor,
    batch_indices: torch.Tensor,
    spatial_indices: torch.Tensor,
    values: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    flat = out_map.flatten(2).transpose(1, 2).contiguous()
    flat[batch_indices, spatial_indices] = values
    return flat.transpose(1, 2).reshape(out_map.size(0), out_map.size(1), height, width)

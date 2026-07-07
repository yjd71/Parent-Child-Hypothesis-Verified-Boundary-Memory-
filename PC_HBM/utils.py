"""Tensor utilities shared by PC-HBM modules.

Unless noted otherwise tensors use BCHW image maps and token tensors use
``[M, C]`` or ``[M, K, C]``.  Helpers are batch-safe and avoid NaN/Inf in empty
or masked cases.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-6
REGION_NAMES = ("fg_core", "fg_boundary", "bg_near", "bg_far")
REGION_TO_ID = {name: idx for idx, name in enumerate(REGION_NAMES)}


def finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with zeros without changing shape or dtype."""

    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Safe L2 normalization."""

    return F.normalize(finite_or_zero(x), dim=dim, eps=eps)


def normalize_prob(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Clamp and normalize probability-like tensors."""

    x = finite_or_zero(x).clamp_min(eps)
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def entropy_from_probs(p: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Normalized entropy in [0, 1] over ``dim``."""

    p = normalize_prob(p, dim=dim, eps=eps)
    ent = -(p * p.clamp_min(eps).log()).sum(dim=dim)
    denom = math.log(max(2, p.size(dim)))
    return (ent / max(denom, eps)).clamp(0.0, 1.0)


def js_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Jensen-Shannon divergence for probability tensors."""

    p = normalize_prob(p, dim=dim, eps=eps)
    q = normalize_prob(q, dim=dim, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m.clamp_min(eps)).clamp_min(eps).log()).sum(dim=dim)
    kl_qm = (q * (q / m.clamp_min(eps)).clamp_min(eps).log()).sum(dim=dim)
    return 0.5 * (kl_pm + kl_qm)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    """Softmax with dtype-safe negative fill for masked positions."""

    logits = finite_or_zero(logits)
    if mask is not None:
        fill = torch.tensor(-1.0e4, device=logits.device, dtype=logits.dtype)
        logits = logits.masked_fill(~mask.bool(), fill)
    probs = torch.softmax(logits, dim=dim)
    if mask is not None:
        probs = probs * mask.to(dtype=probs.dtype)
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
    return finite_or_zero(probs)


def safe_topk(logits: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k that tolerates empty last dimension by returning empty tensors."""

    if logits.size(dim) == 0 or k <= 0:
        out_shape = list(logits.shape)
        out_shape[dim] = 0
        return logits.new_empty(out_shape), torch.empty(out_shape, device=logits.device, dtype=torch.long)
    return torch.topk(logits, k=min(k, logits.size(dim)), dim=dim)


def gradient_strength(prob: torch.Tensor) -> torch.Tensor:
    """Return Sobel-like gradient magnitude for ``[B,1,H,W]`` probabilities."""

    dx = F.pad(prob[..., :, 1:] - prob[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(prob[..., 1:, :] - prob[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + EPS)


def morph_boundary(prob: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological boundary estimate from a probability map."""

    pad = kernel_size // 2
    dil = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_features_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Build morph/uncertainty/gradient/entropy/prob channels from logits."""

    prob = torch.sigmoid(logits)
    morph = morph_boundary(prob)
    unc = 4.0 * prob * (1.0 - prob)
    grad = gradient_strength(prob)
    ent = -(prob.clamp_min(EPS).log() * prob + (1.0 - prob).clamp_min(EPS).log() * (1.0 - prob))
    ent = ent / math.log(2.0)
    return torch.cat([morph, unc.clamp(0.0, 1.0), grad, ent.clamp(0.0, 1.0), prob], dim=1)


def token_indices_from_score(
    score: torch.Tensor,
    top_ratio: float = 0.25,
    threshold: float | None = None,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select variable-count boundary tokens per image from ``[B,1,H,W]`` score.

    Returns ``batch_ids [M]``, ``flat_indices [M]`` and ``token_scores [M]``.
    """

    if score.dim() != 4 or score.size(1) != 1:
        raise ValueError(f"score must be [B,1,H,W], got {tuple(score.shape)}")
    bsz, _, height, width = score.shape
    flat = finite_or_zero(score).flatten(2).squeeze(1)
    batch_ids = []
    flat_ids = []
    vals = []
    total = height * width
    default_k = max(min_tokens, int(round(total * float(top_ratio))))
    if max_tokens is not None:
        default_k = min(default_k, int(max_tokens))
    default_k = min(default_k, total)
    for b in range(bsz):
        row = flat[b]
        if threshold is not None:
            keep = (row >= float(threshold)).nonzero(as_tuple=False).flatten()
            if keep.numel() < min_tokens:
                _, keep = row.topk(k=min(default_k, total), dim=0)
            elif max_tokens is not None and keep.numel() > int(max_tokens):
                local_vals = row.index_select(0, keep)
                _, order = local_vals.topk(k=int(max_tokens), dim=0)
                keep = keep.index_select(0, order)
        else:
            _, keep = row.topk(k=default_k, dim=0)
        if keep.numel() == 0:
            keep = torch.argmax(row).view(1)
        batch_ids.append(torch.full((keep.numel(),), b, device=score.device, dtype=torch.long))
        flat_ids.append(keep.to(torch.long))
        vals.append(row.index_select(0, keep))
    return torch.cat(batch_ids), torch.cat(flat_ids), torch.cat(vals)


def gather_tokens(map_tensor: torch.Tensor, batch_ids: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[M,C]`` tokens from ``[B,C,H,W]`` with batch-aware indices."""

    if batch_ids.numel() == 0:
        return map_tensor.new_empty(0, map_tensor.size(1))
    flat = map_tensor.flatten(2).transpose(1, 2).contiguous()
    return flat[batch_ids.long(), flat_indices.long()]


def scatter_tokens(
    shape: Iterable[int],
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    values: torch.Tensor,
    reduce: str = "replace",
) -> torch.Tensor:
    """Scatter ``[M,C]`` tokens into a BCHW map.

    ``shape`` is ``(B,C,H,W)``.  ``reduce='add'`` uses indexed addition;
    otherwise assignment is used.
    """

    bsz, channels, height, width = [int(v) for v in shape]
    out = values.new_zeros(bsz, channels, height * width)
    if values.numel() == 0:
        return out.view(bsz, channels, height, width)
    vals = values.transpose(0, 1).contiguous()
    if reduce == "add":
        for b in range(bsz):
            keep = batch_ids == b
            if keep.any():
                out[b].index_add_(1, flat_indices[keep].long(), vals[:, keep])
    else:
        out[batch_ids.long(), :, flat_indices.long()] = values
    return out.view(bsz, channels, height, width)


def add_tokens_to_map(base: torch.Tensor, batch_ids: torch.Tensor, flat_indices: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Return base with batch-safe token deltas added at spatial positions."""

    if delta.numel() == 0:
        return base
    out = base.clone()
    flat = out.flatten(2).transpose(1, 2).contiguous()
    flat[batch_ids.long(), flat_indices.long()] = flat[batch_ids.long(), flat_indices.long()] + delta
    return flat.transpose(1, 2).reshape_as(base)


def scale_flat_indices(
    flat_indices: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    """Map flat indices from one grid size to another by nearest coordinate."""

    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    y = torch.div(flat_indices.long(), src_w, rounding_mode="floor")
    x = flat_indices.long().remainder(src_w)
    yy = torch.clamp((y.float() * dst_h / max(src_h, 1)).floor().long(), 0, dst_h - 1)
    xx = torch.clamp((x.float() * dst_w / max(src_w, 1)).floor().long(), 0, dst_w - 1)
    return yy * dst_w + xx


def local_window_gather(
    ref_map: torch.Tensor,
    query_batch_ids: torch.Tensor,
    query_flat_indices: torch.Tensor,
    query_hw: Tuple[int, int],
    ref_hw: Tuple[int, int],
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather local ``window x window`` refs for query tokens.

    Returns refs ``[M, window^2, C]`` and boolean mask ``[M, window^2]``.
    """

    m = int(query_flat_indices.numel())
    channels = ref_map.size(1)
    kernel = int(window)
    radius = kernel // 2
    if m == 0:
        return ref_map.new_empty(0, kernel * kernel, channels), torch.empty(0, kernel * kernel, device=ref_map.device, dtype=torch.bool)
    q_h, q_w = int(query_hw[0]), int(query_hw[1])
    r_h, r_w = int(ref_hw[0]), int(ref_hw[1])
    qy = torch.div(query_flat_indices.long(), q_w, rounding_mode="floor")
    qx = query_flat_indices.long().remainder(q_w)
    cy = torch.clamp((qy.float() * r_h / max(q_h, 1)).floor().long(), 0, r_h - 1)
    cx = torch.clamp((qx.float() * r_w / max(q_w, 1)).floor().long(), 0, r_w - 1)
    refs = []
    masks = []
    flat = ref_map.flatten(2).transpose(1, 2).contiguous()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = cy + dy
            xx = cx + dx
            valid = (yy >= 0) & (yy < r_h) & (xx >= 0) & (xx < r_w)
            idx = yy.clamp(0, r_h - 1) * r_w + xx.clamp(0, r_w - 1)
            refs.append(flat[query_batch_ids.long(), idx.long()])
            masks.append(valid)
    return torch.stack(refs, dim=1), torch.stack(masks, dim=1)


def make_normalized_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a grid_sample grid ``[1,H,W,2]`` with align_corners=False semantics."""

    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2.0 / max(height, 1) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2.0 / max(width, 1) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)

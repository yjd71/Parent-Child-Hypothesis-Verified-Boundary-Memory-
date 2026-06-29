from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class SAMInferenceError(RuntimeError):
    """Raised when no valid SAM result can be produced for a sample."""

    def __init__(
        self,
        message: str,
        *,
        epoch=None,
        step=None,
        sample_indices: Optional[Sequence[int]] = None,
        failures: Optional[Sequence[Any]] = None,
    ) -> None:
        self.message = str(message)
        self.epoch = epoch
        self.step = step
        self.sample_indices = list(sample_indices or [])
        self.failures = list(failures or [])
        context = []
        if epoch is not None:
            context.append("epoch={}".format(epoch))
        if step is not None:
            context.append("step={}".format(step))
        if self.sample_indices:
            context.append("samples={}".format(self.sample_indices))
        if self.failures:
            context.append("failures={}".format(self.failures))
        detail = "{} ({})".format(self.message, ", ".join(context)) if context else self.message
        super().__init__(detail)


def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each map over spatial dims.

    Shape:
        x: [..., H, W]
        return: same shape as x, normalized per [...]-map.
    """
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.numel() == 0:
        return x.clone()
    if x.dim() < 2:
        x_min = x.amin()
        x_max = x.amax()
    else:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def image_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute normalized spatial gradient magnitude.

    Shape:
        x: [B, C, H, W], [B, H, W], [C, H, W], or [H, W]
        return: [B, 1, H, W]
    """
    x4 = _as_bchw_image(x)
    if x4.numel() == 0:
        return x4.new_zeros((x4.size(0), 1, x4.size(-2), x4.size(-1)))
    gx = torch.zeros_like(x4)
    gy = torch.zeros_like(x4)
    if x4.size(-1) > 1:
        gx[..., :-1] = (x4[..., 1:] - x4[..., :-1]).abs()
    if x4.size(-2) > 1:
        gy[..., :-1, :] = (x4[..., 1:, :] - x4[..., :-1, :]).abs()
    grad = (gx + gy).mean(dim=1, keepdim=True)
    return normalize_01(grad)


def binary_reliability(p: torch.Tensor) -> torch.Tensor:
    """Reliability for binary probability maps, high near 0 or 1.

    Shape:
        p: any probability tensor in [0, 1]
        return: same shape as p
    """
    if not torch.is_tensor(p):
        raise TypeError("p must be a torch.Tensor")
    return (p - 0.5).abs().mul(2.0).clamp(0.0, 1.0)


def soft_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU over spatial dims with broadcasting.

    Shape:
        pred: [B, K, H, W] or broadcastable map tensor
        target: [B, K, H, W], [B, 1, H, W], or broadcastable
        return: pred/target broadcast shape without H, W, usually [B, K]
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise TypeError("pred and target must be torch.Tensor")
    pred, target = torch.broadcast_tensors(pred.float(), target.to(device=pred.device).float())
    if pred.numel() == 0:
        return pred.new_zeros(pred.shape[:-2])
    intersection = (pred * target).sum(dim=(-2, -1))
    union = (pred + target - pred * target).sum(dim=(-2, -1))
    return intersection / (union + eps)


def soft_boundary_alignment(mask: torch.Tensor, refine_band: torch.Tensor) -> torch.Tensor:
    """Measure how well soft mask boundaries overlap a refinement band.

    Shape:
        mask: [B, K, H, W] or [B, 1, H, W]
        refine_band: [B, 1, H, W] or broadcastable to mask boundary maps
        return: [B, K]
    """
    mask4 = _as_bchw_maps(mask, name="mask").float()
    if mask4.numel() == 0:
        return mask4.new_zeros(mask4.shape[:-2])
    boundary = _spatial_gradient_per_channel(mask4)
    band = resize_like(_as_b1hw(refine_band, name="refine_band").float(), boundary, mode="bilinear")
    return soft_iou(normalize_01(boundary), band.clamp(0.0, 1.0))


def sample_topk_points(score: torch.Tensor, mask: torch.Tensor, k: int) -> List[torch.Tensor]:
    """Sample top-k valid point coordinates from each batch map.

    Shape:
        score: [B, 1, H, W], [B, H, W], or [H, W]
        mask: same spatial shape as score, valid positions > 0
        return: list length B, each tensor [N, 2] in x,y coordinate order
    """
    score4 = _as_b1hw(score, name="score").float()
    mask4 = _as_b1hw(mask, name="mask").to(device=score4.device).bool()
    if score4.shape[-2:] != mask4.shape[-2:]:
        mask4 = resize_like(mask4.float(), score4, mode="nearest").bool()
    if score4.size(0) != mask4.size(0):
        if mask4.size(0) == 1:
            mask4 = mask4.expand(score4.size(0), -1, -1, -1)
        else:
            raise ValueError("score and mask batch sizes must match or mask batch size must be 1")

    num_points = max(0, int(k))
    points: List[torch.Tensor] = []
    _, _, height, width = score4.shape
    for batch_idx in range(score4.size(0)):
        valid = mask4[batch_idx, 0]
        valid_count = int(valid.sum().item())
        if num_points == 0 or valid_count == 0:
            points.append(score4.new_empty((0, 2), dtype=torch.float32))
            continue
        kk = min(num_points, valid_count)
        flat_score = score4[batch_idx, 0].masked_fill(~valid, -torch.inf).flatten()
        top_idx = torch.topk(flat_score, k=kk, dim=0).indices
        ys = torch.div(top_idx, width, rounding_mode="floor")
        xs = top_idx % width
        points.append(torch.stack((xs, ys), dim=-1).to(dtype=torch.float32))
    return points


def merge_pos_neg_points(
    pos_points: Sequence[torch.Tensor] | torch.Tensor,
    neg_points: Sequence[torch.Tensor] | torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Merge positive and negative point lists for SAM-style prompts.

    Shape:
        pos_points: list of [Np, 2] tensors or [B, Np, 2]
        neg_points: list of [Nn, 2] tensors or [B, Nn, 2]
        return: (coords, labels), each list length B; coords [N, 2], labels [N]
    """
    pos_list = _points_to_list(pos_points)
    neg_list = _points_to_list(neg_points)
    batch_size = max(len(pos_list), len(neg_list))
    coords_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch_idx in range(batch_size):
        pos = pos_list[batch_idx] if batch_idx < len(pos_list) else _empty_points_like(neg_list, batch_idx)
        neg = neg_list[batch_idx] if batch_idx < len(neg_list) else _empty_points_like(pos_list, batch_idx)
        device = pos.device if pos.numel() or not neg.numel() else neg.device
        dtype = pos.dtype if pos.numel() else torch.float32
        pos = pos.to(device=device, dtype=dtype).reshape(-1, 2)
        neg = neg.to(device=device, dtype=dtype).reshape(-1, 2)
        coords = torch.cat((pos, neg), dim=0) if pos.numel() or neg.numel() else pos.new_empty((0, 2))
        labels = torch.cat(
            (
                torch.ones(pos.size(0), device=device, dtype=torch.long),
                torch.zeros(neg.size(0), device=device, dtype=torch.long),
            ),
            dim=0,
        )
        coords_list.append(coords)
        labels_list.append(labels)
    return coords_list, labels_list


def resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize x to ref spatial size.

    Shape:
        x: [B, C, H, W], [B, H, W], or [H, W]
        ref: tensor with spatial shape [..., Hr, Wr]
        return: x resized to Hr, Wr, preserving 2D/3D/4D rank where possible
    """
    if not torch.is_tensor(x) or not torch.is_tensor(ref):
        raise TypeError("x and ref must be torch.Tensor")
    if x.dim() < 2 or ref.dim() < 2:
        raise ValueError("x and ref must have at least two spatial dimensions")
    target_size = tuple(ref.shape[-2:])
    if tuple(x.shape[-2:]) == target_size:
        return x

    original_dim = x.dim()
    if original_dim == 2:
        work = x.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        work = x.unsqueeze(1)
    elif original_dim == 4:
        work = x
    else:
        raise ValueError(f"x must have shape [B,C,H,W], [B,H,W], or [H,W], got {tuple(x.shape)}")

    kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    out = F.interpolate(work.float() if not work.is_floating_point() else work, size=target_size, mode=mode, **kwargs)
    if mode == "nearest" and not x.is_floating_point():
        out = out.to(dtype=x.dtype)
    if original_dim == 2:
        return out[0, 0]
    if original_dim == 3:
        return out[:, 0]
    return out


def safe_detach_to_cpu(x):
    """Detach tensors and move them to CPU.

    Shape:
        x: any object; tensor input keeps same shape
        return: CPU tensor for tensor input, otherwise x unchanged
    """
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def compute_connected_component_boxes(
    mask: torch.Tensor,
    min_area: int,
    expand_ratio: float,
) -> List[torch.Tensor]:
    """Compute connected-component boxes without converting CUDA tensors to numpy.

    Shape:
        mask: [B, 1, H, W], [B, H, W], or [H, W]
        return: list length B, each tensor [num_box, 4] as xyxy on mask.device
    """
    mask4 = _as_b1hw(mask, name="mask").bool()
    boxes_per_batch: List[torch.Tensor] = []
    min_area_value = max(0, int(min_area))
    _, _, height, width = mask4.shape
    image_size = (height, width)

    for batch_idx in range(mask4.size(0)):
        current = mask4[batch_idx, 0]
        if current.numel() == 0 or not current.any():
            boxes_per_batch.append(current.new_empty((0, 4), dtype=torch.float32))
            continue

        visited = torch.zeros_like(current, dtype=torch.bool)
        boxes: List[torch.Tensor] = []
        remaining = current & ~visited
        while remaining.any():
            seed_yx = remaining.nonzero(as_tuple=False)[0]
            component = torch.zeros_like(current, dtype=torch.bool)
            frontier = torch.zeros_like(current, dtype=torch.bool)
            frontier[seed_yx[0], seed_yx[1]] = True
            component |= frontier

            while frontier.any():
                expanded = F.max_pool2d(
                    frontier[None, None].float(),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )[0, 0].bool()
                frontier = expanded & current & ~component
                component |= frontier

            visited |= component
            area = int(component.sum().item())
            if area >= min_area_value:
                ys, xs = component.nonzero(as_tuple=True)
                box = torch.stack((xs.min(), ys.min(), xs.max(), ys.max())).to(dtype=torch.float32)
                boxes.append(expand_box_xyxy(box, expand_ratio, image_size).reshape(4))
            remaining = current & ~visited

        if boxes:
            boxes_per_batch.append(torch.stack(boxes, dim=0))
        else:
            boxes_per_batch.append(current.new_empty((0, 4), dtype=torch.float32))
    return boxes_per_batch


def expand_box_xyxy(
    box: torch.Tensor,
    ratio: float,
    image_size: Tuple[int, int] | Sequence[int],
) -> torch.Tensor:
    """Expand xyxy boxes and clamp to image bounds.

    Shape:
        box: [4] or [N, 4] in x1,y1,x2,y2 order
        image_size: (H, W)
        return: same shape as box
    """
    if not torch.is_tensor(box):
        raise TypeError("box must be a torch.Tensor")
    if box.numel() == 0:
        return box.clone().to(dtype=torch.float32)
    original_shape = box.shape
    boxes = box.to(dtype=torch.float32).reshape(-1, 4)
    height, width = int(image_size[0]), int(image_size[1])
    ratio_value = max(0.0, float(ratio))

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    box_w = (x2 - x1).clamp_min(1.0)
    box_h = (y2 - y1).clamp_min(1.0)
    dx = box_w * ratio_value
    dy = box_h * ratio_value

    expanded = torch.stack(
        (
            (x1 - dx).clamp(0, max(width - 1, 0)),
            (y1 - dy).clamp(0, max(height - 1, 0)),
            (x2 + dx).clamp(0, max(width - 1, 0)),
            (y2 + dy).clamp(0, max(height - 1, 0)),
        ),
        dim=1,
    )
    return expanded.reshape(original_shape)


def gather_by_index(masks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather candidate masks by per-batch indices.

    Shape:
        masks: [B, K, ...]
        indices: [B] or [B, N]
        return: [B, ...] for [B] indices, [B, N, ...] for [B, N] indices
    """
    if not torch.is_tensor(masks) or not torch.is_tensor(indices):
        raise TypeError("masks and indices must be torch.Tensor")
    if masks.dim() < 2:
        raise ValueError(f"masks must have shape [B, K, ...], got {tuple(masks.shape)}")
    if indices.dim() not in (1, 2):
        raise ValueError(f"indices must have shape [B] or [B, N], got {tuple(indices.shape)}")
    if indices.size(0) != masks.size(0):
        raise ValueError("indices batch size must match masks batch size")

    tail_shape = masks.shape[2:]
    if masks.size(1) == 0:
        if indices.dim() == 1:
            return masks.new_empty((masks.size(0), *tail_shape))
        return masks.new_empty((masks.size(0), indices.size(1), *tail_shape))

    idx = indices.to(device=masks.device, dtype=torch.long).clamp(0, masks.size(1) - 1)
    if idx.dim() == 1:
        gather_idx = idx.reshape(idx.size(0), 1, *([1] * len(tail_shape))).expand(-1, -1, *tail_shape)
        return masks.gather(dim=1, index=gather_idx).squeeze(1)
    gather_idx = idx.reshape(idx.size(0), idx.size(1), *([1] * len(tail_shape))).expand(-1, -1, *tail_shape)
    return masks.gather(dim=1, index=gather_idx)


def safe_sigmoid_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Absolute difference between sigmoid probabilities.

    Shape:
        a, b: broadcastable tensors
        return: broadcasted shape, values in [0, 1]
    """
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise TypeError("a and b must be torch.Tensor")
    return (torch.sigmoid(a) - torch.sigmoid(b.to(device=a.device))).abs().clamp(0.0, 1.0)


def _as_bchw_image(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        if x.size(0) in (1, 3, 4):
            return x.unsqueeze(0)
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"x must have shape [B,C,H,W], [B,H,W], [C,H,W], or [H,W], got {tuple(x.shape)}")


def _as_b1hw(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.size(1) != 1:
            raise ValueError(f"{name} must have one channel, got shape {tuple(x.shape)}")
        return x
    raise ValueError(f"{name} must have shape [B,1,H,W], [B,H,W], or [H,W], got {tuple(x.shape)}")


def _as_bchw_maps(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"{name} must have shape [B,K,H,W], [B,H,W], or [H,W], got {tuple(x.shape)}")


def _spatial_gradient_per_channel(x: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    if x.size(-1) > 1:
        gx[..., :-1] = (x[..., 1:] - x[..., :-1]).abs()
    if x.size(-2) > 1:
        gy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return gx + gy


def _points_to_list(points: Sequence[torch.Tensor] | torch.Tensor) -> List[torch.Tensor]:
    if torch.is_tensor(points):
        if points.dim() == 2:
            return [points.reshape(-1, 2)]
        if points.dim() == 3:
            return [points[idx].reshape(-1, 2) for idx in range(points.size(0))]
        raise ValueError(f"point tensor must have shape [N,2] or [B,N,2], got {tuple(points.shape)}")
    return [point.reshape(-1, 2) for point in points]


def _empty_points_like(point_lists: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
    if point_lists:
        ref_idx = min(batch_idx, len(point_lists) - 1)
        ref = point_lists[ref_idx]
        return ref.new_empty((0, 2), dtype=ref.dtype)
    return torch.empty((0, 2), dtype=torch.float32)


__all__ = [
    "normalize_01",
    "image_gradient_magnitude",
    "binary_reliability",
    "soft_iou",
    "soft_boundary_alignment",
    "sample_topk_points",
    "merge_pos_neg_points",
    "resize_like",
    "safe_detach_to_cpu",
    "compute_connected_component_boxes",
    "expand_box_xyxy",
    "gather_by_index",
    "safe_sigmoid_diff",
]

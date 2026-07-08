"""PC-HBM training losses and oracle targets."""

from .branch_oracle import branch_errors, oracle_distribution
from .pc_losses import (
    compute_pc_hbm_labeled_loss,
    compute_pc_hbm_unlabeled_loss,
    dice_loss_with_logits,
    iou_loss_with_logits,
    seg_loss,
    structure_aware_confidence,
    zero_like_loss,
)
from .pc_supervision import (
    REGION_BG_FAR,
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    REGION_FG_CORE,
    build_geometry_target,
    build_need_correction_map,
    build_region_label_map,
    gather_by_boundary_indices,
    parent_meta_to_region_ids,
)

__all__ = [
    "branch_errors",
    "compute_pc_hbm_labeled_loss",
    "compute_pc_hbm_unlabeled_loss",
    "dice_loss_with_logits",
    "iou_loss_with_logits",
    "oracle_distribution",
    "REGION_BG_FAR",
    "REGION_BG_NEAR",
    "REGION_FG_BOUNDARY",
    "REGION_FG_CORE",
    "build_geometry_target",
    "build_need_correction_map",
    "build_region_label_map",
    "gather_by_boundary_indices",
    "parent_meta_to_region_ids",
    "seg_loss",
    "structure_aware_confidence",
    "zero_like_loss",
]

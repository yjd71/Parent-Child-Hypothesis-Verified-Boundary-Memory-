from __future__ import annotations

import os


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_paths(directory: str):
    return sorted(
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file_name))
        and os.path.splitext(file_name)[1].lower() in IMAGE_SUFFIXES
    )


def _index_paths_by_stem(paths, label: str):
    indexed = {}
    duplicates = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in indexed:
            duplicates.setdefault(stem, [indexed[stem]]).append(path)
        else:
            indexed[stem] = path
    if duplicates:
        examples = {name: values for name, values in list(duplicates.items())[:10]}
        raise ValueError(f"duplicate {label} stems: {examples}")
    return indexed


def align_evaluation_paths(gt_paths, pred_paths):
    """Align predictions to GT order, rejecting missing or duplicate stems."""
    gt_by_stem = _index_paths_by_stem(gt_paths, "GT")
    pred_by_stem = _index_paths_by_stem(pred_paths, "prediction")
    missing = sorted(set(gt_by_stem) - set(pred_by_stem))
    if missing:
        raise RuntimeError(
            f"missing predictions: count={len(missing)} examples={missing[:20]} "
            f"gt_count={len(gt_by_stem)} pred_count={len(pred_by_stem)}"
        )
    extras = sorted(set(pred_by_stem) - set(gt_by_stem))
    ordered_stems = sorted(gt_by_stem)
    return (
        [gt_by_stem[stem] for stem in ordered_stems],
        [pred_by_stem[stem] for stem in ordered_stems],
        extras,
    )


__all__ = ["IMAGE_SUFFIXES", "list_image_paths", "align_evaluation_paths"]

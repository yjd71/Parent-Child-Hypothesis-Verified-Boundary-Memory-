import numpy as np
import torch
import os
from collections import defaultdict
import cv2
from tqdm import tqdm
try:
    from ..segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    from SAM.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils2 import prepare_image, extract_bboxes_expand, extract_points, extract_mask


def sam_input_prepare(image, pred_masks, image_embeddings=None, resize_transform=None, use_point=True, use_box=True, use_mask=True, add_neg=True, margin=0.0, gamma=1.0, strength=15):
    ori_size = pred_masks.shape[-2:]
    input_dict = {
         'image': image,
         'original_size': ori_size,
         }

    target_size = image.shape[1:]
    expand_list = torch.zeros((len(pred_masks))).to(image.device)
    if use_box:
        bboxes, box_masks, areas, expand_list = extract_bboxes_expand(image_embeddings, pred_masks, margin=margin)
        input_dict['boxes'] = resize_transform.apply_boxes_torch(bboxes, ori_size)

    point_coords, point_labels, gaus_dt = extract_points(pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma)
    if use_point:
        input_dict['point_coords'] = resize_transform.apply_coords_torch(point_coords, ori_size)
        input_dict['point_labels'] = point_labels



    if use_mask:
        input_dict['mask_inputs'] = extract_mask(pred_masks, gaus_dt, target_size, is01=True, strength=strength, device=image.device, expand_list=expand_list)

    return input_dict,point_coords


def sam_refiner(image,
                coarse_masks,
                sam,
                resize_transform=None,
                use_point=True,
                use_box=True,
                use_mask=True,
                add_neg=True,
                iters=5,
                margin=0.0,
                gamma=4.0,
                strength=30,
                use_samhq=False,
                ddp=False,
                is_train=False,
                coarse_threshold=0.5,
                embedding_cache=None):
    """
    SAMRefiner refines coarse masks from an image by generating noise-tolerant prompts for SAM.

    Arguments:
      image_path (str): The image path for the target image.
      coarse_masks (list(array) or array): The coarse masks to be refined.
      sam (Sam): The Sam model.
      resize_transform (list(float)): The resize_transform used in sam. Default: ResizeLongestSide.
      use_point (bool): Whether to use point prompts. Default: True
      use_box (bool): Whether to use box prompts. Default: True
      use_mask (bool): Whether to use mask prompts. Default: True
      add_neg (bool): Whether to use the negative point prompts. Default: True
      iters (int): The number of iterative refinement. Default: 5
      margin (float): The parameter used to control whether to enlarge the box. Default: 0 (not enlarge)
      gamma (float): The parameter used to control the span of Gaussian distribution in mask prompt. Default: 4.0
      gamma (float): The parameter used to control the amplitude of Gaussian distribution in mask prompt. Default: 30
      use_samhq (bool): Whether to use samhq model. Default: False
      coarse_threshold (float): Threshold used when coarse masks are float/probability maps.
    """

    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, axis=0)

    if len(coarse_masks.shape) == 2:
        coarse_masks = coarse_masks[None, ...]
    coarse_masks = torch.as_tensor(coarse_masks, device=sam.device)
    if coarse_masks.dtype == torch.bool:
        coarse_masks = coarse_masks.to(torch.uint8)
    elif coarse_masks.is_floating_point():
        coarse_masks = (coarse_masks > coarse_threshold).to(torch.uint8)
    else:
        coarse_masks = (coarse_masks > 0).to(torch.uint8)

    assert len(coarse_masks.shape) == 3, "coarse mask dim must be (n, h, w), but got {}".format(coarse_masks.shape)

    if resize_transform is None:
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ori_size = image.shape[:2]
    image = [prepare_image(image, resize_transform, sam.device)]

    with torch.no_grad():
        interm_embeddings = None
        if embedding_cache is not None and not use_samhq:
            def _compute_image_embeddings():
                if ddp:
                    input_images = torch.stack([sam.module.preprocess(x) for x in image], dim=0)
                    return sam.module.image_encoder(input_images)  # torch.Size([1, 256, 64, 64])
                input_images = torch.stack([sam.preprocess(x) for x in image], dim=0)
                return sam.image_encoder(input_images)  # torch.Size([1, 256, 64, 64])

            image_embeddings, _ = embedding_cache.get_or_compute(image[0], _compute_image_embeddings, device=sam.device)
        else:
            if ddp:
                input_images = torch.stack([sam.module.preprocess(x) for x in image], dim=0)
                if not use_samhq:
                    image_embeddings = sam.module.image_encoder(input_images)  # torch.Size([1, 256, 64, 64])
                else:
                    image_embeddings, interm_embeddings = sam.module.image_encoder(input_images)
                    interm_embeddings = interm_embeddings[0]  # early layer
            else:
                input_images = torch.stack([sam.preprocess(x) for x in image], dim=0)
                if not use_samhq:
                    image_embeddings = sam.image_encoder(input_images)  # torch.Size([1, 256, 64, 64])
                else:
                    image_embeddings, interm_embeddings = sam.image_encoder(input_images)
                    interm_embeddings = interm_embeddings[0]  # early layer

    pred_mask_list = coarse_masks.to(torch.uint8)

    for _ in range(iters):
        input_dict, point_coords = sam_input_prepare(image[0],
                                                     pred_mask_list,
                                                     image_embeddings,
                                                     resize_transform,
                                                     use_point=use_point,
                                                     use_box=use_box,
                                                     use_mask=use_mask,
                                                     add_neg=add_neg,
                                                     margin=margin,
                                                     gamma=gamma,
                                                     strength=strength)

        sam_input = [input_dict]

        if not is_train:
            with torch.no_grad():
                if ddp:
                    if not use_samhq:
                        sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    else:
                        sam_output = sam.module.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                else:
                    if not use_samhq:
                        sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    else:
                        sam_output = sam.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
        else:
            if ddp:
                sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
            else:
                sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])

        sam_masks = sam_output['masks']
        sam_masks3 = sam_masks.clone().detach()
        sam_ious = sam_output['iou_predictions']
        sam_masks_logits = sam_output["low_res_logits"]

        if is_train:
            return sam_masks, sam_ious, sam_masks3

        best_masks = []
        best_logits = []
        for sm, si, logits in zip(sam_masks, sam_ious, sam_masks_logits):
            max_idx = torch.argmax(si)
            best_masks.append(sm[max_idx])
            best_logits.append(logits[max_idx])

        sam_masks = torch.stack(best_masks, dim=0)
        sam_masks_logits = torch.stack(best_logits, dim=0)
        pred_mask_list = (sam_masks > 0).to(torch.uint8)

    refined_masks = pred_mask_list.cpu().numpy().astype(np.uint8)
    assert len(refined_masks) == len(coarse_masks)
    return refined_masks, sam_masks_logits

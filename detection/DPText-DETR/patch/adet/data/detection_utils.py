import logging

import numpy as np
import torch
from PIL import Image

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno
from .augmentation import (
    GaussNoise,
    InstanceAwareFixedSizeCrop,
    RandomBlur,
    RandomHueSaturationValue,
    SmartResizeShortestEdge,
)
import random


_INTERP_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_ctrl_pnts_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers

    if "polygons" in annotation:
        polys = transform_ctrl_pnts_annotations(annotation["polygons"], transforms)
        annotation["polygons"] = polys

    return annotation


def transform_ctrl_pnts_annotations(pnts, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    pnts = np.asarray(pnts, dtype="float64").reshape(-1, 2)
    pnts = transforms.apply_coords(pnts).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return pnts


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    if "polygons" in annos[0]:
        polys = [obj.get("polygons", []) for obj in annos]
        instance.polygons = torch.as_tensor(polys, dtype=torch.float32)

    return instance


def _parse_hw(shape_cfg):
    if len(shape_cfg) != 2:
        raise ValueError(f"Expected shape as (h, w), got {shape_cfg}")
    return int(shape_cfg[0]), int(shape_cfg[1])


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    logger = logging.getLogger(__name__)

    train_color_augmentations = []
    if is_train:
        train_color_augmentations = [
            T.RandomContrast(0.5, 1.5),
            T.RandomBrightness(0.5, 1.5),
            T.RandomLighting(random.random() + 0.5),
            RandomHueSaturationValue(hue_shift_limit=40, p=0.5),
            RandomBlur(7, 0.5),
        ]

    interp = _INTERP_MAP.get(cfg.INPUT.INTERP.lower(), Image.BILINEAR)

    augmentation = []
    if cfg.INPUT.FIXED_SIZE.ENABLED:
        if is_train:
            augmentation.extend(train_color_augmentations)
            target_h, target_w = _parse_hw(cfg.INPUT.FIXED_SIZE.TRAIN_SHAPE)
            augmentation.append(
                T.ResizeScale(
                    min_scale=cfg.INPUT.FIXED_SIZE.MIN_SCALE,
                    max_scale=cfg.INPUT.FIXED_SIZE.MAX_SCALE,
                    target_height=target_h,
                    target_width=target_w,
                    interp=interp,
                )
            )
            augmentation.append(
                InstanceAwareFixedSizeCrop(
                    crop_size=(target_h, target_w),
                    crop_instance=cfg.INPUT.FIXED_SIZE.CROP_INSTANCE,
                    pad=cfg.INPUT.FIXED_SIZE.PAD,
                    pad_value=cfg.INPUT.FIXED_SIZE.PAD_VALUE,
                    seg_pad_value=cfg.INPUT.FIXED_SIZE.SEG_PAD_VALUE,
                    max_tries=cfg.INPUT.FIXED_SIZE.CROP_MAX_TRIES,
                )
            )
        else:
            target_h, target_w = _parse_hw(cfg.INPUT.FIXED_SIZE.TEST_SHAPE)
            augmentation.append(
                T.ResizeScale(
                    min_scale=1.0,
                    max_scale=1.0,
                    target_height=target_h,
                    target_width=target_w,
                    interp=interp,
                )
            )
            augmentation.append(
                InstanceAwareFixedSizeCrop(
                    crop_size=(target_h, target_w),
                    crop_instance=False,
                    pad=cfg.INPUT.FIXED_SIZE.PAD,
                    pad_value=cfg.INPUT.FIXED_SIZE.PAD_VALUE,
                    seg_pad_value=cfg.INPUT.FIXED_SIZE.SEG_PAD_VALUE,
                    max_tries=1,
                )
            )
    else:
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        if sample_style == "range":
            assert (
                len(min_size) == 2
            ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

        if cfg.INPUT.SMART_RESIZE.ENABLED:
            augmentation.append(
                SmartResizeShortestEdge(
                    min_size,
                    max_size,
                    sample_style,
                    cfg.INPUT.SMART_RESIZE.FACTOR,
                    cfg.INPUT.SMART_RESIZE.MIN_PIXELS,
                    cfg.INPUT.SMART_RESIZE.MAX_PIXELS,
                    interp=interp,
                )
            )
        else:
            augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style, interp=interp))

        if is_train:
            augmentation.extend(train_color_augmentations)

    if is_train:
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""

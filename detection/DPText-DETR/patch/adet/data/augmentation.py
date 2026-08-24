import random
import numpy as np
import math
from PIL import Image
from fvcore.transforms import transform as T
from detectron2.data.transforms import RandomCrop, StandardAugInput, ResizeTransform
from detectron2.structures import BoxMode
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform
import albumentations as A


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class SmartResizeShortestEdge(Augmentation):
    """ResizeShortestEdge + smart_resize, ensuring target shape is divisible by factor."""

    def __init__(self, short_edge_length, max_size, sample_style, factor, min_pixels, max_pixels, interp=Image.BILINEAR):
        super().__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length,)
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.interp = interp

    def _get_short_edge_target(self):
        if self.sample_style == "range":
            assert len(self.short_edge_length) == 2
            return np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        if self.sample_style == "choice":
            return np.random.choice(self.short_edge_length)
        raise ValueError(f"Unknown sample style: {self.sample_style}")

    def get_transform(self, img):
        h, w = img.shape[:2]
        size = self._get_short_edge_target()
        scale = float(size) / min(h, w)
        newh, neww = size, size
        if h < w:
            neww = scale * w
        else:
            newh = scale * h
        if max(newh, neww) > self.max_size:
            scale = float(self.max_size) / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)

        smart_h, smart_w = smart_resize(
            newh,
            neww,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return ResizeTransform(h, w, smart_h, smart_w, interp=self.interp)


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class InstanceAwareFixedSizeCrop(Augmentation):
    """Fixed-size crop with optional instance-preserving offset sampling.

    If no truncation-free offset is found within a bounded number of random trials,
    it falls back to the offset that truncates the fewest instances among sampled candidates.
    """

    def __init__(
        self,
        crop_size,
        crop_instance=False,
        pad=True,
        pad_value=128.0,
        seg_pad_value=255,
        max_tries=80,
    ):
        super().__init__()
        self._init(locals())
        self.input_args = ("image", "boxes")

    @staticmethod
    def _count_truncated_instances(boxes, x0, y0, crop_w, crop_h):
        x1_crop = x0 + crop_w
        y1_crop = y0 + crop_h
        num_truncated = 0
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            fully_outside = x2 <= x0 or x1 >= x1_crop or y2 <= y0 or y1 >= y1_crop
            fully_inside = x1 >= x0 and x2 <= x1_crop and y1 >= y0 and y2 <= y1_crop
            if not (fully_inside or fully_outside):
                num_truncated += 1
        return num_truncated

    def _sample_crop_offset(self, image_h, image_w, crop_h, crop_w, boxes):
        max_x = max(image_w - crop_w, 0)
        max_y = max(image_h - crop_h, 0)
        if max_x == 0 and max_y == 0:
            return 0, 0

        if boxes is None or len(boxes) == 0 or self.crop_instance:
            return np.random.randint(0, max_x + 1), np.random.randint(0, max_y + 1)

        boxes = np.asarray(boxes, dtype=np.float32)
        best_xy = None
        best_score = None
        # Try random offsets first for speed; zero truncation returns immediately.
        for _ in range(int(self.max_tries)):
            x0 = np.random.randint(0, max_x + 1)
            y0 = np.random.randint(0, max_y + 1)
            truncated = self._count_truncated_instances(boxes, x0, y0, crop_w, crop_h)
            if truncated == 0:
                return x0, y0
            if best_score is None or truncated < best_score:
                best_score = truncated
                best_xy = (x0, y0)

        return best_xy if best_xy is not None else (0, 0)

    def get_transform(self, img, boxes):
        image_h, image_w = img.shape[:2]
        target_h, target_w = int(self.crop_size[0]), int(self.crop_size[1])
        crop_h = min(target_h, image_h)
        crop_w = min(target_w, image_w)

        x0, y0 = self._sample_crop_offset(image_h, image_w, crop_h, crop_w, boxes)
        transforms = [T.CropTransform(x0, y0, crop_w, crop_h, image_w, image_h)]

        if self.pad:
            pad_h = max(target_h - crop_h, 0)
            pad_w = max(target_w - crop_w, 0)
            if pad_h > 0 or pad_w > 0:
                transforms.append(
                    T.PadTransform(
                        0,
                        0,
                        pad_w,
                        pad_h,
                        crop_w,
                        crop_h,
                        self.pad_value,
                        self.seg_pad_value,
                    )
                )

        return T.TransformList(transforms)


class BlurTransform(Transform):
    def __init__(self, kernel_size, p):
        super().__init__()
        blur_aug = A.OneOf([
            A.Blur(blur_limit=kernel_size, p=1),
            A.MotionBlur(blur_limit=kernel_size, p=1)
        ], p=p)
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.blur_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomBlur(Augmentation):
    def __init__(self, kernel_size, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return BlurTransform(self.kernel_size, self.p)


class GaussNoiseTransform(Transform):
    def __init__(self, p):
        super().__init__()
        gauss_noise_aug = A.GaussNoise(p=p)
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.gauss_noise_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class GaussNoise(Augmentation):
    def __init__(self, p):
        """
        Args:
            p (float): probability
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return GaussNoiseTransform(self.p)


class HueSaturationValueTransform(Transform):
    def __init__(self, hue_shift_limit, p):
        super().__init__()
        hue_saturation_aug = A.HueSaturationValue(hue_shift_limit=hue_shift_limit, p=p)
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return self.hue_saturation_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomHueSaturationValue(Augmentation):
    """
    Random hue, saturation & value.
    """
    def __init__(self, hue_shift_limit, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return HueSaturationValueTransform(self.hue_shift_limit, self.p)
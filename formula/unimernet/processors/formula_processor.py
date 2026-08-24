import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from unimernet.common.registry import registry
from omegaconf import OmegaConf
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from unimernet.processors.base_processor import BaseProcessor
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import resize
import random

MONKEY_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
MONKEY_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class FormulaImageBaseProcessor(BaseProcessor):

    def __init__(self, image_size, pad_value=0):
        super(FormulaImageBaseProcessor, self).__init__()
        self.input_size = [int(_) for _ in image_size]
        self.pad_value = int(pad_value)
        assert len(self.input_size) == 2

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    @staticmethod
    def crop_margin_numpy(img: np.ndarray) -> np.ndarray:
        """Crop margins of image using NumPy operations"""
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Normalize and threshold
        if gray.max() == gray.min():
            return img

        normalized = (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8)
        binary = 255 * (normalized < 200).astype(np.uint8)

        # Find bounding box
        coords = cv2.findNonZero(binary)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        # Return cropped image
        return img[y:y + h, x:x + w]

    def prepare_input(self, img, random_padding: bool = False):
        """
        Convert PIL Image or numpy array to properly sized and padded image after:
            - crop margins
            - resize while maintaining aspect ratio
            - pad to target size
        """
        if img is None:
            return None

        # Handle numpy array
        elif isinstance(img, np.ndarray):
            try:
                img = self.crop_margin_numpy(img)
            except Exception:
                # might throw an error for broken files
                return None

            if img.shape[0] == 0 or img.shape[1] == 0:
                return None

            # Get current dimensions
            h, w = img.shape[:2]
            target_h, target_w = self.input_size

            # Calculate scale to preserve aspect ratio (equivalent to resize + thumbnail)
            scale = min(target_h / h, target_w / w)

            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize the image while preserving aspect ratio
            resized_img = cv2.resize(img, (new_w, new_h))

            # Calculate padding values using the existing method
            delta_width = target_w - new_w
            delta_height = target_h - new_h

            pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

            # Apply padding (convert PIL padding format to OpenCV format)
            if len(img.shape) == 3:
                padding_color = [self.pad_value] * img.shape[2]
            else:
                padding_color = [self.pad_value]

            padded_img = cv2.copyMakeBorder(
                resized_img,
                pad_height,  # top
                delta_height - pad_height,  # bottom
                pad_width,  # left
                delta_width - pad_width,  # right
                cv2.BORDER_CONSTANT,
                value=padding_color
            )

            return padded_img

        # Handle PIL Image
        elif isinstance(img, Image.Image):
            try:
                img = self.crop_margin(img.convert("RGB"))
            except OSError:
                # might throw an error for broken files
                return None

            if img.height == 0 or img.width == 0:
                return None

            # Resize while preserving aspect ratio
            img = resize(img, min(self.input_size))
            img.thumbnail((self.input_size[1], self.input_size[0]))
            new_w, new_h = img.width, img.height

            # Calculate and apply padding
            padding = self._calculate_padding(new_w, new_h, random_padding)
            return np.array(ImageOps.expand(img, padding, fill=self.pad_value))

        else:
            return None

    def _calculate_padding(self, new_w, new_h, random_padding):
        """Calculate padding values for PIL images"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

        return (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

    def _get_padding_values(self, new_w, new_h, random_padding):
        """Get padding values based on image dimensions and padding strategy"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2

        return pad_width, pad_height



@registry.register_processor("formula_image_train")
class FormulaImageTrainProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=384):
        super().__init__(image_size)

        # Import weather-related augmentations only when initializing this class
        from unimernet.processors.formula_processor_helper.nougat import Bitmap, Dilation, Erosion
        from unimernet.processors.formula_processor_helper.weather import Fog, Frost, Snow, Rain, Shadow

        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=1,
                        ),
                        alb.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=0.5,
                        )],
                    p=.15),
                # alb.InvertImg(p=.15),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.2),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(quality_range=(95, 100), p=0.3),
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        img = self.prepare_input(item, random_padding=True)
        if img is None:
            return img
        return self.transform(image=img)['image'][:1]


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(
            image_size=image_size,
        )


@registry.register_processor("formula_image_multi_scale_train")
class FormulaImageMultiScaleTrainProcessor(FormulaImageTrainProcessor):
    def __init__(self, all_scales):
        for i, scales in enumerate(all_scales):
            all_scales[i] = [int(_) for _ in scales]
        super(FormulaImageMultiScaleTrainProcessor, self).__init__(all_scales[0])
        self.all_scales = all_scales

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        all_scales = cfg.get("all_scales", [[384, 384]])
        return cls(
            all_scales=all_scales
        )

    def reset_scale(self):
        self.input_size = random.choice(self.all_scales)


@registry.register_processor("formula_image_eval")
class FormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size):
        super().__init__(image_size)

        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=image)['image'][:1]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(image_size=image_size)


@registry.register_processor("formula_image_monkey_train")
class MonkeyFormulaImageTrainProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=384):
        super().__init__(image_size=image_size, pad_value=255)

        self.transform = alb.Compose(
            [
                alb.Affine(
                    scale=(0.9, 1.0),
                    translate_percent=0,
                    rotate=(-1, 1),
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.3,
                ),
                alb.GridDistortion(
                    distort_limit=0.08,
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.15,
                ),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.15),
                alb.RandomBrightnessContrast(0.05, (-0.1, 0), True, p=0.15),
                alb.ImageCompression(quality_range=(95, 100), p=0.15),
                alb.Normalize(MONKEY_IMAGE_MEAN, MONKEY_IMAGE_STD),
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item, random_padding=True)
        if image is None:
            return image
        return self.transform(image=image)["image"]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [196, 672])
        return cls(image_size=image_size)


@registry.register_processor("formula_image_monkey_eval")
class MonkeyFormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size):
        super().__init__(image_size=image_size, pad_value=255)
        self.transform = alb.Compose(
            [
                alb.Normalize(MONKEY_IMAGE_MEAN, MONKEY_IMAGE_STD),
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        if image is None:
            return image
        return self.transform(image=image)["image"]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [196, 672])
        return cls(image_size=image_size)


class MonkeyNativeFixedBaseProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=(196, 672), crop_margin=True):
        super().__init__(image_size=image_size, pad_value=255)
        self.crop_margin_enabled = bool(crop_margin)

    def prepare_native_fixed(self, item):
        if isinstance(item, np.ndarray):
            image = Image.fromarray(item.astype(np.uint8)).convert("RGB")
        elif isinstance(item, Image.Image):
            image = item.convert("RGB")
        else:
            return None

        if self.crop_margin_enabled:
            image = self.crop_margin(image)

        return np.array(
            image.resize((self.input_size[1], self.input_size[0]), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )


@registry.register_processor("formula_image_monkey_native_fixed_train")
class MonkeyNativeFixedImageTrainProcessor(MonkeyNativeFixedBaseProcessor):
    def __init__(self, image_size=(196, 672), crop_margin=True):
        super().__init__(image_size=image_size, crop_margin=crop_margin)
        self.transform = alb.Compose(
            [
                alb.Affine(
                    scale=(0.9, 1.0),
                    translate_percent=0,
                    rotate=(-1, 1),
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.3,
                ),
                alb.GridDistortion(
                    distort_limit=0.08,
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.15,
                ),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.15),
                alb.RandomBrightnessContrast(0.05, (-0.1, 0), True, p=0.15),
                alb.ImageCompression(quality_range=(95, 100), p=0.15),
                alb.Normalize(MONKEY_IMAGE_MEAN, MONKEY_IMAGE_STD),
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_native_fixed(item)
        if image is None:
            return image
        return self.transform(image=image)["image"]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            image_size=cfg.get("image_size", [196, 672]),
            crop_margin=cfg.get("crop_margin", True),
        )


@registry.register_processor("formula_image_monkey_native_fixed_aug_train")
class MonkeyNativeFixedAugImageTrainProcessor(MonkeyNativeFixedBaseProcessor):
    def __init__(self, image_size=(196, 672), crop_margin=True):
        super().__init__(image_size=image_size, crop_margin=crop_margin)

        from unimernet.processors.formula_processor_helper.nougat import Bitmap, Dilation, Erosion
        from unimernet.processors.formula_processor_helper.weather import Fog, Frost, Snow, Rain, Shadow

        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=1,
                        ),
                        alb.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=0.5,
                        ),
                    ],
                    p=0.15,
                ),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.2),
                alb.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
                alb.ImageCompression(quality_range=(95, 100), p=0.3),
                alb.Normalize(MONKEY_IMAGE_MEAN, MONKEY_IMAGE_STD),
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_native_fixed(item)
        if image is None:
            return image
        return self.transform(image=image)["image"]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            image_size=cfg.get("image_size", [196, 672]),
            crop_margin=cfg.get("crop_margin", True),
        )


@registry.register_processor("formula_image_monkey_native_fixed_eval")
class MonkeyNativeFixedImageEvalProcessor(MonkeyNativeFixedBaseProcessor):
    def __init__(self, image_size=(196, 672), crop_margin=True):
        super().__init__(image_size=image_size, crop_margin=crop_margin)
        self.transform = alb.Compose(
            [
                alb.Normalize(MONKEY_IMAGE_MEAN, MONKEY_IMAGE_STD),
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_native_fixed(item)
        if image is None:
            return image
        return self.transform(image=image)["image"]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            image_size=cfg.get("image_size", [196, 672]),
            crop_margin=cfg.get("crop_margin", True),
        )


def _normalize_rgb_to_chw(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        array = image.astype(np.float32)
    array = array / 255.0
    mean = np.asarray(MONKEY_IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(MONKEY_IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def _patchify_monkey_native(
    tensor: torch.Tensor,
    patch_size: int = 14,
    temporal_patch_size: int = 1,
    merge_size: int = 2,
):
    channels, height, width = tensor.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Monkey native preprocessing expects dimensions divisible by {patch_size}, got {(height, width)}")
    grid_h = height // patch_size
    grid_w = width // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(f"Monkey native grid must be divisible by merge_size={merge_size}, got {(grid_h, grid_w)}")
    if temporal_patch_size != 1:
        raise NotImplementedError("Only image temporal_patch_size=1 is supported for formula datasets.")

    patches = tensor.reshape(channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.reshape(
        grid_h // merge_size,
        merge_size,
        grid_w // merge_size,
        merge_size,
        channels,
        patch_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    patches = patches.reshape(grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
    return patches, torch.tensor([1, grid_h, grid_w], dtype=torch.long)


class MonkeyNativeDynamicBaseProcessor(FormulaImageBaseProcessor):
    def __init__(
        self,
        max_pixels=196 * 672,
        min_pixels=28 * 28 * 4,
        crop_margin=False,
        patch_size=14,
        temporal_patch_size=1,
        merge_size=2,
    ):
        # image_size is unused for dynamic processing, but the base class keeps
        # shared crop helpers and validates a 2D size.
        super().__init__(image_size=[patch_size * merge_size, patch_size * merge_size], pad_value=255)
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)
        self.crop_margin_enabled = bool(crop_margin)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)

    def prepare_native_dynamic_image(self, item):
        from qwen_vl_utils import fetch_image

        if isinstance(item, np.ndarray):
            image = Image.fromarray(item.astype(np.uint8)).convert("RGB")
        elif isinstance(item, Image.Image):
            image = item.convert("RGB")
        else:
            return None

        if self.crop_margin_enabled:
            image = self.crop_margin(image)

        return fetch_image(
            {
                "image": image,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
            image_patch_size=self.patch_size,
        )

    def estimate_num_tokens_for_size(self, width: int, height: int) -> int:
        from qwen_vl_utils.vision_process import smart_resize

        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return int((resized_height // self.patch_size) * (resized_width // self.patch_size))

    def pack_native_dynamic(self, image):
        tensor = _normalize_rgb_to_chw(image)
        pixel_values, grid_thw = _patchify_monkey_native(
            tensor,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.merge_size,
        )
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


@registry.register_processor("formula_image_monkey_native_dynamic_train")
class MonkeyNativeDynamicImageTrainProcessor(MonkeyNativeDynamicBaseProcessor):
    def __init__(
        self,
        max_pixels=196 * 672,
        min_pixels=28 * 28 * 4,
        crop_margin=False,
        patch_size=14,
        temporal_patch_size=1,
        merge_size=2,
    ):
        super().__init__(
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            crop_margin=crop_margin,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )
        self.transform = alb.Compose(
            [
                alb.Affine(
                    scale=(0.9, 1.0),
                    translate_percent=0,
                    rotate=(-1, 1),
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.3,
                ),
                alb.GridDistortion(
                    distort_limit=0.08,
                    border_mode=0,
                    interpolation=3,
                    fill=255,
                    p=0.15,
                ),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.15),
                alb.RandomBrightnessContrast(0.05, (-0.1, 0), True, p=0.15),
                alb.ImageCompression(quality_range=(95, 100), p=0.15),
            ]
        )

    def __call__(self, item):
        image = self.prepare_native_dynamic_image(item)
        if image is None:
            return image
        image = self.transform(image=np.asarray(image.convert("RGB"), dtype=np.uint8))["image"]
        return self.pack_native_dynamic(image)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            max_pixels=cfg.get("max_pixels", 196 * 672),
            min_pixels=cfg.get("min_pixels", 28 * 28 * 4),
            crop_margin=cfg.get("crop_margin", False),
            patch_size=cfg.get("patch_size", 14),
            temporal_patch_size=cfg.get("temporal_patch_size", 1),
            merge_size=cfg.get("merge_size", 2),
        )


@registry.register_processor("formula_image_monkey_native_dynamic_aug_train")
class MonkeyNativeDynamicAugImageTrainProcessor(MonkeyNativeDynamicBaseProcessor):
    def __init__(
        self,
        max_pixels=1003520,
        min_pixels=28 * 28 * 4,
        crop_margin=True,
        patch_size=14,
        temporal_patch_size=1,
        merge_size=2,
    ):
        super().__init__(
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            crop_margin=crop_margin,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
        )

        from unimernet.processors.formula_processor_helper.nougat import Bitmap, Dilation, Erosion
        from unimernet.processors.formula_processor_helper.weather import Fog, Frost, Snow, Rain, Shadow

        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=1,
                        ),
                        alb.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            fill=255,
                            p=0.5,
                        ),
                    ],
                    p=0.15,
                ),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                alb.GaussNoise(std_range=(0.01, 0.04), p=0.2),
                alb.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
                alb.ImageCompression(quality_range=(95, 100), p=0.3),
            ]
        )

    def __call__(self, item):
        image = self.prepare_native_dynamic_image(item)
        if image is None:
            return image
        image = self.transform(image=np.asarray(image.convert("RGB"), dtype=np.uint8))["image"]
        return self.pack_native_dynamic(image)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            max_pixels=cfg.get("max_pixels", 1003520),
            min_pixels=cfg.get("min_pixels", 28 * 28 * 4),
            crop_margin=cfg.get("crop_margin", True),
            patch_size=cfg.get("patch_size", 14),
            temporal_patch_size=cfg.get("temporal_patch_size", 1),
            merge_size=cfg.get("merge_size", 2),
        )


@registry.register_processor("formula_image_monkey_native_dynamic_eval")
class MonkeyNativeDynamicImageEvalProcessor(MonkeyNativeDynamicBaseProcessor):
    def __call__(self, item):
        image = self.prepare_native_dynamic_image(item)
        if image is None:
            return image
        return self.pack_native_dynamic(image)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            max_pixels=cfg.get("max_pixels", 196 * 672),
            min_pixels=cfg.get("min_pixels", 28 * 28 * 4),
            crop_margin=cfg.get("crop_margin", False),
            patch_size=cfg.get("patch_size", 14),
            temporal_patch_size=cfg.get("temporal_patch_size", 1),
            merge_size=cfg.get("merge_size", 2),
        )

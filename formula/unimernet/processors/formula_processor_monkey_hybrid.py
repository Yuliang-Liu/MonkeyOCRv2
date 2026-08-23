import albumentations as alb
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf

from unimernet.common.registry import registry
from unimernet.processors.formula_processor import (
    FormulaImageBaseProcessor,
    MONKEY_IMAGE_MEAN,
    MONKEY_IMAGE_STD,
)


@registry.register_processor("formula_image_monkey_hybrid_train")
class MonkeyHybridAugImageTrainProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=(196, 672)):
        super().__init__(image_size=image_size, pad_value=255)

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

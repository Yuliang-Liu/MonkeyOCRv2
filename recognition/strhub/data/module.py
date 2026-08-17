import math
import json
from pathlib import Path, PurePath
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms as T

import pytorch_lightning as pl

from .dataset import JsonDataset, LmdbDataset, build_tree_dataset


class SceneTextDataModule(pl.LightningDataModule):
    TEST_BENCHMARK_SUB = ('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80')
    MONKEY_MEAN = (0.48145466, 0.4578275, 0.40821073)
    MONKEY_STD = (0.26862954, 0.26130258, 0.27577711)
    MONKEY_VIT_DIR = str(Path(__file__).resolve().parents[2] / 'pretrained' / 'monkeyocr_vit')
    MONKEY_SIZES = {(56, 280), (112, 448), (56, 896), (224, 224), (448, 112), (896, 56)}

    def __init__(
        self,
        root_dir: str,
        train_dir: str,
        train_gt_json: Optional[str],
        val_dir: str,
        img_size: Sequence[int],
        multi_scales: Optional[Sequence[Sequence[int]]],
        max_label_length: int,
        charset_train: str,
        charset_test: str,
        batch_size: int,
        num_workers: int,
        augment: bool,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_image_dim: int = 0,
        rotation: int = 0,
        collate_fn: Optional[Callable] = None,
        train_gt_jsons: Optional[Sequence[str]] = None,
        train_gt_json_repeats: Optional[Sequence[int]] = None,
        include_additional_val: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.train_gt_json = train_gt_json
        self.val_dir = val_dir
        self.train_gt_jsons = None if train_gt_jsons is None else list(train_gt_jsons)
        self.train_gt_json_repeats = None if train_gt_json_repeats is None else [int(x) for x in train_gt_json_repeats]
        self.include_additional_val = include_additional_val
        self.img_size = tuple(img_size)
        self.multi_scales = None if multi_scales is None else [tuple(scale) for scale in multi_scales]
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None
        uses_monkey_transform = tuple(self.img_size) in self.MONKEY_SIZES or self.multi_scales is not None
        if self.collate_fn is None and uses_monkey_transform:
            self.collate_fn = self.monkey_collate_fn

    @staticmethod
    def get_transform(
        img_size: tuple[int],
        augment: bool = False,
        rotation: int = 0,
        multi_scales: Optional[Sequence[Sequence[int]]] = None,
    ):
        transforms = []
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        if multi_scales:
            return AspectRatioMonkeyOCRTransform(multi_scales, augment=augment, rotation=rotation)
        monkey_sizes = SceneTextDataModule.MONKEY_SIZES
        if tuple(img_size) in monkey_sizes:
            return MonkeyOCRTransform(img_size, augment=augment, rotation=rotation)
        mean, std = (
            (SceneTextDataModule.MONKEY_MEAN, SceneTextDataModule.MONKEY_STD)
            if tuple(img_size) in monkey_sizes
            else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return T.Compose(transforms)

    @staticmethod
    def monkey_collate_fn(batch):
        images, labels = zip(*batch)
        pixel_values = torch.cat([sample['pixel_values'] for sample in images], dim=0)
        image_grid_thw = torch.cat([sample['image_grid_thw'] for sample in images], dim=0)
        return {'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw}, list(labels)

    def _resolve_data_path(self, kind: str, path_value: str) -> PurePath:
        path = Path(path_value)
        if path.is_absolute():
            return path
        direct = Path(self.root_dir, path_value)
        if direct.exists():
            return direct
        return PurePath(self.root_dir, kind, path_value)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment, multi_scales=self.multi_scales)
            if self.train_gt_jsons:
                repeats = self.train_gt_json_repeats or [1] * len(self.train_gt_jsons)
                if len(repeats) != len(self.train_gt_jsons):
                    raise ValueError('train_gt_json_repeats must have the same length as train_gt_jsons')
                datasets = []
                for gt_json, repeat in zip(self.train_gt_jsons, repeats):
                    dataset = JsonDataset(
                        gt_json,
                        self.charset_train,
                        self.max_label_length,
                        self.min_image_dim,
                        self.remove_whitespace,
                        self.normalize_unicode,
                        transform=transform,
                    )
                    datasets.extend([dataset] * max(1, repeat))
                self._train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            elif self.train_gt_json:
                self._train_dataset = JsonDataset(
                    self.train_gt_json,
                    self.charset_train,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
            else:
                root = self._resolve_data_path('train', self.train_dir)
                self._train_dataset = build_tree_dataset(
                    root,
                    self.charset_train,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
        return self._train_dataset

    def _has_lmdb_children(self, root: PurePath | str) -> bool:
        root = Path(root)
        return any((child / 'data.mdb').exists() for child in root.iterdir() if child.is_dir()) if root.exists() else False

    def _load_manifest_datasets(self, root: PurePath | str, charset: str, transform):
        root = Path(root)
        manifest_path = root / 'manifest.json'
        if not manifest_path.exists() or self._has_lmdb_children(root):
            return None
        with manifest_path.open() as f:
            manifest = json.load(f)
        return [
            JsonDataset(
                gt_json,
                charset,
                self.max_label_length,
                self.min_image_dim,
                self.remove_whitespace,
                self.normalize_unicode,
                transform=transform,
            )
            for _, gt_json in manifest.items()
        ]

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size, multi_scales=self.multi_scales)
            root = self._resolve_data_path('', self.val_dir)
            manifest_datasets = self._load_manifest_datasets(root, self.charset_test, transform)
            if manifest_datasets is not None:
                datasets = manifest_datasets
            else:
                val_dataset = build_tree_dataset(
                    root,
                    self.charset_test,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
                datasets = list(val_dataset.datasets) if isinstance(val_dataset, ConcatDataset) else [val_dataset]
            self._val_dataset = ConcatDataset(datasets)
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if isinstance(self.val_dataset, ConcatDataset):
            return [
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                )
                for dataset in self.val_dataset.datasets
            ]
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation, multi_scales=self.multi_scales)
        root = Path(PurePath(self.root_dir, 'test'))
        manifest_path = root / 'manifest.json'
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
            datasets = {
                s: JsonDataset(
                    manifest[s],
                    self.charset_test,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
                for s in subset if s in manifest
            }
        else:
            datasets = {
                s: LmdbDataset(
                    str(root / s),
                    self.charset_test,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
                for s in subset
            }
        return {
            k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)
            for k, v in datasets.items()
        }


class MonkeyOCRTransform:
    def __init__(self, img_size: tuple[int], augment: bool = False, rotation: int = 0):
        self.img_size = tuple(img_size)
        self.augment = augment
        self.rotation = rotation
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(
                SceneTextDataModule.MONKEY_VIT_DIR,
                trust_remote_code=True,
                max_pixels=1003520,
                use_fast=False,
            )
        return self._processor

    def __call__(self, img: Image.Image):
        if self.augment:
            from .augment import rand_augment_transform
            img = rand_augment_transform()(img)
        if self.rotation:
            img = img.rotate(self.rotation, expand=True)
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BICUBIC)
        processor = self._get_processor()
        media_inputs = processor(images=[img], videos=None, return_tensors='pt')
        return {
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_inputs['image_grid_thw'],
        }


class AspectRatioMonkeyOCRTransform(MonkeyOCRTransform):
    def __init__(self, multi_scales: Sequence[Sequence[int]], augment: bool = False, rotation: int = 0):
        scales = [tuple(scale) for scale in multi_scales]
        if not scales:
            raise ValueError('multi_scales must not be empty.')
        self.multi_scales = scales
        super().__init__(self.multi_scales[0], augment=augment, rotation=rotation)

    def _select_img_size(self, img: Image.Image) -> tuple[int, int]:
        width, height = img.size
        aspect = width / max(height, 1)

        def aspect_distance(size: tuple[int, int]) -> tuple[float, float]:
            target_h, target_w = size
            target_aspect = target_w / target_h
            return abs(math.log(aspect) - math.log(target_aspect)), abs(target_w * target_h - width * height)

        return min(self.multi_scales, key=aspect_distance)

    def __call__(self, img: Image.Image):
        if self.augment:
            from .augment import rand_augment_transform
            img = rand_augment_transform()(img)
        if self.rotation:
            img = img.rotate(self.rotation, expand=True)
        img_size = self._select_img_size(img)
        img = img.resize((img_size[1], img_size[0]), Image.BICUBIC)
        processor = self._get_processor()
        media_inputs = processor(images=[img], videos=None, return_tensors='pt')
        return {
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_inputs['image_grid_thw'],
        }

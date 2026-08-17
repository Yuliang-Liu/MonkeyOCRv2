# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import io
import json
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image

from torch.utils.data import ConcatDataset, Dataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(
            charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False, readahead=False, meminit=False, lock=False
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class JsonDataset(Dataset):
    """Dataset backed by a JSON annotation list with {"image": ..., "label": ...} items."""

    def __init__(
        self,
        gt_path: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root = str(Path(gt_path).absolute().parent)
        self.transform = transform
        self.samples = []
        charset_adapter = CharsetAdapter(charset)
        gt_path = Path(gt_path)
        with gt_path.open() as f:
            entries = json.load(f)
        for entry in entries:
            label = entry['label']
            if remove_whitespace:
                label = ''.join(label.split())
            if normalize_unicode:
                label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
            if len(label) > max_label_len:
                continue
            label = charset_adapter(label)
            if not label:
                continue
            image_path = entry['image']
            if min_image_dim > 0:
                width, height = Image.open(image_path).size
                if min(width, height) < min_image_dim:
                    continue
            self.samples.append((image_path, label))
        log.info(f'json:\t{gt_path}\tnum samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TxtListDataset(Dataset):
    """Dataset backed by a tab-separated text file with `relative_or_abs_image_path<TAB>label` lines."""

    def __init__(
        self,
        gt_path: str,
        image_root: str,
        charset: str,
        max_label_len: Optional[int],
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        transform: Optional[Callable] = None,
    ):
        gt_path = Path(gt_path)
        image_root = Path(image_root)
        self.root = str(gt_path.with_suffix(''))
        self.transform = transform
        self.samples = []
        charset_adapter = CharsetAdapter(charset)
        with gt_path.open() as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                image_rel, label = line.split('\t', 1)
                if remove_whitespace:
                    label = ''.join(label.split())
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                if max_label_len is not None and len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                if not label:
                    continue
                image_path = Path(image_rel)
                if not image_path.is_absolute():
                    image_path = image_root / image_path
                if min_image_dim > 0:
                    width, height = Image.open(image_path).size
                    if min(width, height) < min_image_dim:
                        continue
                self.samples.append((str(image_path), label))
        log.info(f'txt:\t{gt_path}\tnum samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

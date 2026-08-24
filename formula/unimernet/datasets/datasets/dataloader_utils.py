"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import time
import random
import torch
from unimernet.datasets.data_utils import move_to_cuda
from torch.utils.data import DataLoader, Sampler


class MultiIterLoader:
    """
    A simple wrapper for iterating over multiple iterators.

    Args:
        loaders (List[Loader]): List of Iterator loaders.
        ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
    """

    def __init__(self, loaders, ratios=None):
        # assert all loaders has __next__ method
        for loader in loaders:
            assert hasattr(
                loader, "__next__"
            ), "Loader {} has no __next__ method.".format(loader)

        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]

        self.loaders = loaders
        self.ratios = ratios

    def __next__(self):
        # random sample from each loader by ratio
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        return next(self.loaders[loader_idx])

    def __len__(self):
        return sum([len(_) for _ in self.loaders if hasattr(_, "__len__")])


class ConcatLoader:
    """
    A simple wrapper for iterating over multiple iterators.

    Args:
        loaders (List[Loader]): List of Iterator loaders.
        ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
    """

    def __init__(self, loaders):
        # assert all loaders has __next__ method
        for loader in loaders:
            assert hasattr(
                loader, "__len__"
            ), "Loader {} has no __len__ method.".format(loader)

        self._epoch = 0
        self._loader_lens = [len(_) for _ in loaders]
        self._rest_lens = self._loader_lens.copy()

        self.loaders = loaders

    def __next__(self):
        # random sample from each loader by ratio
        loader_idx = random.choices(range(len(self.loaders)), self._rest_lens, k=1)[0]
        self._rest_lens[loader_idx] -= 1
        if sum(self._rest_lens) == 0:
            self._epoch += 1
            self._rest_lens = self._loader_lens.copy()
        return next(self.loaders[loader_idx])

    def __len__(self):
        return sum([len(_) for _ in self.loaders if hasattr(_, "__len__")])


class PrefetchLoader(object):
    """
    Modified from https://github.com/ChenRocks/UNITER.

    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch

            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class TokenBucketBatchSampler(Sampler):
    """Batch indices with similar visual token counts to reduce padding."""

    def __init__(
        self,
        lengths,
        batch_size,
        *,
        shuffle=True,
        drop_last=True,
        bucket_size=2048,
        seed=0,
        num_replicas=1,
        rank=0,
    ):
        self.lengths = [int(length) if length is not None else 0 for length in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(int(bucket_size), self.batch_size)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        if self.num_replicas > 1:
            if self.drop_last:
                usable = len(batches) - (len(batches) % self.num_replicas)
                batches = batches[:usable]
            elif batches:
                while len(batches) % self.num_replicas != 0:
                    batches.append(batches[len(batches) % len(batches)])
        return batches

    def __iter__(self):
        batches = self._global_batches()
        yield from batches[self.rank :: self.num_replicas]

    def __len__(self):
        batches = self._global_batches()
        return len(batches[self.rank :: self.num_replicas])


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            if hasattr(self._dataloader.batch_sampler, "set_epoch"):
                self._dataloader.batch_sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)

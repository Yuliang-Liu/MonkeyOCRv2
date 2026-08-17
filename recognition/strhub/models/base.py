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

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nltk import edit_distance

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.optim import create_optimizer_v2

from strhub.data.utils import BaseTokenizer, CharsetAdapter, CTCTokenizer, Tokenizer


@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


EPOCH_OUTPUT = list[dict[str, BatchResult]]


class BaseSystem(pl.LightningModule, ABC):

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        charset_test: str,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.outputs: dict[int, EPOCH_OUTPUT] = {}
        self._train_start_time: Optional[float] = None

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def on_train_start(self) -> None:
        self._train_start_time = time.monotonic()

    def log_training_loss(self, loss: Tensor) -> None:
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elapsed = 0.0 if self._train_start_time is None else time.monotonic() - self._train_start_time
        completed_steps = self.global_step + 1
        total_steps = self.trainer.estimated_stepping_batches
        lr = 0.0
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        remaining = 0.0
        if completed_steps > 0 and total_steps >= completed_steps:
            remaining = elapsed / completed_steps * (total_steps - completed_steps)
        print(
            f'[train] epoch={self.current_epoch} '
            f'step={completed_steps}/{total_steps} '
            f'loss={loss.detach().item():.6f} '
            f'lr={lr:.8f} '
            f'elapsed={self._format_seconds(elapsed)} '
            f'eta={self._format_seconds(remaining)}',
            flush=True,
        )

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            labels: Text labels of the images

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.0
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(
            optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct, cycle_momentum=False
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, labels = batch

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images)
            loss = loss_numel = None  # Only used for validation; not needed at test-time.

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            # Follow ICDAR 2019 definition of N.E.D.
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))

    @staticmethod
    def _aggregate_results(outputs: EPOCH_OUTPUT) -> tuple[float, float, float]:
        if not outputs:
            return 0.0, 0.0, torch.tensor(float('nan'))
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_norm_ED = 0
        total_size = 0
        for result in outputs:
            result = result['output']
            if result.loss_numel:
                total_loss += result.loss_numel * result.loss
                total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_norm_ED += result.ned
            total_size += result.num_samples
        acc = total_n_correct / total_size
        ned = 1 - total_norm_ED / total_size
        loss = total_loss / total_loss_numel if total_loss_numel else torch.tensor(float('nan'), device='cpu')
        return acc, ned, loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        result = self._eval_step(batch, True)
        self.outputs.setdefault(dataloader_idx, []).append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        all_outputs = []
        val_sets = getattr(self.trainer.datamodule.val_dataset, 'datasets', [self.trainer.datamodule.val_dataset])
        per_dataset_metrics = []
        for dataloader_idx in sorted(self.outputs):
            dataset_outputs = self.outputs[dataloader_idx]
            all_outputs.extend(dataset_outputs)
            acc, ned, loss = self._aggregate_results(dataset_outputs)
            ds_name = Path(val_sets[dataloader_idx].root).name if dataloader_idx < len(val_sets) else f'val_{dataloader_idx}'
            per_dataset_metrics.append((ds_name, acc, ned, loss))
        self.outputs.clear()

        acc, ned, loss = self._aggregate_results(all_outputs)
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)
        for ds_name, ds_acc, ds_ned, ds_loss in per_dataset_metrics:
            print(
                f'[val] epoch={self.current_epoch} '
                f'dataset={ds_name} accuracy={100 * ds_acc:.4f} '
                f'NED={100 * ds_ned:.4f} loss={ds_loss.item():.6f}',
                flush=True,
            )
        print(
            f'[val] epoch={self.current_epoch} dataset=combined '
            f'accuracy={100 * acc:.4f} NED={100 * ned:.4f} loss={loss.item():.6f}',
            flush=True,
        )

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, False)


class CrossEntropySystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        model_max_len = getattr(getattr(self, 'model', None), 'max_label_length', max_len)
        if max_len > model_max_len:
            logits = self.forward(images)
            return logits, torch.tensor(float('nan'), device=self.device), 0
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class CTCSystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        tokenizer = CTCTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        logits = self.forward(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, labels)), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N

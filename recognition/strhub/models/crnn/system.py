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

from typing import Optional, Sequence

from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT

from strhub.models.base import CTCSystem
from strhub.models.utils import init_weights

from .model import CRNN as Model
from .model import CRNNMonkey


class CRNN(CTCSystem):

    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        hidden_size: int,
        leaky_relu: bool,
        encoder_backend: str = 'baseline',
        patch_size: Sequence[int] = (14, 14),
        monkey_vit_dir: Optional[str] = None,
        freeze_mode: str = 'none',
        **kwargs,
    ) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        if encoder_backend == 'baseline':
            self.model = Model(img_size[0], 3, len(self.tokenizer), hidden_size, leaky_relu)
            self.model.apply(init_weights)
        elif encoder_backend == 'monkey':
            if monkey_vit_dir is None:
                raise ValueError('monkey_vit_dir is required when encoder_backend=monkey')
            self.model = CRNNMonkey(img_size, patch_size, len(self.tokenizer), hidden_size, monkey_vit_dir)
            self.model.height_pool.apply(init_weights)
            self.model.rnn.apply(init_weights)
        else:
            raise ValueError(f'Unsupported encoder_backend: {encoder_backend}')
        self.freeze_mode = freeze_mode
        self._apply_freeze_mode()

    def _apply_freeze_mode(self) -> None:
        if self.freeze_mode == 'none':
            return
        if self.freeze_mode == 'freeze_encoder':
            encoder = getattr(self.model, 'encoder', None)
            if encoder is None:
                raise ValueError('freeze_encoder requires a model with an encoder')
            encoder.requires_grad_(False)
            return
        raise ValueError(f'Unsupported freeze_mode: {self.freeze_mode}')

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(images)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        self.log_training_loss(loss)
        return loss

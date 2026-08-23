"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from unimernet.processors.base_processor import BaseProcessor


from unimernet.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)

from unimernet.processors.formula_processor import (
    FormulaImageTrainProcessor,
    FormulaImageEvalProcessor,
    FormulaImageMultiScaleTrainProcessor,
)
from unimernet.processors.formula_processor_monkey_hybrid import (
    MonkeyHybridAugImageTrainProcessor,
)

from unimernet.common.registry import registry

__all__ = [
    "BaseProcessor",
    "BlipCaptionProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor

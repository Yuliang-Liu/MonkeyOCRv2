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

"""Extends default ops to accept optional parameters."""
from functools import partial

from timm.data.auto_augment import _LEVEL_DENOM, LEVEL_TO_ARG, NAME_TO_OP, _randomly_negate, rotate


def rotate_expand(img, degrees, **kwargs):
    """Rotate operation with expand=True to avoid cutting off the characters"""
    kwargs['expand'] = True
    return rotate(img, degrees, **kwargs)


def _level_to_arg(level, hparams, key, default):
    magnitude = hparams.get(key, default)
    level = (level / _LEVEL_DENOM) * magnitude
    level = _randomly_negate(level)
    return (level,)


def apply():
    # Overrides
    NAME_TO_OP.update({
        'Rotate': rotate_expand,
    })
    LEVEL_TO_ARG.update({
        'Rotate': partial(_level_to_arg, key='rotate_deg', default=30.0),
        'ShearX': partial(_level_to_arg, key='shear_x_pct', default=0.3),
        'ShearY': partial(_level_to_arg, key='shear_y_pct', default=0.3),
        'TranslateXRel': partial(_level_to_arg, key='translate_x_pct', default=0.45),
        'TranslateYRel': partial(_level_to_arg, key='translate_y_pct', default=0.45),
    })

# Copyright (c) OpenMMLab. All rights reserved.
from .citys_metric import CityscapesMetric
from .iou_metric import IoUMetric
from .doc_tamper_metric import DocTamperMetric

__all__ = ['IoUMetric', 'CityscapesMetric', 'DocTamperMetric']

# =============================================================================
# Script is from https://github.com/qcf-568/DocTamper/blob/main/metrics.py
# =======================================================================

from typing import List, Sequence
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
import torch


@METRICS.register_module()
class DocTamperMetric(BaseMetric):

    def __init__(self, num_classes=10):
        super().__init__(collect_device="cpu")
        self.num_classes = num_classes
        self.results = []

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            (self.num_classes * label_true[mask].type(torch.int) + label_pred[mask]).cpu().numpy(),
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            gt_label = data_sample['gt_sem_seg']['data'].squeeze()

            hist = self._fast_hist(pred_label.flatten(), gt_label.flatten())

            # cal precision and recall
            matched = (pred_label * gt_label).sum()
            pred_sum = pred_label.sum()
            target_sum = gt_label.sum()
            precisons = (matched / (pred_sum + 1e-8)).mean().item()
            recalls = (matched / target_sum).mean().item()

            self.results.append([hist, precisons, recalls])

    def compute_metrics(self, results: list) -> dict:
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.presicion = []
        self.recall = []
        for result in results:
            self.hist += result[0]
            self.presicion.append(result[1])
            self.recall.append(result[2])

        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        metrics = dict(
            acc=acc_cls,
            iou=iu[1],
            precison=np.array(self.presicion).mean(),
            recall=np.array(self.recall).mean(),
            f1=(2 * np.array(self.presicion).mean() * np.array(self.recall).mean() / (
                        np.array(self.presicion).mean() + np.array(self.recall).mean() + 1e-8)),
            TP=self.hist[1, 1],
            TN=self.hist[0, 0],
            FP=self.hist[0, 1],
            FN=self.hist[1, 0],
        )
        return metrics

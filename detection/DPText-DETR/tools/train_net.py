# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import distutils.version  # Ensure distutils.version is available for torch tensorboard import.
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
import itertools

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator,TextDetEvaluator


from detectron2.engine.train_loop import SimpleTrainer
import time

class Bf16AMPTrainer(SimpleTrainer):
    def run_step(self):
        assert self.model.training, "[Bf16AMPTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not getattr(self, "zero_grad_before_forward", False):
            self.optimizer.zero_grad()
            
        losses.backward()

        self.after_backward()

        if getattr(self, "async_write_metrics", False):
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)
            
        self.optimizer.step()


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Use Bfloat16 Custom AMP instead of default Float16 AMP
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = Bf16AMPTrainer(self.model, self.data_loader, self.optimizer)

    # @classmethod
    # def build_model(cls, cfg):
    #     model = super().build_model(cfg)
    #     model = torch.compile(model, dynamic=True)
    #     return model

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        try:
            checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
            if resume and self.checkpointer.has_checkpoint():
                self.start_iter = checkpoint.get("iteration", -1) + 1
        except ValueError as e:
            if "loaded state dict has a different number of parameter groups" in str(e):
                import logging
                logger = logging.getLogger("adet.trainer")
                logger.warning("Optimizer parameter groups differ (probably due to unfreezing). Loading ONLY model weights!")
                optimizer = self.checkpointer.checkpointables.pop("optimizer", None)
                scheduler = self.checkpointer.checkpointables.pop("scheduler", None)
                # clear _parsed_url_during_load before calling resume_or_load again
                if hasattr(self.checkpointer, '_parsed_url_during_load'):
                    self.checkpointer._parsed_url_during_load = None
                checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
                if optimizer:
                    self.checkpointer.checkpointables["optimizer"] = optimizer
                if scheduler:
                    self.checkpointer.checkpointables["scheduler"] = scheduler
                # Keep iteration from checkpoint if available; otherwise fall back to 0
                if resume and checkpoint and self.checkpointer.has_checkpoint():
                    self.start_iter = checkpoint.get("iteration", -1) + 1
                else:
                    self.start_iter = 0
            else:
                raise e

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_test_loader` with a customized
        DatasetMapper.
        """
        mapper = DatasetMapperWithBasis(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            if cfg.TEST.DET_ONLY:
                return TextDetEvaluator(dataset_name, cfg, True, output_folder)
            else:
                return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        def get_monkey_v2_vitae_layer_id(n, num_layers):
            if ".stem." in n:
                return 0
            if ".stage1." in n:
                splits = n[n.find(".stage1.") :].split(".")
                if len(splits) > 2 and splits[2].isdigit():
                    return int(splits[2]) + 1
                return 2
            if ".rc1." in n:
                return 2
            if ".stage2." in n:
                splits = n[n.find(".stage2.") :].split(".")
                if len(splits) > 2 and splits[2].isdigit():
                    return int(splits[2]) + 3
                return 4
            if ".rc2." in n:
                return 4
            if ".stage3." in n:
                splits = n[n.find(".stage3.") :].split(".")
                if len(splits) > 2 and splits[2].isdigit():
                    return int(splits[2]) + 5
                return 12
            if ".rc3." in n:
                return 12
            if ".stage4." in n:
                splits = n[n.find(".stage4.") :].split(".")
                if len(splits) > 2 and splits[2].isdigit():
                    return int(splits[2]) + 13
                return 14
            if ".post_trunk_norm." in n:
                return num_layers + 1
            return num_layers + 1

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if match_name_keywords(key, cfg.SOLVER.ZERO_WEIGHT_DECAY_NAMES):
                weight_decay = 0.0

            if match_name_keywords(key, cfg.SOLVER.LR_BACKBONE_NAMES):
                lr = cfg.SOLVER.LR_BACKBONE
                
                # Apply ViT backbone layer-wise learning rate decay
                is_monkey_v2_vitae = "monkeyocr_v2_vitae" in cfg.MODEL.BACKBONE.NAME.lower() or ".vitae." in key
                is_vit = (
                    "vitdet" in cfg.MODEL.BACKBONE.NAME.lower()
                    or "vision_model" in key
                    or is_monkey_v2_vitae
                )
                if is_vit:
                    decay_rate = getattr(cfg.SOLVER, "LR_DECAY_RATE", 0.7)
                    if decay_rate < 1.0:
                        num_layers = cfg.SOLVER.VIT_NUM_LAYERS
                        layer_id = num_layers + 1
                        if is_monkey_v2_vitae:
                            layer_id = get_monkey_v2_vitae_layer_id(key, num_layers)
                        elif ".pos_embed" in key or ".patch_embed" in key or "embeddings" in key:
                            layer_id = 0
                        elif ".blocks." in key and ".residual." not in key:
                            splits = key[key.find(".blocks.") :].split(".")
                            if len(splits) > 2 and splits[2].isdigit():
                                layer_id = int(splits[2]) + 1
                        lr = lr * (decay_rate ** (num_layers + 1 - layer_id))
            elif match_name_keywords(key, cfg.SOLVER.LR_LINEAR_PROJ_NAMES):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

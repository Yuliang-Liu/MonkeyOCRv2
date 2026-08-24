from typing import List
import numpy as np
import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances

from adet.layers.pos_encoding import PositionalEncoding2D
from adet.modeling.dptext_detr.losses import SetCriterion
from adet.modeling.dptext_detr.matcher import build_matcher
from adet.modeling.dptext_detr.models import DPText_DETR
from adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.uses_external_processor = getattr(self.backbone, "uses_external_processor", False)
        self.expects_image_list = getattr(self.backbone, "expects_image_list", False)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        if self.expects_image_list:
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        valid_sizes = getattr(images, "valid_sizes", images.image_sizes)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            valid_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(results, output_height, output_width):
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    # scale point coordinates
    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


@META_ARCH_REGISTRY.register()
class TransformerPureDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        d2_backbone = MaskedBackbone(cfg)
        self.use_external_image_processor_backbone = d2_backbone.uses_external_processor
        self.backbone_size_divisibility = d2_backbone.backbone.size_divisibility
        self.backbone_padding_constraints = d2_backbone.backbone.padding_constraints
        if cfg.INPUT.FIXED_SIZE.ENABLED:
            # Keep fixed-size recipe behavior when mapper does not pad: pad to square in ImageList.
            train_h, train_w = cfg.INPUT.FIXED_SIZE.TRAIN_SHAPE
            test_h, test_w = cfg.INPUT.FIXED_SIZE.TEST_SHAPE
            square_size = max(int(train_h), int(train_w), int(test_h), int(test_w))
            if square_size > 0:
                self.backbone_padding_constraints = dict(self.backbone_padding_constraints)
                self.backbone_padding_constraints["square_size"] = square_size
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        assert self.use_polygon and self.num_ctrl_points == 16  # only the polygon version is released now
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.dptext_detr = DPText_DETR(cfg, backbone)

        box_matcher, point_matcher = build_matcher(cfg)

        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT}
        enc_weight_dict = {
            'loss_bbox': loss_cfg.BOX_COORD_WEIGHT,
            'loss_giou': loss_cfg.BOX_GIOU_WEIGHT,
            'loss_ce': loss_cfg.BOX_CLASS_WEIGHT
        }
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points']

        self.criterion = SetCriterion(
            self.dptext_detr.num_classes,
            box_matcher,
            point_matcher,
            weight_dict,
            enc_losses,
            dec_losses,
            self.dptext_detr.num_ctrl_points,
            focal_alpha=loss_cfg.FOCAL_ALPHA,
            focal_gamma=loss_cfg.FOCAL_GAMMA
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        if self.use_external_image_processor_backbone:
            images = [x["image"].to(self.device) for x in batched_inputs]
        else:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(
            images,
            self.backbone_size_divisibility,
            padding_constraints=self.backbone_padding_constraints,
        )
        # images.valid_sizes = [
        #     tuple(int(v) for v in x.get("valid_size", image.shape[-2:]))
        #     for x, image in zip(batched_inputs, images.tensor)
        # ]
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "scores", "pred_classes", "polygons"
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            output = self.dptext_detr(images)
            # compute the loss
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            output = self.dptext_detr(images)
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            valid_sizes = getattr(images, "valid_sizes", images.image_sizes)
            results = self.inference(ctrl_point_cls, ctrl_point_coord, valid_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, valid_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes.to(self.device)
            gt_boxes = targets_per_image.gt_boxes.tensor.to(self.device) / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            wh_tensor = torch.tensor([w, h], dtype=torch.float, device=self.device)
            # Ensure everything is on the correct device
            raw_ctrl_points = raw_ctrl_points.to(self.device) 
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.dptext_detr.num_ctrl_points, 2) / \
                             wh_tensor[None, None, :]
            gt_ctrl_points = torch.clamp(gt_ctrl_points[:, :, :2], 0, 1)
            new_targets.append(
                {"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points}
            )
        return new_targets

    def inference(self, ctrl_point_cls, ctrl_point_coord, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, image_size in zip(
                scores, labels, ctrl_point_coord, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]

            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            results.append(result)

        return results

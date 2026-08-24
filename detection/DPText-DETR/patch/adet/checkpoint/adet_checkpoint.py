import logging, pickle, os
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer


logger = logging.getLogger(__name__)


def _is_oclip_state_dict(sd) -> bool:
    """Heuristic: an oCLIP R-50 checkpoint contains only CLIPResNet keys
    (stem.*, layer1.*, layer2.*, layer3.*, layer4.*) and nothing else."""
    if not sd:
        return False
    sd_keys = set(sd.keys())
    if not sd_keys:
        return False
    allowed_prefixes = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
    return all(k.startswith(allowed_prefixes) for k in sd_keys)


def _find_clip_resnet_prefix(model) -> str:
    """Locate the parameter-name prefix of the CLIPResNet wrapped inside
    ``MMOCRCLIPResNetBackbone`` so that a raw oCLIP state_dict can be aligned
    with the meta-arch's full state_dict.

    The model structure (built by ``TransformerPureDetector``) is::

        TransformerPureDetector
        └── dptext_detr (DPText_DETR)
            └── backbone (Joiner, Sequential)
                ├── 0: MaskedBackbone
                │   └── backbone: MMOCRCLIPResNetBackbone
                │       └── net: CLIPResNet  <-- this is what we want
                └── 1: PositionalEncoding2D

    The returned prefix is the dotted path to that ``CLIPResNet``, with a
    trailing dot (e.g. ``"dptext_detr.backbone.0.backbone.net."``).
    """
    sd = model.state_dict()
    # ``stem.0.weight`` is unique to a deep-stem ResNet and is the first
    # parameter in CLIPResNet, so it is the most reliable marker.
    for key in sd:
        if key.endswith("stem.0.weight"):
            return key[: -len("stem.0.weight")]
    raise RuntimeError(
        "Could not locate CLIPResNet in the model: no parameter ending "
        "with 'stem.0.weight' was found."
    )


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, but is able to convert models
    in AdelaiDet, such as LPF backbone.
    """
    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True

        # The oCLIP R-50 weights from mmocr are saved as a raw
        # ``CLIPResNet.state_dict()`` (keys like ``stem.0.weight``,
        # ``layer1.0.conv1.weight``, ...).  The AdetCheckpointer feeds
        # them to ``model.load_state_dict``, but the model wraps the
        # ``CLIPResNet`` under
        # ``dptext_detr.backbone.0.backbone.net.``, so we have to add
        # that prefix back before loading.  We detect the format by the
        # filename and by the shape of the keys; for any matching
        # checkpoint we add the correct prefix.
        if "oclip" in basename and _is_oclip_state_dict(loaded["model"]):
            prefix = _find_clip_resnet_prefix(self.model)
            original_sd = loaded["model"]
            loaded["model"] = {prefix + k: v for k, v in original_sd.items()}
            logger.info(
                "AdetCheckpointer: remapped %d oCLIP keys with prefix '%s' "
                "to match the MMOCRCLIPResNetBackbone wrapper.",
                len(original_sd),
                prefix,
            )

        return loaded

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = False
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# The options for the quality of box prediction
# It can be "ctrness" (as described in FCOS paper) or "iou"
# Using "iou" here generally has ~0.4 better AP on COCO
# Note that for compatibility, we still use the term "ctrness" in the code
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# The normalizer of the classification loss
# The normalizer can be "fg" (normalized by the number of the foreground samples),
# "moving_fg" (normalized by the MOVING number of the foreground samples),
# or "all" (normalized by the number of all samples)
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False
_C.MODEL.FCOS.YIELD_BOX_FEATURES = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)
_C.MODEL.BATEXT.USE_COORDCONV = False
_C.MODEL.BATEXT.USE_AET = False
_C.MODEL.BATEXT.CUSTOM_DICT = "" # Path to the class file.

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1

# if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
_C.MODEL.CONDINST.MAX_PROPOSALS = -1
# if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
# PER IMAGE in terms of their detection scores
_C.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False

# The options for BoxInst, which can train the instance segmentation model with box annotations only
# Please refer to the paper https://arxiv.org/abs/2012.02310
_C.MODEL.BOXINST = CN()
# Whether to enable BoxInst
_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# Instance hyper-parameters
_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5
_C.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0


# ---------------------------------------------------------------------------- #
# (Deformable) Transformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.USE_POLYGON = False
_C.MODEL.TRANSFORMER.ENABLED = True
_C.MODEL.TRANSFORMER.INFERENCE_TH_TEST = 0.3
_C.MODEL.TRANSFORMER.VOC_SIZE = 96
_C.MODEL.TRANSFORMER.NUM_CHARS = 25
_C.MODEL.TRANSFORMER.AUX_LOSS = True
_C.MODEL.TRANSFORMER.ENC_LAYERS = 6
_C.MODEL.TRANSFORMER.DEC_LAYERS = 6
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 256
_C.MODEL.TRANSFORMER.DROPOUT = 0.1
_C.MODEL.TRANSFORMER.NHEADS = 8
_C.MODEL.TRANSFORMER.NUM_QUERIES = 100
_C.MODEL.TRANSFORMER.ENC_N_POINTS = 4
_C.MODEL.TRANSFORMER.DEC_N_POINTS = 4
_C.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE = 6.283185307179586  # 2 PI
_C.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS = 4
_C.MODEL.TRANSFORMER.NUM_CTRL_POINTS = 16

_C.MODEL.TRANSFORMER.EPQM = False # for DPText-DETR
_C.MODEL.TRANSFORMER.EFSA = False

_C.MODEL.TRANSFORMER.LOSS = CN()
_C.MODEL.TRANSFORMER.LOSS.AUX_LOSS = True
_C.MODEL.TRANSFORMER.LOSS.POINT_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.POINT_COORD_WEIGHT = 5.0
_C.MODEL.TRANSFORMER.LOSS.BOX_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.BOX_COORD_WEIGHT = 5.0
_C.MODEL.TRANSFORMER.LOSS.BOX_GIOU_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.TRANSFORMER.LOSS.FOCAL_GAMMA = 2.0

# ---------------------------------------------------------------------------- #
# MonkeyOCR ViTDet Backbone Options
# ---------------------------------------------------------------------------- #
_C.MODEL.MONKEY_VIT = CN()
_C.MODEL.MONKEY_VIT.ENABLED = False
_C.MODEL.MONKEY_VIT.MODEL_DIR = ""
_C.MODEL.MONKEY_VIT.WEIGHTS = ""
_C.MODEL.MONKEY_VIT.MAX_PIXELS = 1003520
_C.MODEL.MONKEY_VIT.OUT_CHANNELS = 384
_C.MODEL.MONKEY_VIT.USE_BF16 = True
_C.MODEL.MONKEY_VIT.TRUST_REMOTE_CODE = True
_C.MODEL.MONKEY_VIT.FREEZE = False
_C.MODEL.MONKEY_VIT.DROP_PATH_RATE = 0.0
_C.MODEL.MONKEY_VIT.ADAPTER = CN()
_C.MODEL.MONKEY_VIT.ADAPTER.ENABLED = False
_C.MODEL.MONKEY_VIT.ADAPTER.CONV_INPLANE = 64
_C.MODEL.MONKEY_VIT.ADAPTER.N_POINTS = 4
_C.MODEL.MONKEY_VIT.ADAPTER.DEFORM_NUM_HEADS = 6
_C.MODEL.MONKEY_VIT.ADAPTER.CFFN_RATIO = 0.25
_C.MODEL.MONKEY_VIT.ADAPTER.DEFORM_RATIO = 1.0
_C.MODEL.MONKEY_VIT.ADAPTER.INTERACTION_INDEXES = [[0, 2], [3, 5], [6, 8], [9, 11]]
_C.MODEL.MONKEY_VIT.ADAPTER.ADD_VIT_FEATURE = True
_C.MODEL.MONKEY_VIT.ADAPTER.USE_EXTRA_EXTRACTOR = True
_C.MODEL.MONKEY_VIT.ADAPTER.DROP_PATH_RATE = 0.2
_C.MODEL.MONKEY_VIT.ADAPTER.WITH_CP = True

# ---------------------------------------------------------------------------- #
# MonkeyOCRv2 ViTAE Backbone Options
# ---------------------------------------------------------------------------- #
_C.MODEL.MONKEY_V2_VITAE = CN()
_C.MODEL.MONKEY_V2_VITAE.MODEL_DIR = "/data/xhwang/projects/monkeyocrv2vitae/vitae_small_new"
_C.MODEL.MONKEY_V2_VITAE.WEIGHTS = "/data/xhwang/projects/monkeyocrv2vitae/vitae_small_new/model.safetensors"
_C.MODEL.MONKEY_V2_VITAE.OUT_FEATURES = ["res3", "res4", "res5"]
_C.MODEL.MONKEY_V2_VITAE.OUT_CHANNELS = 0
_C.MODEL.MONKEY_V2_VITAE.USE_BF16 = True
_C.MODEL.MONKEY_V2_VITAE.TRUST_REMOTE_CODE = True
_C.MODEL.MONKEY_V2_VITAE.FREEZE = False
_C.MODEL.MONKEY_V2_VITAE.GRADIENT_CHECKPOINTING = True

# ---------------------------------------------------------------------------- #
# mmocr CLIPResNet backbone (oCLIP-pretrained R-50) Options
# ---------------------------------------------------------------------------- #
_C.MODEL.MMOCR_CLIP_RESNET = CN()
_C.MODEL.MMOCR_CLIP_RESNET.DEPTH = 50
_C.MODEL.MMOCR_CLIP_RESNET.STRIDES = [1, 2, 2, 2]
_C.MODEL.MMOCR_CLIP_RESNET.DEEP_STEM = True
_C.MODEL.MMOCR_CLIP_RESNET.AVG_DOWN = True
_C.MODEL.MMOCR_CLIP_RESNET.FROZEN_STAGES = -1
_C.MODEL.MMOCR_CLIP_RESNET.NORM_TYPE = "BN"
_C.MODEL.MMOCR_CLIP_RESNET.NORM_EVAL = False
_C.MODEL.MMOCR_CLIP_RESNET.WITH_CP = False
_C.MODEL.MMOCR_CLIP_RESNET.ZERO_INIT_LAST_BN = True
_C.MODEL.MMOCR_CLIP_RESNET.USE_BF16 = False
_C.MODEL.MMOCR_CLIP_RESNET.OUT_FEATURES = ["res3", "res4", "res5"]
_C.MODEL.MMOCR_CLIP_RESNET.INIT_CFG = None

# ---------------------------------------------------------------------------- #
# Smart Resize Options (short-edge + multiple of factor)
# ---------------------------------------------------------------------------- #
_C.INPUT.SMART_RESIZE = CN()
_C.INPUT.SMART_RESIZE.ENABLED = False
_C.INPUT.SMART_RESIZE.FACTOR = 28
_C.INPUT.SMART_RESIZE.MIN_PIXELS = 3136
_C.INPUT.SMART_RESIZE.MAX_PIXELS = 1003520

# Image interpolation method for resizing.
# Options: "bilinear", "bicubic", "nearest", "lanczos"
_C.INPUT.INTERP = "bilinear"

# ---------------------------------------------------------------------------- #
# ViTDet-style Fixed Size Options (resize-scale + fixed-size crop)
# ---------------------------------------------------------------------------- #
_C.INPUT.FIXED_SIZE = CN()
_C.INPUT.FIXED_SIZE.ENABLED = False
_C.INPUT.FIXED_SIZE.TRAIN_SHAPE = (1024, 1024)
_C.INPUT.FIXED_SIZE.TEST_SHAPE = (1024, 1024)
_C.INPUT.FIXED_SIZE.MIN_SCALE = 0.1
_C.INPUT.FIXED_SIZE.MAX_SCALE = 2.0
_C.INPUT.FIXED_SIZE.CROP_INSTANCE = False
_C.INPUT.FIXED_SIZE.CROP_MAX_TRIES = 80
_C.INPUT.FIXED_SIZE.PAD = True
_C.INPUT.FIXED_SIZE.PAD_VALUE = 128.0
_C.INPUT.FIXED_SIZE.SEG_PAD_VALUE = 255


_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.LR_DECAY_RATE = 0.7
_C.SOLVER.VIT_NUM_LAYERS = 12
_C.SOLVER.ZERO_WEIGHT_DECAY_NAMES = []
_C.SOLVER.LR_BACKBONE_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_MULT = 0.1


_C.TEST.DET_ONLY = True
_C.TEST.USE_LEXICON = False
# 1 - Full lexicon (for totaltext, ctw1500...)
_C.TEST.LEXICON_TYPE = 1
_C.TEST.WEIGHTED_EDIT_DIST = False

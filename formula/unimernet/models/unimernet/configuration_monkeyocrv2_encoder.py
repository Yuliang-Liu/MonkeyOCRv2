from transformers.configuration_utils import PretrainedConfig


class MonkeyOCRv2VisionConfig(PretrainedConfig):
    model_type = "monkeyocr_vit"

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_size: int | None = None,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        vision_attn_implementation: str = "flash_attention_2",
        initializer_range: float = 0.02,
        init_merger_std: float = 0.02,
        is_causal: bool = False,
        post_norm: bool = True,
        gradient_checkpointing: bool = False,
        use_native_token_order: bool = False,
        use_2d_token_bridge: bool = False,
        bridge_output_dim: int | None = None,
        bridge_target_grid: list[int] | tuple[int, int] | None = None,
        bridge_num_local_blocks: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # VisionEncoderDecoderModel builds enc_to_dec_proj from config.hidden_size.
        # Monkey's encoder output dim is embed_dim, so hidden_size must match it.
        self.hidden_size = hidden_size if hidden_size is not None else embed_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.vision_attn_implementation = vision_attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.use_native_token_order = use_native_token_order
        self.use_2d_token_bridge = use_2d_token_bridge
        self.bridge_output_dim = bridge_output_dim
        self.bridge_target_grid = list(bridge_target_grid) if bridge_target_grid is not None else None
        self.bridge_num_local_blocks = bridge_num_local_blocks
        self.is_decoder = False
        self.add_cross_attention = False

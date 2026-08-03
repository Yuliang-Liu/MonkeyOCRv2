from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field
from functools import partial
from packaging import version
import transformers
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall, Word
from .utils import TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta
from PIL import Image
from ..vision_utils import load_audio, load_batch, load_image, rescale_image
import torch
import torch.nn.functional as F

@dataclass
class MonkeyOCRv2TemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


class MonkeyOCRv2Template(Template):
    image_token_id = 151655
    placeholder_tokens = ['<|image_pad|>']
    support_padding_free = True
    
    def init_env_args(self):
        super().init_env_args()
        self.transformers_version = version.parse(transformers.__version__)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image
        assert media_type == 'image'
        patch_size = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size / 2
        inputs.images[index] = fetch_image({'image': inputs.images[index]}, image_patch_size=patch_size)

        if "images_ori" in getattr(inputs, "extra_kwargs") and inputs.extra_kwargs['images_ori']:
            inputs.extra_kwargs['images_ori'][index] = fetch_image({'image': inputs.extra_kwargs['images_ori'][index]}, image_patch_size=patch_size)

        if self.mode == 'lmdeploy':
            return ['<|vision_start|>', [-100], '<|vision_end|>']
        else:
            return ['<|vision_start|><|image_pad|><|vision_end|>']

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        self._preprocess_function_call(inputs)
        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)
        
        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images
        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                images[i] = self._load_image(images[i], load_images)
        if inputs.objects:
            self._get_height_width(inputs)
        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            images = [rescale_image(img, self.max_pixels) for img in images]
        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images

        if "images_ori" in getattr(inputs, "extra_kwargs") and inputs.extra_kwargs['images_ori']:
            images_ori_path = inputs.extra_kwargs['images_ori']
            images_ori = []
            for image_ori_path in images_ori_path:
                image_ori = {'bytes': None, 'path': image_ori_path}
                images_ori.append(image_ori)
            load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
            load_images_origin = load_images
            if self.max_pixels is not None or inputs.objects:
                load_images = True
            if images_ori:
                for i, image_ori in enumerate(images_ori):
                    images_ori[i] = self._load_image(images_ori[i], load_images)
            if inputs.objects:
                self._get_height_width(inputs)
            if self.max_pixels is not None:
                # Scale the image proportionally without affecting the scaled objects.
                images_ori = [rescale_image(img_ori, self.max_pixels) for img_ori in images_ori]
            if images_ori and not load_images_origin:  # fix pt & qwen-vl
                for i, image_ori in enumerate(images_ori):
                    if isinstance(image_ori, Image.Image):
                        images_ori[i] = self._save_pil_image(image_ori)
            inputs.extra_kwargs['images_ori'] = images_ori
        
        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))
        if inputs.is_multimodal:
            self._add_default_tags(inputs)
    

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:

        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        images = getattr(inputs, "images")
        if images:
            media_token = self.image_token_id
            media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt', do_resize=False)
            media_grid_thw = media_inputs['image_grid_thw']
            idx_list = findall(input_ids, media_token)
            merge_length = processor.image_processor.merge_size**2

            # add
            if "images_ori" in getattr(inputs, "extra_kwargs") and inputs.extra_kwargs['images_ori']:
                images_ori = inputs.extra_kwargs['images_ori']
                pixel_values_ori = processor.image_processor(images=images_ori, videos=None, return_tensors='pt', do_resize=False)['pixel_values']
                encoded['pixel_values_ori'] = pixel_values_ori
                

            def _get_new_tokens(i):
                token_len = (media_grid_thw[i].prod() // merge_length)
                return [media_token] * token_len
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded.update(media_inputs)
        
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        res = super()._data_collator_mm_data(batch)

        pixel_values_ori = [b['pixel_values_ori'] for b in batch if b.get('pixel_values_ori') is not None]
        if len(pixel_values_ori) > 0:
            res['pixel_values_ori'] = torch.concat(pixel_values_ori)

        grid_thw = self.concat_tensor(batch, 'image_grid_thw', 0)
        if grid_thw is not None:
            res['image_grid_thw'] = grid_thw
        return res
    
    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        # packed['position_ids'] = torch.concat(position_ids, dim=-1)
        return packed
    
    def _get_position_ids(self, encoded: Dict[str, Any]) -> torch.Tensor:
        """Build position_ids for a single (un-padded) sample.

        - Prefer model-provided mRoPE position ids (e.g. Qwen2/2.5-VL via `get_rope_index`)
        - Fallback to standard 1D position ids.
        """
        input_ids = encoded.get('input_ids')
        if input_ids is None:
            raise KeyError(f'encoded has no input_ids: {list(encoded.keys())}')

        if isinstance(input_ids, (list, tuple)):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        # now: [bs(=1), seq_len]
        device = input_ids.device
        dtype = torch.long
        seq_len = input_ids.shape[-1]

        # Fallback: standard causal LM position ids [1, seq_len]
        return torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(0)
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
    
class MonkeyOCRv2ExportTemplate(Template):
    image_token_id = 151655
    placeholder_tokens = ['<|image_pad|>']
    support_padding_free = True
    
    def init_env_args(self):
        super().init_env_args()
        self.transformers_version = version.parse(transformers.__version__)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image
        assert media_type == 'image'
        inputs.images[index] = fetch_image({'image': inputs.images[index]})

        if self.mode == 'lmdeploy':
            return ['<|vision_start|>', [-100], '<|vision_end|>']
        else:
            return ['<|vision_start|><|image_pad|><|vision_end|>']

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        self._preprocess_function_call(inputs)
        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)
        
        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images
        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                images[i] = self._load_image(images[i], load_images)
        if inputs.objects:
            self._get_height_width(inputs)
        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            images = [rescale_image(img, self.max_pixels) for img in images]
        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images
        
        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))
        if inputs.is_multimodal:
            self._add_default_tags(inputs)
    

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:

        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        images = getattr(inputs, "images")
        if images:
            media_token = self.image_token_id
            media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt', do_resize=False)
            media_grid_thw = media_inputs['image_grid_thw']
            idx_list = findall(input_ids, media_token)
            merge_length = processor.image_processor.merge_size**2

            def _get_new_tokens(i):
                token_len = (media_grid_thw[i].prod() // merge_length)
                return [media_token] * token_len
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded.update(media_inputs)
        
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        res = super()._data_collator_mm_data(batch)

        grid_thw = self.concat_tensor(batch, 'image_grid_thw', 0)
        if grid_thw is not None:
            res['image_grid_thw'] = grid_thw
        return res
    
    # def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     position_ids = []
    #     for r in row:
    #         r = r.copy()
    #         r['input_ids'] = torch.tensor(r['input_ids'])[None]
    #         position_ids.append(self._get_position_ids(r))
    #     packed = super().packing_row(row)
    #     packed['position_ids'] = torch.concat(position_ids, dim=-1)
    #     return packed
    
    def _get_position_ids(self, encoded: Dict[str, Any]) -> torch.Tensor:
        """Build position_ids for a single (un-padded) sample.

        - Prefer model-provided mRoPE position ids (e.g. Qwen2/2.5-VL via `get_rope_index`)
        - Fallback to standard 1D position ids.
        """
        input_ids = encoded.get('input_ids')
        if input_ids is None:
            raise KeyError(f'encoded has no input_ids: {list(encoded.keys())}')

        if isinstance(input_ids, (list, tuple)):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        # now: [bs(=1), seq_len]
        device = input_ids.device
        dtype = torch.long
        seq_len = input_ids.shape[-1]

        # Try model-specific mRoPE logic if available
        try:
            model = self._get_model()
            base_model = self.get_base_model(model)
            get_rope_index = getattr(base_model, 'get_rope_index', None)
            if callable(get_rope_index):
                attention_mask = encoded.get('attention_mask')
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                else:
                    if isinstance(attention_mask, (list, tuple)):
                        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
                    if attention_mask.ndim == 1:
                        attention_mask = attention_mask[None, :]

                kwargs = {}
                if encoded.get('image_grid_thw') is not None:
                    kwargs['image_grid_thw'] = encoded['image_grid_thw']
                if encoded.get('video_grid_thw') is not None:
                    kwargs['video_grid_thw'] = encoded['video_grid_thw']

                rope_out = get_rope_index(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                position_ids = rope_out[0] if isinstance(rope_out, (tuple, list)) else rope_out
                if isinstance(position_ids, torch.Tensor):
                    return position_ids.to(device=device)
        except Exception:
            # Fallback to standard position ids below
            pass

        # Fallback: standard causal LM position ids [1, seq_len]
        return torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(0)
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(MonkeyOCRv2TemplateMeta(MLLMTemplateType.monkeyocrv2, template_cls=MonkeyOCRv2Template, default_system=None))

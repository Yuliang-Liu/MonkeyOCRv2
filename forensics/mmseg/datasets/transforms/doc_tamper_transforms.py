import os
import pickle
import random
import tempfile
from pathlib import Path

import jpegio
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS
from PIL import Image
import copy


@TRANSFORMS.register_module()
class ResizeImageOnly(BaseTransform):
    """Resize image only while keeping ground-truth mask at original size.

    This is used for evaluation when the RGB branch expects a different input
    size from the metric resolution. The transform updates image-related meta
    fields but leaves ``gt_seg_map`` untouched.
    """

    def __init__(self, scale, keep_ratio=False, backend='cv2', interpolation='bilinear'):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.backend = backend
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'],
                self.scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'],
                self.scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale'] = self.scale
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = self.keep_ratio
        return results


@TRANSFORMS.register_module()
class RandomJpegCompressAndLoadInfo(BaseTransform):
    def __init__(self, jpeg_compress_time=(1, 2, 3), course=False, quality_lower=75, compress_pk=None, load_info=True,
                 return_rgb=False, compress_mode='single'):
        super().__init__()
        self.jpeg_compress_time = jpeg_compress_time
        self.course = course
        self.quality_lower = quality_lower
        self.compress_pk = compress_pk
        if self.compress_pk is not None:
            assert os.path.exists(self.compress_pk), f"{self.compress_pk} not exists"
            self.compress_pk = pickle.load(open(self.compress_pk, 'rb'))
            # List of [q2, q1, q]

        self.load_info = load_info
        if course:
            raise NotImplementedError
        self.return_rgb = return_rgb
        self.compress_mode = compress_mode
        if self.compress_mode not in ('single', 'doctamper'):
            raise ValueError(
                f'Unsupported compress_mode={compress_mode}. '
                'Expected "single" or "doctamper".'
            )

    def _get_compress_qualities(self, results: dict):
        if self.compress_pk is None:
            jpeg_compress_time = random.choice(self.jpeg_compress_time)
            compress_quality = np.random.randint(
                self.quality_lower,
                101,
                jpeg_compress_time,
            )
        else:
            image_path = results['img_path']
            index = int(Path(image_path).stem)
            compress_quality = self.compress_pk[index]

        if isinstance(compress_quality, np.ndarray):
            compress_quality = compress_quality.tolist()
        elif not isinstance(compress_quality, (list, tuple)):
            compress_quality = [compress_quality]

        compress_quality = [int(q) for q in compress_quality]
        if self.compress_mode == 'single':
            return [compress_quality[0]]
        return compress_quality

    def transform(self, results: dict) -> dict:
        img: np.ndarray = results['img']

        if self.course:
            raise NotImplementedError
        compress_quality = self._get_compress_qualities(results)

        im = Image.fromarray(img)

        if self.return_rgb:
            im_ = im.copy()
        else:
            im_ = im.convert("L")

        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
            for q in compress_quality:
                im_.save(tmp.name, "JPEG", quality=int(q))
                im_ = Image.open(tmp.name)

            if self.load_info:
                jpg = jpegio.read(tmp.name)
                dct = copy.deepcopy(jpg.coef_arrays[0])

                use_qtb = copy.deepcopy(jpg.quant_tables[0]).astype(np.uint8)

                results['dct'] = np.clip(np.abs(dct), 0, 20)
                results['qtb'] = np.expand_dims(np.clip(use_qtb, 0, 63).astype(np.int32), 0)


        # if self.load_info:
        #     jpg = jpegio.read(buffer.getvalue())
        #
        #     dct = copy.deepcopy(jpg.coef_arrays[0])
        #     use_qtb = copy.deepcopy(jpg.quant_tables[0]).astype(np.uint8)
        #
        #     results['dct'] = np.clip(np.abs(dct), 0, 20)
        #     results['qtb'] = np.expand_dims(np.clip(use_qtb, 0, 63).astype(np.int32), 0)

        im = im_.convert('RGB')
        results['img'] = np.array(im)

        return results

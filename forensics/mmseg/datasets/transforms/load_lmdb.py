import os.path as osp

import mmcv
import mmengine
import numpy as np

from mmseg.registry import TRANSFORMS
from pathlib import Path


@TRANSFORMS.register_module()
class LoadImageLabelFromFileLMDB(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmengine.LmdbBackend(**self.file_client_args)

        filename = results['img_path']
        image_key = "image-" + Path(filename).stem
        label_key = "label-" + Path(filename).stem

        img_bytes = self.file_client.get(image_key)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        label_bytes = self.file_client.get(label_key)
        label = mmcv.imfrombytes(label_bytes, flag='unchanged')
        label = (label > 0).astype(np.uint8)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')

        return results

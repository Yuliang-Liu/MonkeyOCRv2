import torch
from .base_dataset import BaseDataset
import os.path as osp
import glob
from io import BytesIO
from PIL import Image


class Im2LatexDataset(BaseDataset):

    def init_samples(self):
        samples = []
        for vis_root, anno_path in zip(self.vis_root, self.anno_path):
            images = [path.replace('\\', '/') for path in glob.glob(osp.join(vis_root, '*.png'))]
            indices = [int(osp.basename(img).split('.')[0]) for img in images]

            with open(anno_path, 'r', encoding='utf-8') as f:
                eqs = f.read().split('\n')
            eqs = [eqs[_] for _ in indices]

            for i, e in zip(images, eqs):
                samples.append({"image": osp.basename(i), "equation": e, "vis_root": vis_root})
        return samples

    def __getitem__(self, index):
        ann = self.samples[index]
        try:
            image = self.vis_processor(self._read_image(ann))
        except Exception:
            return self[(index + 1) % len(self)]
        if image is None:
            return self[(index + 1) % len(self)]
        equation = ann["equation"]
        return {"image": image, "text_input": equation, "id": index}

    def estimate_vision_tokens(self, index):
        estimator = getattr(self.vis_processor, "estimate_num_tokens_for_size", None)
        if estimator is None:
            return None

        if not hasattr(self, "_vision_token_cache"):
            self._vision_token_cache = {}
        if index in self._vision_token_cache:
            return self._vision_token_cache[index]

        ann = self.samples[index]
        image_path = self._resolve_image_path(ann["image"], ann["vis_root"])
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            tokens = int(estimator(width=width, height=height))
        except Exception:
            tokens = None
        self._vision_token_cache[index] = tokens
        return tokens

    def _read_image(self, sample, image_key="image"):
        img_file = sample[image_key]
        vis_root = sample["vis_root"]
        image_path = self._resolve_image_path(img_file, vis_root)
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        return image

    @staticmethod
    def _resolve_image_path(img_file, vis_root):
        if osp.isabs(img_file):
            return img_file

        normalized_img = osp.normpath(img_file)
        normalized_root = osp.normpath(vis_root)
        if normalized_img == normalized_root or normalized_img.startswith(normalized_root + osp.sep):
            return img_file

        return osp.join(vis_root, img_file)

    def init_reader(self):
        if not isinstance(self.vis_root, str):
            vis_root = self.vis_root[0]
        else:
            vis_root = self.vis_root
        if vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            reader = {'type': 'LocalReader', 'body': Image.open}
        return reader

    def collater(self, samples):
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        if image_list and isinstance(image_list[0], dict):
            image = {
                "pixel_values": torch.cat([item["pixel_values"] for item in image_list], dim=0),
                "image_grid_thw": torch.stack([item["image_grid_thw"] for item in image_list], dim=0),
            }
        else:
            image = torch.stack(image_list, dim=0)

        return {
            "image": image,
            "text_input": question_list,
            "id": id_list
        }

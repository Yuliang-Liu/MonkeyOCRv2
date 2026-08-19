import torch
from .formula import Im2LatexDataset


class MultiScaleIm2LatexDataset(Im2LatexDataset):

    def __getitem__(self, index):
        ann = self.samples[index]
        try:
            pil_image = self._read_image(ann)
            image = self.vis_processor(pil_image)
        except Exception:
            return self[(index + 1) % len(self)]
        if image is None:
            return self[(index + 1) % len(self)]
        equation = ann["equation"]
        return {"image": image, "text_input": equation, "id": index, "raw_image": pil_image}

    def collater(self, samples):
        self.vis_processor.reset_scale()
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(self.vis_processor(sample["raw_image"]))
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "id": id_list
        }

import logging
from unimernet.common.registry import registry
from unimernet.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from unimernet.datasets.datasets.formula import Im2LatexDataset
from unimernet.datasets.datasets.formula_multi_scale import MultiScaleIm2LatexDataset


@registry.register_builder("formula_rec_train")
class FormulaRecTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = Im2LatexDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/formula/formula_train.yaml"
    }
    LOG_INFO = "Formula Recgnition Train"

    def build_datasets(self):
        logging.info(f"Building {self.LOG_INFO} datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images
        anno_path = [anno_path] if isinstance(anno_path, str) else anno_path
        vis_root = [vis_root] if isinstance(vis_root, str) else vis_root
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
        )
        print(datasets['train'][0])

        return datasets


@registry.register_builder("multi_scale_formula_rec_train")
class MultiScaleFormulaRecTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultiScaleIm2LatexDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/formula/multi_scale_formula_train.yaml"
    }
    LOG_INFO = "Multi Scale Formula Recgnition Train"

    def build_datasets(self):
        logging.info(f"Building {self.LOG_INFO} datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        anno_path = [anno_path] if isinstance(anno_path, str) else anno_path
        vis_root = [vis_root] if isinstance(vis_root, str) else vis_root

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
        )
        print(datasets['train'][0])

        return datasets


@registry.register_builder("formula_rec_eval")
class FormulaRecEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Im2LatexDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/formula/formula_eval.yaml"
    }
    LOG_INFO = "Formula Recgnition Eval"

    def build_datasets(self):
        logging.info(f"Building {self.LOG_INFO} datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        anno_path = [anno_path] if isinstance(anno_path, str) else anno_path
        vis_root = [vis_root] if isinstance(vis_root, str) else vis_root

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_path=anno_path,
        )
        print(datasets['eval'][0])

        return datasets

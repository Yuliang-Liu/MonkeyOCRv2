# MonkeyOCRv2 Document Tampering Detection

MonkeyOCRv2 Document Tampering Detection combines the visual encoder from [MonkeyOCRv2-S](https://huggingface.co/zenosai/MonkeyOCRv2-S) with the [FFDN](https://github.com/Rapisurazurite/FFDN) method for document tampering detection.


# Models and Results

| Method | Param. | Overall IoU | Overall F | Test IoU | Test P | Test R | Test F | FCD IoU | FCD P | FCD R | FCD F | SCD IoU | SCD P | SCD R | SCD F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSCC-Net | 5M | 13.7 | 31.3 | 17.0 | 25.0 | 83.0 | 39.0 | 13.0 | 19.0 | 82.0 | 30.0 | 11.0 | 15.0 | **83.0** | 25.0 |
| UperNet | 67M | 49.3 | 54.0 | 70.0 | 66.0 | 60.0 | 62.0 | 30.0 | 57.0 | 35.0 | 43.0 | 48.0 | 57.0 | 58.0 | 57.0 |
| CAT-Net | 114M | 67.3 | 71.0 | 78.0 | 75.0 | 69.0 | 72.0 | 66.0 | 85.0 | 70.0 | 76.0 | 58.0 | 65.0 | 65.0 | 65.0 |
| Swin-UPer | 81M | 66.7 | 71.7 | 79.0 | 75.0 | 72.0 | 73.0 | 64.0 | 80.0 | 70.0 | 75.0 | 57.0 | 66.0 | 68.0 | 67.0 |
| SegFormer | 85M | 70.3 | 74.0 | 81.0 | 77.0 | 74.0 | 75.0 | 69.0 | 82.0 | 74.0 | 78.0 | 61.0 | 68.0 | 70.0 | 69.0 |
| Mask2Former | 69M | 69.7 | 78.0 | 84.0 | 82.0 | 83.0 | 82.0 | 66.0 | 81.0 | 75.0 | 78.0 | 59.0 | 70.0 | 79.0 | 74.0 |
| ConvNeXt | 122M | 69.7 | 75.3 | 84.0 | 81.0 | 78.0 | 79.0 | 62.0 | 76.0 | 71.0 | 74.0 | 63.0 | 71.0 | 74.0 | 73.0 |
| ConvNeXtV2 | 121M | 72.7 | 77.7 | 86.0 | 82.0 | 79.0 | 81.0 | 65.0 | 79.0 | 75.0 | 77.0 | 67.0 | 74.0 | 76.0 | 75.0 |
| InternImage | 128M | 73.3 | 77.7 | 84.0 | 81.0 | 77.0 | 79.0 | 72.0 | 83.0 | 79.0 | 81.0 | 64.0 | 73.0 | 74.0 | 73.0 |
| ASC-Former | 80M | 68.2 | 80.8 | 81.5 | 91.8 | 87.8 | 89.8 | 61.3 | 74.9 | 77.1 | 76.0 | 61.9 | 78.0 | 75.0 | 76.5 |
| DTD | 66M | <u>77.0</u> | 79.7 | 84.0 | 81.0 | 77.0 | 79.0 | 79.0 | 88.0 | 82.0 | 85.0 | **68.0** | 75.0 | 76.0 | 75.0 |
| FFDN* (DeepSolo-ViTAEv2) | 69M | 70.7 | <u>82.7</u> | 69.4 | 76.2 | 88.7 | 82.0 | 79.0 | **92.5** | 84.4 | 88.3 | 63.6 | 79.1 | 76.5 | 77.8 |
| FFDN (MonkeyOCRv2-AS) | 71M | **78.2** | **87.5** | **87.4** | **94.8** | **91.8** | **93.3** | **79.9** | 90.4 | **87.4** | **88.9** | 67.2 | **81.0** | 79.8 | **80.4** |

Download the checkpoints from [here](https://huggingface.co/pokeluo/MonkeyOCRv2_forensics/tree/main):
```
hf download pokeluo/MonkeyOCRv2_forensics --include "*.pth" --local-dir ./work_dirs/FFDN/FFDN_monkey_vitae
```


# Prepare Dataset

1. download the dataset from [DocTamper](https://github.com/qcf-568/DocTamper)
2. Link the dataset to `./data/DocTamperV1/unzip_files.`

Your folder structure should look like this:
```text
data
└── DocTamperV1
    ├── unzip_files
    │   ├── DocTamperV1-TrainingSet
    │   ├── DocTamperV1-TestingSet
    │   ├── DocTamperV1-FCD
    │   └── DocTamperV1-SCD
    ├── pks
    │   ├── DocTamperV1-TestingSet_75.pk
    │   ├── DocTamperV1-FCD_75.pk
    │   └── DocTamperV1-SCD_75.pk
    └── processed
        ├── train.txt
        ├── val.txt
        ├── fcd.txt
        └── scd.txt

```

# Getting Started
## Installations

To install, follow these steps:

```bash
conda create -n monkeyocrv2 python=3.10 -y
conda activate monkeyocrv2
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# install jpegio
cd forensics/libs/jpegio
pip install -r requirements.txt
python setup.py install

# install mmsegmentation
cd ../../
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -v -e .
```


### Train and Evaluate

```bash
export GPU_NUMS=4
export PRETRAINED_MODEL=work_dirs/FFDN/FFDN_monkey_vitae/FFDN_monkey_vitae.pth

bash tools/dist_train_val.sh work_config/FFDN/FFDN_monkey_vitae.py ${GPU_NUMS}
bash tools/dist_test_docTamper_lmdb.sh work_config/FFDN/FFDN_monkey_vitae.py ${PRETRAINED_MODEL} ${GPU_NUMS}
```



# Acknowledgement

This project builds upon [FFDN](https://github.com/Rapisurazurite/FFDN)，[MMSeg](https://github.com/open-mmlab/mmsegmentation)，[JPEGIO](https://github.com/dwgoon/jpegio) and the [DocTamper](https://github.com/qcf-568/DocTamper) dataset.



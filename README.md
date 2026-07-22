<div align="center" xmlns="http://www.w3.org/1999/html">
<h2>
<b>MonkeyOCRv2: A Visual-Text Foundation Model for Document AI</b>
</h2>

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCRv2-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2607.11562)
[![MonkeyOCRv2](https://img.shields.io/badge/MonkeyOCRv2-black.svg?logo=Huggingface)](https://huggingface.co/collections/zenosai/monkeyocrv2)
[![MonkeyDocv2](https://img.shields.io/badge/MonkeyDoc_v2-blue.svg?logo=ModelScope)](https://modelscope.cn/datasets/zenosai/MonkeyDocv2)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCRv2?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCRv2/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCRv2?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCRv2/issues?q=is%3Aissue+is%3Aclosed)
[![Demo](https://img.shields.io/badge/Demo-white.svg)](http://vlrlabmonkey.xyz:8891/)

<img src="asserts/overview.png?raw=true" width="600"/>
</div>

## News
* `2026.07.22` 🏆 MonkeyOCRv2-B-Parsing ranks #1 among evaluated open-source models on the official [MDPBench Leaderboard](https://huggingface.co/spaces/Delores-Lin/MDPBench-leaderboard), achieving 83.3 overall across 17 languages, including digital-born and photographed documents.
* `2026.07.21` 📦 We release [MonkeyDoc v2](https://modelscope.cn/datasets/zenosai/MonkeyDocv2), an open multilingual corpus for document-oriented pretraining. We hope it can serve as a shared data foundation for more transparent, reproducible, and fair comparisons in Document AI.
* `2026.07.20` 🤗 MonkeyOCRv2 was featured in [Hugging Face Daily Papers](https://huggingface.co/papers/2607.11562) and has entered the Weekly Papers list. Thanks to the community for the support, upvotes, and feedback!
* `2026.07.14` 🚀 We release [MonkeyOCRv2](https://arxiv.org/abs/2607.11562), including MonkeyOCRv2 vision encoder, MonkeyOCRv2-Parsing for multilingual document parsing, MonkeyOCRv2-Und for efficient document understanding.

## Use MonkeyOCRv2 as a Vision Backbone

MonkeyOCRv2 is released as a standalone, document-native vision encoder. It can be integrated into different OCR and document AI systems as a visual backbone.

The current release has been evaluated on document parsing, document understanding, text recognition, formula recognition, text detection, document tampering detection, and overlapping-text segmentation.

Beyond these evaluated tasks, the encoder may also be useful for text-rich scenarios such as scientific papers, historical documents, medical reports, charts and tables, and remote-sensing maps or reports. We welcome community exploration of these directions.

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained(
    "zenosai/MonkeyOCRv2-B",
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```
See the Vision Encoder [Quick Start](https://github.com/Yuliang-Liu/MonkeyOCRv2#vision-encoder) for installation and feature-extraction examples. If you adapt MonkeyOCRv2 to a new task or domain, feel free to open an issue or pull request and share the results.

## MonkeyDoc v2
MonkeyDoc v2 is currently the largest document image pre-training image-text pair dataset, comprising 113 million document images across 17 languages. The open-sourcing of MonkeyDoc v2 is still underway. So far, we have released 52 million synthetic samples and 41 million real-world samples. You can download the full datset as follows:
```bash
pip install modelscope
modelscope download --dataset zenosai/MonkeyDocv2 --local_dir ./MonkeyDocv2
```
Downloading the complete dataset requires approximately 10 TB of disk space. We recommend ensuring that you have at least 11 TB of available storage before starting the download.

## Model Zoo

#### 1. Vision Encoder

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Backbone</th>
      <th>Params</th>
      <th>Pretraining<br>Resolution</th>
      <th>Applicable Tasks</th>
      <th>Checkpoint Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Monkey<br>OCRv2-S</td><td>ViT-S</td><td>28M</td><td>1280*28*28</td><td>Recognition / Parsing / Understanding</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-S">🤗HuggingFace</a><br><a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-S">🤖ModelScope</a></td>
    </tr>
    <tr>
      <td>Monkey<br>OCRv2-B</td><td>ViT-B</td><td>113M</td><td>1280*28*28</td><td>Recognition / Parsing / Understanding</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-B">🤗HuggingFace</a><br><a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-B">🤖ModelScope</a></td>
    </tr>
    <tr>
      <td>Monkey<br>OCRv2-AS</td><td>ViTAEv2-S</td><td>21M</td><td>1760*32*32</td><td>Detection / Segmentation</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-AS">🤗HuggingFace</a><br><a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-AS">🤖ModelScope</a></td>
    </tr>
  </tbody>
</table>

#### 2. Document Parsing Model

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Link</th>
      <th>Total Params</th>
      <th>ViT</th>
      <th>LLM</th>
      <th>All</th>
      <th>Digit.</th>
      <th>Photo.</th>
      <th>Latin Avg.</th>
      <th>DE</th>
      <th>EN</th>
      <th>ES</th>
      <th>FR</th>
      <th>ID</th>
      <th>IT</th>
      <th>NL</th>
      <th>PT</th>
      <th>VI</th>
      <th>Non-Latin Avg.</th>
      <th>AR</th>
      <th>HI</th>
      <th>JP</th>
      <th>KO</th>
      <th>RU</th>
      <th>TH</th>
      <th>ZH</th>
      <th>ZH-T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MonkeyOCRv2-S-Parsing</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing">HuggingFace</a> <a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-S-Parsing">ModelScope</a></td>
      <td>0.6B</td>
      <td>0.03B</td>
      <td>0.6B</td>
      <td>82.5</td>
      <td>87.9</td>
      <td>80.7</td>
      <td>83.2</td>
      <td>87.3</td>
      <td>83.6</td>
      <td>76.8</td>
      <td>73.6</td>
      <td>85.4</td>
      <td>87.2</td>
      <td>85.5</td>
      <td>87.4</td>
      <td>81.9</td>
      <td>81.7</td>
      <td>91.2</td>
      <td>87.1</td>
      <td>69.9</td>
      <td>88.7</td>
      <td>78.0</td>
      <td>79.8</td>
      <td>84.4</td>
      <td>74.7</td>
    </tr>
    <tr>
      <td>MonkeyOCRv2-B-Parsing</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing">HuggingFace</a> <a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-B-Parsing">ModelScope</a></td>
      <td>0.7B</td>
      <td>0.1B</td>
      <td>0.6B</td>
      <td>83.3</td>
      <td>88.1</td>
      <td>81.7</td>
      <td>84.2</td>
      <td>87.7</td>
      <td>84.5</td>
      <td>75.2</td>
      <td>78.4</td>
      <td>86.5</td>
      <td>88.6</td>
      <td>86.1</td>
      <td>87.9</td>
      <td>83.2</td>
      <td>82.1</td>
      <td>90.7</td>
      <td>87.2</td>
      <td>71.9</td>
      <td>87.6</td>
      <td>80.1</td>
      <td><strong>80.8</strong></td>
      <td>83.6</td>
      <td>75.3</td>
    </tr>
  </tbody>
</table>


#### 3. Document Understanding Model
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Link</th>
      <th>Total Params</th>
      <th>Overall</th>
      <th>DocVQA</th>
      <th>InfoVQA</th>
      <th>DF</th>
      <th>KLC</th>
      <th>WTQ</th>
      <th>ChartQA</th>
      <th>DT-VQA</th>
      <th>OCRBench</th>
    </tr>
  </thead>
    <tr>
      <td>MonkeyOCRv2-S-Und</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-S-Und">HuggingFace</a> <a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-S-Und">ModelScope</a></td>
      <td>1.7B</td>
      <td>55.9</td>
      <td>79.3</td>
      <td>44.5</td>
      <td>65.1</td>
      <td>37.6</td>
      <td>43.0</td>
      <td>62.0</td>
      <td>63.1</td>
      <td>52.2</td>
    </tr>
    <tr>
      <td>MonkeyOCRv2-B-Und</td><td><a href="https://huggingface.co/zenosai/MonkeyOCRv2-B-Und">HuggingFace</a> <a href="https://modelscope.cn/models/zenosai/MonkeyOCRv2-B-Und">ModelScope</a></td>
      <td>1.8B</td>
      <td>57.2</td>
      <td>79.3</td>
      <td>46.3</td>
      <td>65.8</td>
      <td>38.2</td>
      <td>43.2</td>
      <td>62.0</td>
      <td>64.3</td>
      <td>58.1</td>
    </tr>
  </tbody>
</table>

## Quick Start
### Vision Encoder
#### 1. Install
Install transformers and flash attention:
```bash
conda create -n MonkeyOCRv2 python=3.10
conda activate MonkeyOCRv2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.57.6
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install accelerate
pip install qwen_vl_utils
```
#### 2. Download Model Weights
Download our model from Huggingface.
```bash
python download_model.py -n MonkeyOCRv2-B # or MonkeyOCRv2-S / MonkeyOCRv2-AS
```
You can also download our model from ModelScope.

```bash
pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-B # or MonkeyOCRv2-S / MonkeyOCRv2-AS
```
#### 3. Extract Image Feature
```bash
cd vision
# For MonkeyOCRv2-B and MonkeyOCRv2-S
python extract_feature.py
# For MonkeyOCRv2-AS
python extract_feature_vitae.py
```

### Document Parsing
#### 1. Install
Install vLLM following its [official guide](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/):
```bash
conda create -n MonkeyOCRv2Parsing python=3.10
conda activate MonkeyOCRv2Parsing
pip install uv
uv pip install vllm==0.11.2 --torch-backend=auto -i https://pypi.tuna.tsinghua.edu.cn/simple requests
pip install -r parsing/requirements.txt
```

#### 2. Download Model Weights
Download our model from Huggingface.
```bash
python download_model.py -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```
You can also download our model from ModelScope.

```bash
pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```

#### 3. vLLM Serving
You should start a vLLM service before parsing documents:
```bash
cd parsing
python serve.py -m ../model_weight/MonkeyOCRv2-B-Parsing -p 8888
# Show help messages
python serve.py -h
```

#### 4. Inference
You can parse documents using CLI or serve with demo and FastAPI.

##### 4.1 Parse using CLI
Parse a single document or a directory containing PDFs or images:
```bash
cd parsing
python parse.py \
    -i ../images_test \
    -o output/test \
    -s http://127.0.0.1:8888 \
    --draw-layout \
    --skip-processed
# Show help messages
python parse.py -h
```

##### 4.2 Serve with Web Demo
```bash
cd parsing
python demo/gradio_demo.py -s http://127.0.0.1:8888 -p 8891
# Show help messages
python demo/gradio_demo.py -h
```
You can access the web demo at http://localhost:8891.

##### 4.3 Serve with FastAPI
```bash
cd parsing
python fastapi/main.py -s http://127.0.0.1:8888 -p 8000
# Show help messages
python fastapi/main.py -h
```
You can access the API documentation at http://localhost:8000/docs to explore available endpoints.

### Document Understanding
#### 1. Install
See install part of MonkeyOCRv2.

#### 2. Download Model Weights
Download our model from Huggingface.
```bash
python download_model.py -n MonkeyOCRv2-B-Und # or MonkeyOCRv2-S-Und
```
You can also download our model from ModelScope.

```bash
pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-B-Und # or MonkeyOCRv2-S-Und
```
#### 3. Inference
```bash
cd understanding
python infer.py \
    -m ../model_weight/MonkeyOCRv2-B-Und \
    -i ../images_test/vqa.png \
    -q 'What is the serving size?'
# Show help messages
python infer.py -h
```

## Visualization

Our model supports robust document parsing in real-world scenarios across 17 languages, including Simplified Chinese (ZH), Traditional Chinese (ZH-T), English (EN), Arabic (AR), German (DE), Spanish (ES), French (FR), Hindi (HI), Indonesian (ID), Italian (IT), Japanese (JP), Korean (KO), Dutch (NL), Portuguese (PT), Russian (RU), Thai (TH), and Vietnamese (VI).

<p align="center">
  <img src="asserts/Visualization.gif?raw=true" width="600"/>
</p>

## Evaluation Results

#### 1. Text recognition results on Common Benchmarks, Union14M-Benchmark, OST, and Chinese Benchmarks. We follow the training and evaluation protocols of [OpenOCR](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md).
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2"><strong>Overall</strong></th>
      <th style="text-align: center;" colspan="8">Union14M-Benchmark</th>
      <th style="text-align: center;" colspan="5">Chinese Benchmarks</th>
      <th style="text-align: center;" rowspan="2">Occlusion SceneText</th>
    </tr>
    <tr>
      <th><strong>Avg</th>
      <th>Artistic</th>
      <th>Context less</th>
      <th>Curve</th>
      <th>General</th>
      <th>Multi Oriented</th>
      <th>Multi Words </th>
      <th>Saliency</th>
      <th><strong>Avg</th>
      <th>Scene</th>
      <th>Web</th>
      <th>Document</th>
      <th>Hand writing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ABINet</td>
      <td>73.7</td>
      <td>75.7</td><td>71.7</td><td>74.7</td><td>80.4</td><td>79.8</td><td>69.0</td><td>76.8</td><td>77.6</td>
      <td>70.3</td><td>66.6</td><td>63.2</td><td>98.2</td><td>53.1</td>
      <td>75.0</td>
    </tr>
    <tr>
      <td>MAERec</td>
      <td>81.6</td>
      <td>85.2</td><td>79.0</td><td>84.2</td><td>89.1</td><td>84.6</td><td>87.1</td><td>85.9</td><td>86.3</td>
      <td>83.1</td><td>84.4</td><td>83.0</td><td><b>99.5</b></td><td>65.6</td>
      <td>76.4</td>
    </tr>
    <tr>
      <td>CPPD</td>
      <td>80.4</td>
      <td>81.9</td><td>76.5</td><td>82.9</td><td>86.2</td><td>83.5</td><td>78.7</td><td>81.9</td><td>83.5</td>
      <td>81.7</td><td>82.7</td><td>82.4</td><td>99.4</td><td>62.3</td>
      <td>79.6</td>
    </tr>
    <tr>
      <td>IGTR-AR</td>
      <td>81.0</td>
      <td>84.9</td><td>77.0</td><td>82.4</td><td>90.4</td><td>84.4</td><td>91.2</td><td>84.0</td><td>84.7</td>
      <td>81.7</td><td>82.0</td><td>81.7</td><td><b>99.5</b></td><td>63.8</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td>SMTR</td>
      <td>80.4</td>
      <td>85.0</td><td>76.8</td><td>83.9</td><td>89.1</td><td>83.7</td><td>87.7</td><td><b>89.3</b></td><td>84.6</td>
      <td>82.7</td><td>83.4</td><td>83.0</td><td>99.3</td><td>65.1</td>
      <td>73.5</td>
    </tr>
    <tr>
      <td>SVTRv2</td>
      <td>83.1</td>
      <td>86.1</td><td><b>79.3</b></td><td>86.1</td><td>90.6</td><td>85.1</td><td>89.0</td><td>86.7</td><td>86.2</td>
      <td>83.3</td><td>83.5</td><td><b>83.3</b></td><td><b>99.5</b></td><td>67.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <td colspan="16">&nbsp;</td>
    </tr>
    <tr>
      <td>CRNN (ResNet)</td>
      <td>58.7</td>
      <td>49.2</td><td>51.2</td><td>62.3</td><td>48.1</td><td>68.2</td><td>13.0</td><td>60.4</td><td>41.4</td>
      <td>68.8</td><td>63.8</td><td>68.2</td><td>97.0</td><td>46.1</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>CRNN (MonkeyOCRv2-S)</td>
      <td>67.3</td>
      <td>65.2</td><td>63.7</td><td>73.0</td><td>71.1</td><td>74.5</td><td>28.6</td><td>72.1</td><td>73.4</td>
      <td>74.2</td><td>73.0</td><td>74.9</td><td>96.9</td><td>51.8</td>
      <td>62.4</td>
    </tr>
    <tr>
      <td>PARSeq (ViT)</td>
      <td>82.2</td>
      <td>84.3</td><td>76.5</td><td>83.4</td><td>87.6</td><td>84.9</td><td>88.8</td><td>84.3</td><td>84.4</td>
      <td>82.4</td><td>84.2</td><td>82.8</td><td><b>99.5</b></td><td>63.0</td>
      <td>79.9</td>
    </tr>
    <tr>
      <td>PARSeq (MonkeyOCRv2-S)</td>
      <td><b>84.3</b></td>
      <td><b>87.6</b></td><td>78.6</td><td><b>86.4</b></td><td><b>92.1</b></td><td><b>85.4</b></td><td><b>93.9</b></td><td>88.7</td><td><b>87.7</b></td>
      <td><b>83.7</b></td><td><b>84.6</b></td><td>83.2</td><td><b>99.5</b></td><td><b>67.3</b></td>
      <td><b>81.5</b></td>
    </tr>
  </tbody>
</table>


#### 2. Formula recognition results on [OmniDocBench 1.6](https://github.com/opendatalab/OmniDocBench), [MathWriting](https://arxiv.org/pdf/2404.10690), and [UniMER-Test](https://github.com/opendatalab/unimernet).

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Params</th>
      <th style="text-align: center;" colspan="2">Overall</th>
      <th style="text-align: center;" colspan="2">OmniDocBench 1.6</th>
      <th style="text-align: center;" colspan="2">MathWriting</th>
      <th style="text-align: center;" colspan="2">SPE</th>
      <th style="text-align: center;" colspan="2">CPE</th>
      <th style="text-align: center;" colspan="2">HWE</th>
      <th style="text-align: center;" colspan="2">SCE</th>
    </tr>
    <tr>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
      <th>CDM</th>
      <th>ExpRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pix2tex</td>
      <td>25.5M</td>
      <td>53.8</td>
      <td>23.3</td>
      <td>69.4</td><td>27.0</td>
      <td>0.4</td><td>0.0</td>
      <td>96.2</td><td>72.4</td>
      <td>64.9</td><td>7.1</td>
      <td>24.5</td><td>0.6</td>
      <td>67.6</td><td>32.8</td>
    </tr>
    <tr>
      <td>Texify</td>
      <td>312M</td>
      <td>67.3</td>
      <td>40.4</td>
      <td>76.5</td><td>46.4</td>
      <td>26.6</td><td>2.0</td>
      <td>98.5</td><td>91.0</td>
      <td>70.4</td><td>28.2</td>
      <td>52.7</td><td>23.6</td>
      <td>79.3</td><td>51.3</td>
    </tr>
    <tr>
      <td>UniMERNet-B</td>
      <td>325M</td>
      <td>89.5</td>
      <td>64.5</td>
      <td>90.4</td><td>59.5</td>
      <td>63.8</td><td>12.3</td>
      <td>99.1</td><td>93.3</td>
      <td>96.0</td><td><b>80.5</b></td>
      <td>94.0</td><td>64.3</td>
      <td>93.7</td><td>77.0</td>
    </tr>
    <tr>
      <td>UniMERNet-S</td>
      <td>202M</td>
      <td>89.8</td>
      <td>64.0</td>
      <td>90.1</td><td>59.1</td>
      <td>65.9</td><td>12.7</td>
      <td>99.1</td><td>93.4</td>
      <td>95.9</td><td>77.7</td>
      <td>93.7</td><td>63.9</td>
      <td><b>94.1</b></td><td>76.9</td>
    </tr>
    <tr>
      <td colspan="16">&nbsp;</td>
    </tr>
    <tr>
      <td>UniMERNet-T (Swin)</td>
      <td>107M</td>
      <td>89.4</td>
      <td>61.8</td>
      <td>89.9</td><td>57.2</td>
      <td>65.6</td><td>12.9</td>
      <td>99.1</td><td>92.3</td>
      <td>94.9</td><td>69.9</td>
      <td>93.3</td><td>61.9</td>
      <td>93.8</td><td>76.6</td>
    </tr>
    <tr>
      <td><b>UniMERNet-T (MonkeyOCRv2-S)</b></td>
      <td>110M</td>
      <td><b>90.9</b></td>
      <td><b>66.4</b></td>
      <td><b>90.8</b></td><td><b>61.1</b></td>
      <td><b>70.8</b></td><td><b>16.2</b></td>
      <td><b>99.2</b></td><td><b>93.8</b></td>
      <td><b>96.1</b></td><td>79.2</td>
      <td><b>94.3</b></td><td><b>69.5</b></td>
      <td>94.0</td><td><b>78.6</b></td>
    </tr>
  </tbody>
</table>
</table>


#### 3. Text detection results on Total-Text, CTW1500, ICDAR2015 and ArT. We follow the training and evaluation protocols of [MMOCR](https://github.com/open-mmlab/mmocr) and [DPText-DETR](https://github.com/ymy-k/DPText-DETR).

<p align="center">
  <img src="asserts/text_detection.png?raw=true" width="600"/>
</p>


#### 4. Document tampering detection results on [DocTamper](https://github.com/qcf-568/DocTamper) benchmark.
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Params</th>
      <th style="text-align: center;" colspan="2">Overall</th>
      <th style="text-align: center;" colspan="4">DocTamper-Test</th>
      <th style="text-align: center;" colspan="4">DocTamper-FCD</th>
      <th style="text-align: center;" colspan="4">DocTamper-SCD</th>
    </tr>
    <tr>
      <th style="text-align: center;">IoU</th>
      <th style="text-align: center;">F</th>
      <th style="text-align: center;">IoU</th>
      <th style="text-align: center;">P</th>
      <th style="text-align: center;">R</th>
      <th style="text-align: center;">F</th>
      <th style="text-align: center;">IoU</th>
      <th style="text-align: center;">P</th>
      <th style="text-align: center;">R</th>
      <th style="text-align: center;">F</th>
      <th style="text-align: center;">IoU</th>
      <th style="text-align: center;">P</th>
      <th style="text-align: center;">R</th>
      <th style="text-align: center;">F</th>
    </tr>
  </thead>
  <tbody>
<tr>
  <td>PSCC-Net</td>
  <td>5M</td>
  <td>13.7</td><td>31.3</td><td>17.0</td><td>25.0</td><td>83.0</td><td>39.0</td>
  <td>13.0</td><td>19.0</td><td>82.0</td><td>30.0</td>
  <td>11.0</td><td>15.0</td><td><b>83.0</b></td><td>25.0</td>
</tr>

<tr>
  <td>UperNet</td>
  <td>67M</td>
  <td>49.3</td><td>54.0</td><td>70.0</td><td>66.0</td><td>60.0</td><td>62.0</td>
  <td>30.0</td><td>57.0</td><td>35.0</td><td>43.0</td>
  <td>48.0</td><td>57.0</td><td>58.0</td><td>57.0</td>
</tr>

<tr>
  <td>CAT-Net</td>
  <td>114M</td>
  <td>67.3</td><td>71.0</td><td>78.0</td><td>75.0</td><td>69.0</td><td>72.0</td>
  <td>66.0</td><td>85.0</td><td>70.0</td><td>76.0</td>
  <td>58.0</td><td>65.0</td><td>65.0</td><td>65.0</td>
</tr>

<tr>
  <td>Swin-UPer</td>
  <td>81M</td>
  <td>66.7</td><td>71.7</td><td>79.0</td><td>75.0</td><td>72.0</td><td>73.0</td>
  <td>64.0</td><td>80.0</td><td>70.0</td><td>75.0</td>
  <td>57.0</td><td>66.0</td><td>68.0</td><td>67.0</td>
</tr>

<tr>
  <td>SegFormer</td>
  <td>85M</td>
  <td>70.3</td><td>74.0</td><td>81.0</td><td>77.0</td><td>74.0</td><td>75.0</td>
  <td>69.0</td><td>82.0</td><td>74.0</td><td>78.0</td>
  <td>61.0</td><td>68.0</td><td>70.0</td><td>69.0</td>
</tr>

<tr>
  <td>Mask2Former</td>
  <td>69M</td>
  <td>69.7</td><td>78.0</td><td>84.0</td><td>82.0</td><td>83.0</td><td>82.0</td>
  <td>66.0</td><td>81.0</td><td>75.0</td><td>78.0</td>
  <td>59.0</td><td>70.0</td><td>79.0</td><td>74.0</td>
</tr>

<tr>
  <td>ConvNext</td>
  <td>122M</td>
  <td>69.7</td><td>75.3</td><td>84.0</td><td>81.0</td><td>78.0</td><td>79.0</td>
  <td>62.0</td><td>76.0</td><td>71.0</td><td>74.0</td>
  <td>63.0</td><td>71.0</td><td>74.0</td><td>73.0</td>
</tr>

<tr>
  <td>ConvNextV2</td>
  <td>121M</td>
  <td>72.7</td><td>77.7</td><td>86.0</td><td>82.0</td><td>79.0</td><td>81.0</td>
  <td>65.0</td><td>79.0</td><td>75.0</td><td>77.0</td>
  <td>67.0</td><td>74.0</td><td>76.0</td><td>75.0</td>
</tr>

<tr>
  <td>InternImage</td>
  <td>128M</td>
  <td>73.3</td><td>77.7</td><td>84.0</td><td>81.0</td><td>77.0</td><td>79.0</td>
  <td>72.0</td><td>83.0</td><td>79.0</td><td>81.0</td>
  <td>64.0</td><td>73.0</td><td>74.0</td><td>73.0</td>
</tr>

<tr>
  <td>ASC-Former</td>
  <td>80M</td>
  <td>68.2</td><td>80.8</td><td>81.5</td><td>91.8</td><td>87.8</td><td>89.8</td>
  <td>61.3</td><td>74.9</td><td>77.1</td><td>76.0</td>
  <td>61.9</td><td>78.0</td><td>75.0</td><td>76.5</td>
</tr>

<tr>
  <td>DTD</td>
  <td>66M</td>
  <td><u>77.0</u></td><td>79.7</td><td>84.0</td><td>81.0</td><td>77.0</td><td>79.0</td>
  <td>79.0</td><td>88.0</td><td>82.0</td><td>85.0</td>
  <td><b>68.0</b></td><td>75.0</td><td>76.0</td><td>75.0</td>
</tr>

<tr>
  <td colspan="14">&nbsp;</td>
</tr>

<tr>
  <td>FFDN* (ViTAEv2)</td>
  <td>69M</td>
  <td>70.7</td><td><u>82.7</u></td>
  <td>69.4</td><td>76.2</td><td>88.7</td><td>82.0</td>
  <td>79.0</td><td><b>92.5</b></td><td>84.4</td><td>88.3</td>
  <td>63.6</td><td>79.1</td><td>76.5</td><td>77.8</td>
</tr>

<tr>
  <td><b>FFDN (MonkeyOCRv2-AS)</b></td>
  <td>71M</td>
  <td><b>78.2</b></td><td><b>87.5</b></td>
  <td><b>87.4</b></td><td><b>94.8</b></td><td><b>91.8</b></td><td><b>93.3</b></td>
  <td><b>79.9</b></td><td>90.4</td><td><b>87.4</b></td><td><b>88.9</b></td>
  <td>67.2</td><td><b>81.0</b></td><td>79.8</td><td><b>80.4</b></td>
</tr>
</tbody>
</table>
<p><small>* denotes models trained with the ViTAEv2 pretrained by DeepSolo</small></p>


#### 5. Overlapping text segmentation results on [MOT](https://github.com/willpat1213/MOTS) dataset.
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th><b>mIoU<sub>Text</sub></b></th>
      <th>IoU<sub>Occ</sub></th>
      <th>IoU<sub>Occd</sub></th>
      <th>IoU<sub>Ov</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Unet</td>
      <td>62.2</td>
      <td>80.2</td>
      <td>65.7</td>
      <td>40.7</td>
    </tr>
    <tr>
      <td>Deeplab v3</td>
      <td>67.9</td>
      <td>83.2</td>
      <td>71.2</td>
      <td>49.3</td>
    </tr>
    <tr>
      <td>OCRNet</td>
      <td>65.8</td>
      <td>81.0</td>
      <td>68.5</td>
      <td>47.8</td>
    </tr>
    <tr>
      <td>Segformer</td>
      <td>69.0</td>
      <td>83.6</td>
      <td>74.1</td>
      <td>49.3</td>
    </tr>
    <tr>
      <td>MaskFormer</td>
      <td>68.4</td>
      <td>83.5</td>
      <td>70.3</td>
      <td>51.4</td>
    </tr>
    <tr>
      <td>TexRNet</td>
      <td>68.9</td>
      <td>84.2</td>
      <td>73.2</td>
      <td>49.3</td>
    </tr>
    <tr>
      <td>EAFormer</td>
      <td>69.1</td>
      <td>83.8</td>
      <td>74.2</td>
      <td>50.5</td>
    </tr>
    <tr>
      <td>WASNet</td>
      <td>70.8</td>
      <td>84.8</td>
      <td>74.4</td>
      <td>53.1</td>
    </tr>
    <tr>
      <td colspan="5">&nbsp;</td>
    </tr>
    <tr>
      <td>Mask2Former (ResNet)</td>
      <td>70.3</td>
      <td>84.7</td>
      <td>73.3</td>
      <td>52.8</td>
    </tr>
    <tr>
      <td><b>Mask2Former (MonkeyOCRv2-AS)</b></td>
      <td><u>76.6</u></td>
      <td><b>88.6</b></td>
      <td><b>83.4</b></td>
      <td>57.7</td>
    </tr>
    <tr>
      <td>MOTS (ResNet)</td>
      <td>72.6</td>
      <td>85.2</td>
      <td>77.5</td>
      <td>54.9</td>
    </tr>
    <tr>
      <td><b>MOTS (MonkeyOCRv2-AS)</b></td>
      <td><b>76.9</b></td>
      <td><b>88.6</b></td>
      <td><u>82.6</u></td>
      <td><b>59.4</b></td>
    </tr>
  </tbody>
</table>

#### 6. Document parsing results on [MDPBench](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/MDPBench), a comprehensive multilingual benchmark for real-world document parsing.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Total Params</th>
      <th>ViT</th>
      <th>LLM</th>
      <th>All</th>
      <th>Digit.</th>
      <th>Photo.</th>
      <th>Latin Avg.</th>
      <th>DE</th>
      <th>EN</th>
      <th>ES</th>
      <th>FR</th>
      <th>ID</th>
      <th>IT</th>
      <th>NL</th>
      <th>PT</th>
      <th>VI</th>
      <th>Non-Latin Avg.</th>
      <th>AR</th>
      <th>HI</th>
      <th>JP</th>
      <th>KO</th>
      <th>RU</th>
      <th>TH</th>
      <th>ZH</th>
      <th>ZH-T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="25" class="section">
            <strong>Closed-source VLMs</strong>
        </td>
    </tr>
    <tr>
      <td>ChatGPT-5.2-2025-12-11</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>68.6</td>
      <td>85.6</td>
      <td>63.0</td>
      <td>75.2</td>
      <td>70.8</td>
      <td>79.4</td>
      <td>71.4</td>
      <td>60.0</td>
      <td>77.7</td>
      <td>78.5</td>
      <td>71.6</td>
      <td>85.0</td>
      <td>82.1</td>
      <td>61.1</td>
      <td>64.9</td>
      <td>63.4</td>
      <td>55.8</td>
      <td>65.4</td>
      <td>60.7</td>
      <td>63.8</td>
      <td>56.3</td>
      <td>58.7</td>
    </tr>
    <tr>
      <td>Claude-Sonnet-4.6</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>73.1</td>
      <td>85.0</td>
      <td>69.3</td>
      <td>79.2</td>
      <td>79.8</td>
      <td>80.6</td>
      <td>72.8</td>
      <td>66.5</td>
      <td>82.3</td>
      <td>83.3</td>
      <td>76.7</td>
      <td>88.0</td>
      <td>83.1</td>
      <td>66.2</td>
      <td>67.8</td>
      <td>71.7</td>
      <td>63.4</td>
      <td>64.3</td>
      <td>70.8</td>
      <td>65.2</td>
      <td>61.3</td>
      <td>65.1</td>
    </tr>
    <tr>
      <td>Doubao-2.0-pro</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>74.2</td>
      <td>78.9</td>
      <td>72.8</td>
      <td>75.7</td>
      <td>82.8</td>
      <td>74.4</td>
      <td>69.0</td>
      <td>70.0</td>
      <td>73.3</td>
      <td>82.0</td>
      <td>69.9</td>
      <td>83.4</td>
      <td>76.5</td>
      <td>72.5</td>
      <td>81.3</td>
      <td>75.7</td>
      <td>65.8</td>
      <td>74.7</td>
      <td>63.3</td>
      <td>71.9</td>
      <td>71.9</td>
      <td>75.2</td>
    </tr>
    <tr>
      <td>Gemini-3-pro</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><strong>86.4</strong></td>
      <td><strong>90.4</strong></td>
      <td><strong>85.1</strong></td>
      <td><strong>88.4</strong></td>
      <td><strong>91.2</strong></td>
      <td><strong>90.6</strong></td>
      <td><strong>83.4</strong></td>
      <td><strong>82.7</strong></td>
      <td><strong>91.5</strong></td>
      <td><strong>91.6</strong></td>
      <td><strong>87.7</strong></td>
      <td><strong>91.4</strong></td>
      <td><strong>85.9<strong></td>
      <td><strong>84.1</strong></td>
      <td><strong>89.4<strong></td>
      <td><strong>90.4</strong></td>
      <td><strong>74.8<strong></td>
      <td><strong>85.5<strong></td>
      <td><strong>84.9</strong></td>
      <td><strong>80.6<strong></td>
      <td><strong>85.1</strong></td>
      <td><strong>82.1</strong></td>
    </tr>
    <tr>
        <td colspan="25" class="section">
            <strong>Open-source VLMs</strong>
        </td>
    </tr>
    <tr>
<td>InternVL-3.5-8B</td><td>8.3B</td><td>0.3B</td><td>8B</td><td>42.7</td><td>59.7</td><td>37.0</td><td>53.4</td><td>39.8</td><td>64.2</td><td>47.5</td><td>42.7</td><td>53.8</td><td>60.6</td><td>52.2</td><td>63.2</td><td>57.0</td><td>30.6</td><td>8.2</td><td>9.0</td><td>45.6</td><td>30.3</td><td>26.1</td><td>10.8</td><td>55.3</td><td>59.3</td>
</tr>
<tr>
<td>MinerU-2.5</td><td>1.2B</td><td>0.7B</td><td>0.5B</td><td>46.3</td><td>61.9</td><td>40.8</td><td>63.0</td><td>68.8</td><td>78.4</td><td>54.7</td><td>57.3</td><td>67.5</td><td>75.2</td><td>60.4</td><td>58.8</td><td>46.0</td><td>27.4</td><td>1.3</td><td>9.0</td><td>39.1</td><td>14.7</td><td>8.6</td><td>11.3</td><td>72.9</td><td>62.2</td>
</tr>
<tr>
<td>DeepSeek-OCR</td><td>3.4B</td><td>0.4B</td><td>3B</td><td>51.8</td><td>80.7</td><td>42.2</td><td>54.5</td><td>55.0</td><td>58.3</td><td>44.1</td><td>43.2</td><td>60.9</td><td>69.3</td><td>52.4</td><td>53.0</td><td>54.1</td><td>48.9</td><td>56.9</td><td>52.2</td><td>49.1</td><td>28.2</td><td>36.2</td><td>49.4</td><td>59.7</td><td>59.2</td>
</tr>
<tr>
<td>MonkeyOCR-pro-3B</td><td>3.7B</td><td>0.7B</td><td>3B</td><td>52.2</td><td>68.0</td><td>47.0</td><td>65.1</td><td>71.7</td><td>77.9</td><td>55.9</td><td>62.1</td><td>66.2</td><td>74.5</td><td>66.3</td><td>71.1</td><td>40.2</td><td>37.6</td><td>4.6</td><td>4.2</td><td>55.2</td><td>60.5</td><td>42.6</td><td>9.1</td><td>72.2</td><td>52.4</td>
</tr>
<tr>
<td>Nanonets-OCR-s</td><td>4.7B</td><td>0.7B</td><td>4B</td><td>63.7</td><td>78.8</td><td>58.7</td><td>71.3</td><td>75.1</td><td>78.5</td><td>61.2</td><td>62.5</td><td>70.3</td><td>81.0</td><td>69.6</td><td>75.9</td><td>67.5</td><td>55.0</td><td>59.5</td><td>61.8</td><td>55.9</td><td>51.2</td><td>43.5</td><td>39.5</td><td>67.4</td><td>61.5</td>
</tr>
<tr>
<td>Nanonets-OCR2-3B</td><td>3.7B</td><td>0.7B</td><td>3B</td><td>64.2</td><td>79.2</td><td>59.3</td><td>71.4</td><td>76.7</td><td>76.4</td><td>61.8</td><td>66.1</td><td>68.4</td><td>78.5</td><td>74.1</td><td>74.2</td><td>66.0</td><td>56.2</td><td>60.2</td><td>59.2</td><td>52.1</td><td>54.7</td><td>45.5</td><td>44.6</td><td>68.3</td><td>65.1</td>
</tr>
<tr>
<td>Qwen3.5-Instruct-9B</td><td>9.7B</td><td>0.7B</td><td>9B</td><td>65.7</td><td>74.8</td><td>62.7</td><td>72.5</td><td>72.8</td><td>72.0</td><td>72.0</td><td>64.4</td><td>66.2</td><td>77.6</td><td>74.5</td><td>79.1</td><td>74.0</td><td>58.2</td><td>53.4</td><td>56.2</td><td>55.7</td><td>60.3</td><td>54.7</td><td>56.7</td><td>60.8</td><td>67.5</td>
</tr>
<tr>
<td>GLM-OCR</td><td>0.9B</td><td>0.4B</td><td>0.5B</td><td>67.3</td><td>77.9</td><td>63.7</td><td>78.7</td><td>82.7</td><td>84.5</td><td><u>75.8</u></td><td>76.2</td><td>79.7</td><td>82.8</td><td>80.2</td><td>77.4</td><td>69.2</td><td>54.3</td><td>21.7</td><td>39.6</td><td>65.5</td><td>61.2</td><td>64.2</td><td>27.4</td><td>78.5</td><td>76.7</td>
</tr>
<tr>
<td>Qwen3-VL-Instruct-8B</td><td>8.3B</td><td>0.3B</td><td>8B</td><td>68.3</td><td>78.4</td><td>65.0</td><td>73.6</td><td>73.7</td><td>71.4</td><td>69.3</td><td>66.2</td><td>68.5</td><td>79.1</td><td>78.3</td><td>82.2</td><td>73.4</td><td>62.5</td><td>63.1</td><td>58.4</td><td>59.9</td><td>61.9</td><td>57.9</td><td>62.0</td><td>62.6</td><td>73.8</td>
</tr>
<tr>
<td>HunyuanOCR</td><td>1B</td><td>0.4B</td><td>0.6B</td><td>68.3</td><td>80.2</td><td>64.3</td><td>72.4</td><td>75.0</td><td>73.1</td><td>63.0</td><td>66.1</td><td>69.9</td><td>80.3</td><td>61.4</td><td>81.9</td><td>80.6</td><td>63.7</td><td>68.3</td><td>73.1</td><td>55.6</td><td>68.9</td><td>52.2</td><td>60.7</td><td>66.8</td><td>64.2</td>
</tr>
<tr>
<td>PaddleOCR-VL</td><td>0.9B</td><td>0.6B</td><td>0.3B</td><td>69.6</td><td>87.6</td><td>63.6</td><td>72.1</td><td>78.2</td><td>79.3</td><td>62.9</td><td>66.0</td><td>77.4</td><td>78.4</td><td>67.9</td><td>72.0</td><td>66.6</td><td>66.7</td><td>65.8</td><td>68.4</td><td>59.9</td><td>77.8</td><td>56.9</td><td>57.8</td><td>78.2</td><td>68.5</td>
</tr>
<tr>
<td>olmOCR2</td><td>7.7B</td><td>0.7B</td><td>7B</td><td>70.4</td><td>79.9</td><td>67.2</td><td>76.7</td><td>75.7</td><td>77.3</td><td>72.5</td><td>68.9</td><td>70.6</td><td>81.0</td><td>72.0</td><td><u>88.0</u></td><td>84.0</td><td>63.3</td><td>59.0</td><td>60.8</td><td>59.4</td><td>70.6</td><td>65.8</td><td>59.2</td><td>68.6</td><td>63.4</td>
</tr>
<tr>
<td>MinerU-2.5-Pro</td><td>1.2B</td><td>0.7B</td><td>0.5B</td><td>71.0</td><td>86.2</td><td>66.1</td><td>74.6</td><td>78.3</td><td>79.5</td><td>63.4</td><td>67.4</td><td>78.0</td><td>79.7</td><td>72.1</td><td>78.6</td><td>74.2</td><td>67.0</td><td>56.6</td><td>72.2</td><td>59.1</td><td>77.6</td><td>62.6</td><td>61.8</td><td>76.5</td><td>69.7</td>
</tr>
<tr>
<td>PaddleOCR-VL-1.6</td><td>0.9B</td><td>0.6B</td><td>0.3B</td><td>75.0</td><td>82.8</td><td>72.6</td><td>78.0</td><td>84.1</td><td>79.7</td><td>69.2</td><td>74.8</td><td>81.6</td><td>82.0</td><td>74.7</td><td>76.4</td><td>79.3</td><td>71.6</td><td>69.4</td><td>65.6</td><td>68.7</td><td>82.5</td><td>70.7</td><td>62.3</td><td>78.0</td><td>75.7</td>
</tr>
<tr>
<td>HunyuanOCR-1.5</td><td>1B</td><td>0.4B</td><td>0.6B</td><td>76.8</td><td>86.2</td><td>73.6</td><td>79.7</td><td>79.6</td><td>80.4</td><td>74.2</td><td>70.0</td><td>81.5</td><td>84.5</td><td>78.4</td><td>86.4</td><td>82.4</td><td>73.5</td><td>71.8</td><td>71.6</td><td>65.5</td><td>75.7</td><td>67.4</td><td>77.7</td><td>80.8</td><td>77.2</td>
</tr>
<tr>
<td>Kimi-K2.5</td><td>1T</td><td>0.4B</td><td>1T</td><td>77.5</td><td>85.0</td><td>75.0</td><td>81.6</td><td>85.9</td><td>86.2</td><td>72.7</td><td>71.0</td><td>80.6</td><td>86.6</td><td>77.4</td><td>87.6</td><td><strong>86.2</strong></td><td>72.9</td><td>75.8</td><td>74.5</td><td><u>72.5</u></td><td>70.9</td><td>61.8</td><td>67.0</td><td>81.7</td><td><u>78.6</u></td>
</tr>
<tr>
<td>PaddleOCR-VL-1.5</td><td>0.9B</td><td>0.6B</td><td>0.3B</td><td>78.3</td><td>87.4</td><td>75.2</td><td>81.2</td><td>84.8</td><td>83.0</td><td>75.7</td><td><u>78.1</u></td><td>83.9</td><td>85.2</td><td>80.6</td><td>80.2</td><td>78.9</td><td>74.9</td><td>71.3</td><td>67.7</td><td>69.5</td><td>86.0</td><td>76.0</td><td>68.4</td><td><strong>84.8</strong></td><td>75.7</td>
</tr>
<tr>
<td>chandra-ocr-2</td><td>5.3B</td><td>0.5B</td><td>4.8B</td><td>79.7</td><td>87.8</td><td>77.1</td><td>82.7</td><td>86.6</td><td><u>86.5</u></td><td>69.7</td><td>70.3</td><td>84.6</td><td>87.4</td><td>82.7</td><td><strong>90.7</strong></td><td><u>85.6</u></td><td>76.4</td><td>78.2</td><td>81.1</td><td>68.8</td><td>80.3</td><td>74.0</td><td>78.5</td><td>73.8</td><td>76.3</td>
</tr>
<tr>
<td>dots.mocr</td><td>3B</td><td>1.2B</td><td>1.8B</td><td>80.5</td><td><strong>90.5</strong></td><td>77.2</td><td>81.7</td><td>82.6</td><td><strong>87.4</strong></td><td>71.3</td><td>70.1</td><td>84.5</td><td><strong>89.3</strong></td><td>83.2</td><td>86.8</td><td>79.9</td><td>79.2</td><td>83.3</td><td>83.6</td><td><strong>75.0</strong></td><td>78.7</td><td>71.2</td><td>77.9</td><td><u>84.6</u></td><td><strong>79.6</strong></td>
</tr>
<tr>
<td><strong>MonkeyOCRv2-S-Parsing<a href="https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing">🤗</a></strong></td><td>0.6B</td><td>0.03B</td><td>0.6B</td><td><u>82.5</u></td><td>87.9</td><td><u>80.7</u></td><td><u>83.2</u></td><td><u>87.3</u></td><td>83.6</td><td><strong>76.8</strong></td><td>73.6</td><td><u>85.4</u></td><td>87.2</td><td><u>85.5</u></td><td>87.4</td><td>81.9</td><td><u>81.7</u></td><td><strong>91.2</strong></td><td><u>87.1</u></td><td>69.9</td><td><strong>88.7</strong></td><td><u>78.0</u></td><td><u>79.8</u></td><td>84.4</td><td>74.7</td>
</tr>
<tr>
<td><strong>MonkeyOCRv2-B-Parsing<a href="https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing">🤗</a><strong></td><td>0.7B</td><td>0.1B</td><td>0.6B</td><td><strong>83.3</strong></td><td><u>88.1</u></td><td><strong>81.7</strong></td><td><strong>84.2</strong></td><td><strong>87.7</strong></td><td>84.5</td><td>75.2</td><td><strong>78.4</strong></td><td><strong>86.5</strong></td><td><u>88.6</u></td><td><strong>86.1</strong></td><td>87.9</td><td>83.2</td><td><strong>82.1</strong></td><td><u>90.7</u></td><td><strong>87.2</strong></td><td>71.9</td><td><u>87.6</u></td><td><strong>80.1</strong></td><td><strong>80.8</strong></td><td>83.6</td><td>75.3</td>
</tr>
  </tbody>
</table>


#### 7. Document understanding performance comparison across different vision foundation models. The evaluation benchmarks are selected following [TextMonkey](https://arxiv.org/pdf/2403.04473) and [DT-VQA](https://arxiv.org/pdf/2405.06706).
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Params</th>
            <th>Overall</th>
            <th>DocVQA</th>
            <th>InfoVQA</th>
            <th>DF</th>
            <th>KLC</th>
            <th>WTQ</th>
            <th>ChartQA</th>
            <th>DT-VQA</th>
            <th>OCRBench</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CLIP-B</td>
            <td>86M</td>
            <td>16.0</td>
            <td>20.1</td>
            <td>24.2</td>
            <td>2.3</td>
            <td>13.8</td>
            <td>12.8</td>
            <td>22.2</td>
            <td>22.3</td>
            <td>10.6</td>
        </tr>
        <tr>
            <td>SigLIP2-B</td>
            <td>93M</td>
            <td>24.9</td>
            <td>27.0</td>
            <td>23.5</td>
            <td>3.1</td>
            <td>16.7</td>
            <td>17.4</td>
            <td>35.0</td>
            <td>41.5</td>
            <td>35.1</td>
        </tr>
        <tr>
            <td>RADIOv2.5-B</td>
            <td>98M</td>
            <td>37.5</td>
            <td>60.3</td>
            <td>31.2</td>
            <td>29.9</td>
            <td>30.4</td>
            <td>29.7</td>
            <td>51.1</td>
            <td>44.2</td>
            <td>23.1</td>
        </tr>
        <tr>
            <td>OpenVision-B</td>
            <td>87M</td>
            <td>44.0</td>
            <td>63.3</td>
            <td>30.7</td>
            <td>19.8</td>
            <td>33.1</td>
            <td>31.1</td>
            <td>58.3</td>
            <td>62.6</td>
            <td><u>52.9</u></td>
        </tr>
        <tr>
            <td>DINOv3-B</td>
            <td>86M</td>
            <td>16.1</td>
            <td>26.5</td>
            <td>20.8</td>
            <td>5.6</td>
            <td>13.2</td>
            <td>14.0</td>
            <td>28.9</td>
            <td>15.8</td>
            <td>3.9</td>
        </tr>
        <tr>
            <td>SAM-B</td>
            <td>90M</td>
            <td>25.2</td>
            <td>37.8</td>
            <td>22.2</td>
            <td>4.7</td>
            <td>17.5</td>
            <td>17.6</td>
            <td>46.5</td>
            <td>33.3</td>
            <td>21.9</td>
        </tr>
        <tr>
            <td>SAM2-B</td>
            <td>69M</td>
            <td>22.3</td>
            <td>32.5</td>
            <td>21.9</td>
            <td>2.7</td>
            <td>15.8</td>
            <td>16.6</td>
            <td>40.2</td>
            <td>30.3</td>
            <td>18.4</td>
        </tr>
        <tr>
            <td>oCLIP</td>
            <td>24M</td>
            <td>12.4</td>
            <td>14.8</td>
            <td>19.5</td>
            <td>1.4</td>
            <td>7.4</td>
            <td>11.4</td>
            <td>17.9</td>
            <td>19.2</td>
            <td>7.4</td>
        </tr>
        <tr>
            <td>DiT</td>
            <td>86M</td>
            <td>8.9</td>
            <td>11.3</td>
            <td>20.9</td>
            <td>0.9</td>
            <td>5.2</td>
            <td>9.9</td>
            <td>12.0</td>
            <td>9.2</td>
            <td>1.9</td>
        </tr>
        <tr>
            <td><strong>MonkeyOCRv2-S*<a href="https://huggingface.co/zenosai/MonkeyOCRv2-S-Und">🤗Link</a></strong></td>
            <td>28M</td>
            <td><u>55.9</u></td>
            <td><strong>79.3</strong></td>
            <td><u>44.5</u></td>
            <td><u>65.1</u></td>
            <td><u>37.6</u></td>
            <td><u>43.0</u></td>
            <td><strong>62.0</strong></td>
            <td><u>63.1</u></td>
            <td>52.2</td>
        </tr>
        <tr>
            <td><strong>MonkeyOCRv2-B*<a href="https://huggingface.co/zenosai/MonkeyOCRv2-B-Und">🤗Link</a></strong></td>
            <td>113M</td>
            <td><strong>57.2</strong></td>
            <td><strong>79.3</strong></td>
            <td><strong>46.3</strong></td>
            <td><strong>65.8</strong></td>
            <td><strong>38.2</strong></td>
            <td><strong>43.2</strong></td>
            <td><strong>62.0</strong></td>
            <td><strong>64.3</strong></td>
            <td><strong>58.1</strong></td>
        </tr>
    </tbody>
</table>

## MonkeyDoc v2
MonkeyDoc v2 is currently the largest document image pre-training image-text pair dataset, comprising 113 million document images across 17 languages. The open-sourcing of MonkeyDoc v2 is still underway. So far, we have released all 52 million synthetic samples and 41 million real-world samples derived from FinePDF. Additional real-world data from other sources is currently being organized and will be released progressively.

### Expert Model Labeling Toolchain

We adopt a multi-expert labeling pipeline to obtain reliable annotations for documents. The pipeline includes the following steps:
1. **Structure Detection**  
   We use **dots.mocr** for document structure detection and reading-order prediction. The detected regions, including text blocks, tables, formulas, and other layout elements, are cropped from the original page image for subsequent recognition.
2. **Content Recognition**  
   Each cropped block is independently recognized by three expert models: **dots.mocr**, **PaddleOCR-VL**, and **Qwen3-VL**. These complementary models provide multiple annotations for the same block, reducing reliance on any single OCR system.
3. **Block-Level Agreement Filtering**  
   We compare the recognition results from the three expert models and filter out blocks with low agreement. For retained blocks, we select the prediction that has the highest average agreement with the other two predictions as the final block-level annotation.
4. **Page-Level Quality Control**  
   Pages containing any filtered block are discarded. In addition, we use **Qwen3** to verify whether the predicted reading order is reasonable, and **Qwen3-VL** to check whether document regions are missed during structure detection.
This multi-expert agreement strategy reduces model-specific annotation errors and improves the reliability of the generated annotations.

### References
- **dots.mocr**: https://github.com/rednote-hilab/dots.mocr
- **PaddleOCR-VL**: https://github.com/PaddlePaddle/PaddleOCR
- **Qwen3-VL**: https://github.com/QwenLM/Qwen3-VL
- **Qwen3**: https://github.com/QwenLM/Qwen3
- **MonkeyOCR**: https://github.com/Yuliang-Liu/MonkeyOCR
- **MDPBench**: https://github.com/Yuliang-Liu/MultimodalOCR
- **MonkeyDoc**: https://modelscope.cn/datasets/zenosai/MonkeyDoc
- **FinePDF**: https://huggingface.co/datasets/HuggingFaceFW/finepdfs

## Citation
If you use any part of this release — the MonkeyOCRv2 encoders, MonkeyOCRv2-Parsing,
MonkeyOCRv2-Und, the MDPBench benchmark, or the MonkeyDoc v2 dataset — please cite:

```bibtex
@article{liu2026monkeyocrv2,
  title   = {MonkeyOCRv2: A Visual-Text Foundation Model for Document AI},
  author  = {Liu, Yuliang and Li, Zhang and Zhang, Ziyang and Zhang, Shuo and
             Liu, Qiang and Song, Jiajun and Guo, Zidun and Wang, Xinhan and
             Zheng, Handong and Liu, Yang and Luo, Dongliang and Ma, Zhiyin and
             Zhang, Jiarui and Bai, Xiang},
  journal = {arXiv preprint arXiv:2607.11562},
  year    = {2026}
}
```

## Copyright
We warmly welcome your feedback, suggestions, and contributions, which are essential to the continued development and improvement of our framework. The models are  are released under the [Apache License 2.0](LICENSE) and are free for both research and commercial use. For any questions, please contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn.

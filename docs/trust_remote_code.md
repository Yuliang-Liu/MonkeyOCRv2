# What `trust_remote_code=True` loads

MonkeyOCRv2 is not registered in the Transformers library yet. Loading a
checkpoint with `AutoModel`, `AutoModelForCausalLM`, or `AutoProcessor`
therefore requires `trust_remote_code=True`, which imports the `.py` files that
ship in that Hugging Face snapshot (config, processor, and modeling classes).

Those files are part of the model repo, same as `config.json`. After you
download a snapshot to disk, `from_pretrained("/local/path", trust_remote_code=True)`
imports the local copies. It does not open extra network connections at load
time unless the path you pass is a Hub repo id.

A Transformers-native registration (no `trust_remote_code`) is not available
today. That would need a coordinated change in Transformers upstream and is
out of scope for this page.

Hub commit SHAs below were read from each repo's `main` ref on 2026-08-25.
Re-query the Hub if you need a newer snapshot. Every file link is pinned to
that SHA.

## Where this repo sets the flag

| Call site | Loader | Checkpoints |
| --- | --- | --- |
| README backbone snippet, `vision/extract_feature.py` | `AutoModel`, `AutoImageProcessor` | S, B |
| `vision/extract_feature_vitae.py` | `AutoModel`, `AutoImageProcessor` | AS |
| `understanding/infer.py` | `AutoModelForCausalLM`, `AutoProcessor` | S-Und, B-Und |
| `parsing/cpu/core_runner.py` | `AutoModelForCausalLM`, `AutoProcessor` | S-Parsing, B-Parsing |
| `parsing/serve.py` | vLLM `serve --trust-remote-code` | S-Parsing, B-Parsing, optional DFlash draft |
| `understanding/serve.py` | vLLM `serve --trust-remote-code` | S-Und, B-Und |

Recognition and detection training load the encoder checkpoints the same way
(`trust_remote_code=True` on `AutoModel`).

`parsing/serve.py` first imports this GitHub repo's `parsing/modeling/` modules
and registers `MonkeyOCRv2ForCausalLM` with vLLM. It still passes
`--trust-remote-code` so Hugging Face can load the snapshot's custom config and
processor modules.

`understanding/serve.py` first imports this GitHub repo's
`understanding/modeling/` adapters: vLLM 0.11.x loads
`modeling_monkeyocrv2und_vllm_011.py`, and vLLM ≥ 0.25 loads
`modeling_monkeyocrv2und_vllm.py`. It still passes `--trust-remote-code` so
Hugging Face can load the Und snapshot's custom config and processor modules
(listed under Document understanding below). Those local adapters are not the
`.py` files that ship in the Hugging Face snapshot.

## Custom Python files

Transformers executes the modules named in each snapshot's `auto_map`, plus
whatever those modules import. Parsing and Und `modeling_monkeyocrv2.py` import
`modeling_monkeyocrv2_vision.py`, so that vision file runs even though it is not
an `auto_map` entry of its own.

Quick Start vision scripts also pass `trust_remote_code=True` to
`AutoImageProcessor`. Preprocessor configs set `image_processor_type` to
in-tree `Qwen2VLImageProcessor`. Encoder `auto_map` still names a Processor
class: on S/B that class is commented out in `configuration_monkeyocrv2vit.py`;
on AS it is `MonkeyOCRv2ViTAEProcessor`.

### Vision encoders

S and B ship the same two Python files (byte-identical at the revisions below).
Weights differ.

**[zenosai/MonkeyOCRv2-S](https://huggingface.co/zenosai/MonkeyOCRv2-S/tree/8f10bf04773e5690f64b44137340910929996918)**
(`8f10bf04773e5690f64b44137340910929996918`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2vit.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S/blob/8f10bf04773e5690f64b44137340910929996918/configuration_monkeyocrv2vit.py) | `AutoConfig` → `MonkeyOCRv2VisionConfig` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S/blob/8f10bf04773e5690f64b44137340910929996918/modeling_monkeyocrv2_vision.py) | `AutoModel` → `MonkeyOCRv2VisionTransformer` |

SHA-256:

```
f2ac1fd2c5b2ecc772f57e19381c0274510044800b88b4b74b40b271db2a1dc5  configuration_monkeyocrv2vit.py
db095ab35ccf068d059e2282b983e29694de1dd2ad588beb61aa7dd09de0d160  modeling_monkeyocrv2_vision.py
```

**[zenosai/MonkeyOCRv2-B](https://huggingface.co/zenosai/MonkeyOCRv2-B/tree/aa2385cb1c8c4d2e84c3cf816f428ab19afd7261)**
(`aa2385cb1c8c4d2e84c3cf816f428ab19afd7261`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2vit.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B/blob/aa2385cb1c8c4d2e84c3cf816f428ab19afd7261/configuration_monkeyocrv2vit.py) | `AutoConfig` → `MonkeyOCRv2VisionConfig` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B/blob/aa2385cb1c8c4d2e84c3cf816f428ab19afd7261/modeling_monkeyocrv2_vision.py) | `AutoModel` → `MonkeyOCRv2VisionTransformer` |

SHA-256: same as MonkeyOCRv2-S above.

**[zenosai/MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS/tree/240c1a813531e7fb1f85460cbe4e9dda4cbe84a7)**
(`240c1a813531e7fb1f85460cbe4e9dda4cbe84a7`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2_vitae.py`](https://huggingface.co/zenosai/MonkeyOCRv2-AS/blob/240c1a813531e7fb1f85460cbe4e9dda4cbe84a7/configuration_monkeyocrv2_vitae.py) | `AutoConfig` → `MonkeyOCRv2ViTAEEncoderConfig`; `AutoProcessor` → `MonkeyOCRv2ViTAEProcessor` |
| [`modeling_monkeyocrv2_vitae_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-AS/blob/240c1a813531e7fb1f85460cbe4e9dda4cbe84a7/modeling_monkeyocrv2_vitae_vision.py) | `AutoModel` → `MonkeyOCRv2ViTAEVisionTransformer` |

SHA-256:

```
251d471e9f8d400c6159aa998295bd27b4a2e514b4e3528da9abb44e8c44595d  configuration_monkeyocrv2_vitae.py
82d7614d14de76a0a6961e9678296ca8695901d378032ea3e6b5c2e22bbc2a31  modeling_monkeyocrv2_vitae_vision.py
```

### Document parsing

S-Parsing and B-Parsing share `configuration_monkeyocrv2.py` and
`modeling_monkeyocrv2_vision.py`. `modeling_monkeyocrv2.py` differs between the
two checkpoints.

**[zenosai/MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing/tree/efd808db05a076124aa54baa7d6a4b50a3a14b8c)**
(`efd808db05a076124aa54baa7d6a4b50a3a14b8c`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing/blob/efd808db05a076124aa54baa7d6a4b50a3a14b8c/configuration_monkeyocrv2.py) | `AutoConfig` → `MonkeyOCRv2Config`; `AutoProcessor` → `MonkeyOCRv2Processor` |
| [`modeling_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing/blob/efd808db05a076124aa54baa7d6a4b50a3a14b8c/modeling_monkeyocrv2.py) | `AutoModelForCausalLM` → `MonkeyOCRv2ForCausalLM` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing/blob/efd808db05a076124aa54baa7d6a4b50a3a14b8c/modeling_monkeyocrv2_vision.py) | imported by `modeling_monkeyocrv2.py` (`MonkeyOCRv2VisionTransformer`) |

SHA-256:

```
c045b7476f1d2278a9953438ca89a0383849cf46ffe286989680b314d006be1e  configuration_monkeyocrv2.py
3969cbe5b79c2f4a7adea2d04bd23d0685a4b22872aa664a9045416535c8a19f  modeling_monkeyocrv2.py
411161a04945e36f60217b72e39b1ccdc2c309a1190c2750f01679a51c0eb3aa  modeling_monkeyocrv2_vision.py
```

**[zenosai/MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing/tree/2419139b7bcd3fda2689b2a83167172afba91c8b)**
(`2419139b7bcd3fda2689b2a83167172afba91c8b`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing/blob/2419139b7bcd3fda2689b2a83167172afba91c8b/configuration_monkeyocrv2.py) | `AutoConfig` → `MonkeyOCRv2Config`; `AutoProcessor` → `MonkeyOCRv2Processor` |
| [`modeling_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing/blob/2419139b7bcd3fda2689b2a83167172afba91c8b/modeling_monkeyocrv2.py) | `AutoModelForCausalLM` → `MonkeyOCRv2ForCausalLM` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing/blob/2419139b7bcd3fda2689b2a83167172afba91c8b/modeling_monkeyocrv2_vision.py) | imported by `modeling_monkeyocrv2.py` (`MonkeyOCRv2VisionTransformer`) |

SHA-256:

```
c045b7476f1d2278a9953438ca89a0383849cf46ffe286989680b314d006be1e  configuration_monkeyocrv2.py
3a9e1f2508c9ccac71f12b080fbaf135264c625ff4191cb06696b1a43793b960  modeling_monkeyocrv2.py
411161a04945e36f60217b72e39b1ccdc2c309a1190c2750f01679a51c0eb3aa  modeling_monkeyocrv2_vision.py
```

**[zenosai/MonkeyOCRv2-B-Parsing-DFlash](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing-DFlash/tree/0ec13149bdcae12176306bffe935a8f73f60d356)**
(`0ec13149bdcae12176306bffe935a8f73f60d356`) has no `.py` files.
`config.json` uses `model_type: qwen3` and
`architectures: ["MonkeyOCRv2ForCausalLM"]`. The draft class is the local
adapter `parsing/modeling/modeling_monkeyocrv2_dflash_vllm.py`. `serve.py`
still passes `--trust-remote-code` because the target Parsing checkpoint needs
it.

### Document understanding

S-Und and B-Und ship the same three Python files (byte-identical at the
revisions below). Weights differ.

**[zenosai/MonkeyOCRv2-S-Und](https://huggingface.co/zenosai/MonkeyOCRv2-S-Und/tree/001868aad7426d50759d3a645e4a2020d6f5fd0b)**
(`001868aad7426d50759d3a645e4a2020d6f5fd0b`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Und/blob/001868aad7426d50759d3a645e4a2020d6f5fd0b/configuration_monkeyocrv2.py) | `AutoConfig` → `MonkeyOCRv2Config`; `AutoProcessor` → `MonkeyOCRv2Processor` |
| [`modeling_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Und/blob/001868aad7426d50759d3a645e4a2020d6f5fd0b/modeling_monkeyocrv2.py) | `AutoModelForCausalLM` → `MonkeyOCRv2ForCausalLM` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-S-Und/blob/001868aad7426d50759d3a645e4a2020d6f5fd0b/modeling_monkeyocrv2_vision.py) | imported by `modeling_monkeyocrv2.py` (`MonkeyOCRv2VisionTransformer`) |

SHA-256:

```
97f7d75111064596c8603285223a97da49f1ef7abc0f5ce0a583a7a2ce2d78f6  configuration_monkeyocrv2.py
d805cf4291e917ea4ab5bc5c51ca812b3287dc96cf94e89d052b6a8a895c4bdd  modeling_monkeyocrv2.py
340badf4d4bc2516756839d61f240c0e53f06a5e1ce62c831680e80a0191c5d2  modeling_monkeyocrv2_vision.py
```

**[zenosai/MonkeyOCRv2-B-Und](https://huggingface.co/zenosai/MonkeyOCRv2-B-Und/tree/2b3bd145c8627ef5ecfbb15099332c4de59fcc3f)**
(`2b3bd145c8627ef5ecfbb15099332c4de59fcc3f`)

| File | Role |
| --- | --- |
| [`configuration_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Und/blob/2b3bd145c8627ef5ecfbb15099332c4de59fcc3f/configuration_monkeyocrv2.py) | `AutoConfig` → `MonkeyOCRv2Config`; `AutoProcessor` → `MonkeyOCRv2Processor` |
| [`modeling_monkeyocrv2.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Und/blob/2b3bd145c8627ef5ecfbb15099332c4de59fcc3f/modeling_monkeyocrv2.py) | `AutoModelForCausalLM` → `MonkeyOCRv2ForCausalLM` |
| [`modeling_monkeyocrv2_vision.py`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Und/blob/2b3bd145c8627ef5ecfbb15099332c4de59fcc3f/modeling_monkeyocrv2_vision.py) | imported by `modeling_monkeyocrv2.py` (`MonkeyOCRv2VisionTransformer`) |

SHA-256: same as MonkeyOCRv2-S-Und above.

## Pin a revision

Pass `revision="<commit-sha>"` so a later Hub push cannot change the code you
reviewed:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained(
    "zenosai/MonkeyOCRv2-B",
    trust_remote_code=True,
    revision="aa2385cb1c8c4d2e84c3cf816f428ab19afd7261",
    dtype="auto",
    device_map="auto",
)
```

Download a pinned snapshot, then load from the local directory (no Hub id at
load time):

```python
from huggingface_hub import snapshot_download
from transformers import AutoModel

local_dir = snapshot_download(
    repo_id="zenosai/MonkeyOCRv2-B",
    revision="aa2385cb1c8c4d2e84c3cf816f428ab19afd7261",
    local_dir="model_weight/MonkeyOCRv2-B",
)

encoder = AutoModel.from_pretrained(
    local_dir,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```

`download_model.py` does not pass `revision=` today. Use `snapshot_download` as
above, or set `HF_HUB_OFFLINE=1` after the files are already on disk.

For vLLM serving, pin the GitHub commit of this repository as well: the
`--trust-remote-code` snapshot files listed above are separate from
`parsing/modeling/*.py`, which are imported from the local clone.

## Check hashes

After a local snapshot:

```bash
cd model_weight/MonkeyOCRv2-B
sha256sum configuration_monkeyocrv2vit.py modeling_monkeyocrv2_vision.py
```

Compare the output to the SHA-256 values in the tables. Custom code is stored
as ordinary Git blobs, not LFS, so `sha256sum` on the `.py` files is the right
check. Weight files (`.safetensors`) are LFS objects with their own SHA-256 in
the pointer.

To hash a file without downloading weights:

```bash
curl -sL \
  https://huggingface.co/zenosai/MonkeyOCRv2-B/resolve/aa2385cb1c8c4d2e84c3cf816f428ab19afd7261/modeling_monkeyocrv2_vision.py \
  | sha256sum
```

import modeling_monkeyocrv2_vllm

import os
import json
import time
import torch
import base64
import argparse
import requests
from io import BytesIO
from pathlib import Path
from typing import Union
from vllm import LLM, SamplingParams
from PIL import Image, ImageFile, ImageDraw

from modeling_preprocessor import Preprocessor

ALL_PROMPT = {
    "Caption": "Please output the text content from the image.",
    # "Footnote": "Please output the text content from the image.",
    "List-item": "Please output the text content from the image.",
    "Page-footer": "Please output the text content from the image.",
    "Page-header": "Please output the text content from the image.",
    "Section-header": "Please output the text content from the image.",
    "Text": "Please output the text content from the image.",
    "Title": "Please output the text content from the image.",
    "Formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "Table": "Please extract the table from the image and represent it in OTSL format.",
    "LAYOUT": "Please output the categories and coordinates of the document elements in reading order.",
    "END2END": "List the document elements in reading order, including their categories, coordinates, and the content of each element.",
}


def image_to_png_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_picture_block(image: Image.Image, image_dir: Path, doc_name: str, sub_idx: int) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{doc_name}_sub{sub_idx}.jpg"
    image.convert("RGB").save(image_dir / image_name, format="JPEG", quality=95)
    return f"../images/{image_name}"


class MonkeyOCRv2_Parsing:
    def __init__(self, model_path, tp=1):
        self.model_name = os.path.basename(model_path)
        self.pipe = LLM(model=model_path,
                        mm_processor_kwargs={'use_fast': True},
                        max_model_len=16384,
                        gpu_memory_utilization=self._auto_gpu_mem_ratio(0.5),
                        tensor_parallel_size=tp,
                        trust_remote_code=True,
                        )
        self.gen_config = SamplingParams(max_tokens=10000,temperature=0)
    
    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio

    def batch_inference(
        self,
        images,
        questions,
        min_pixels=None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ):
        prompts = [
            ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n") for question in questions
        ]
        max_pixels = int(os.getenv("MOCR2_MAX_PIXELS")) if os.getenv("MOCR2_MAX_PIXELS") else None
        inputs = [{
            "prompt": prompts[i],
            "multi_modal_data": {
                "image": load_image(images[i], max_pixels=max_pixels, min_pixels=min_pixels),
            }
        } for i in range(len(prompts))]
        gen_config = self.gen_config.clone()
        if max_tokens is not None:
            gen_config.max_tokens = max_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        outputs = self.pipe.generate(inputs, sampling_params=gen_config)
        return [o.outputs[0].text for o in outputs]
    

def get_layout(model: MonkeyOCRv2_Parsing, images: list[Image.Image]):
    outputs = model.batch_inference(images, [ALL_PROMPT["LAYOUT"]] * len(images), min_pixels=1003520)

    def _safe_eval(text: str):
        return eval(text, {"__builtins__": {}}, {})

    def _normalize_item(item):
        if not isinstance(item, dict):
            return None
        if "bbox" not in item or "label" not in item:
            return None
        bbox = item["bbox"]
        label = item["label"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            bbox = [float(v) for v in bbox]
        except Exception:
            return None
        if not isinstance(label, str):
            label = str(label)
        return {"bbox": bbox, "label": label}

    def _normalize_list(obj):
        if not isinstance(obj, list):
            return []
        out = []
        for x in obj:
            nx = _normalize_item(x)
            if nx is not None:
                out.append(nx)
        return out

    def _extract_balanced_blocks(text: str, lch: str, rch: str):
        res = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == lch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == rch and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    res.append(text[start:i + 1])
                    start = -1
        return res

    def _dedup_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _extract_tolerant_list_blocks(text: str):
        blocks = _extract_balanced_blocks(text, "[", "]")
        first = text.find("[")
        if first != -1:
            tail = text[first:].strip()
            if tail:
                lcnt, rcnt = tail.count("["), tail.count("]")
                if lcnt > rcnt:
                    tail = tail + ("]" * (lcnt - rcnt))
                blocks.append(tail)
        return _dedup_keep_order(blocks)

    def _extract_tolerant_dict_blocks(text: str):
        blocks = _extract_balanced_blocks(text, "{", "}")
        n = len(text)
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            end = None
            for j in range(i, n):
                cj = text[j]
                if cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                # Truncated output: add missing closing braces.
                blk = text[i:] + ("}" * max(depth, 1))
            else:
                blk = text[i:end]
            blocks.append(blk)
        return _dedup_keep_order(blocks)

    def _parse_one_output(text: str):
        text = (text or "").strip()
        if not text:
            return []

        try:
            obj = _safe_eval(text)
            full = _normalize_list(obj)
            if full:
                return full
        except Exception:
            pass

        best = []

        # 1) Recover list-level output, including truncated lists.
        for blk in _extract_tolerant_list_blocks(text):
            try:
                obj = _safe_eval(blk)
                cur = _normalize_list(obj)
                if len(cur) > len(best):
                    best = cur
            except Exception:
                continue

        # 2) Recover dict-level output and append items one by one.
        dict_items = []
        for blk in _extract_tolerant_dict_blocks(text):
            try:
                obj = _safe_eval(blk)
                nobj = _normalize_item(obj)
                if nobj is not None:
                    dict_items.append(nobj)
            except Exception:
                continue
        if len(dict_items) > len(best):
            best = dict_items

        return best

    def _map_bbox_to_image(bbox, w, h):
        x1, y1, x2, y2 = bbox
        # Model coordinates are normalized to 0-1000.
        x1 = x1 / 1000.0 * w
        x2 = x2 / 1000.0 * w
        y1 = y1 / 1000.0 * h
        y2 = y2 / 1000.0 * h

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1 = max(0, min(int(round(x1)), w - 1 if w > 0 else 0))
        y1 = max(0, min(int(round(y1)), h - 1 if h > 0 else 0))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        return [x1, y1, x2, y2]

    page_layouts = []
    for i, out in enumerate(outputs):
        parsed = _parse_one_output(out)
        w, h = images[i].size
        mapped = []
        for item in parsed:
            mapped.append({
                "bbox": _map_bbox_to_image(item["bbox"], w, h),
                "label": item["label"]
            })
        page_layouts.append(mapped)

    return page_layouts

import re
from html import escape
def otsl_to_html(otsl_str):
    if not otsl_str or not otsl_str.strip():
        return "<table></table>"
    
    rows_tokens = otsl_str.split("<nl>")
    if rows_tokens and rows_tokens[-1] == "":
        rows_tokens.pop()
    
    grid = []
    
    for r_idx, row_str in enumerate(rows_tokens):
        if not row_str.strip():
            if r_idx >= len(grid):
                grid.append([])
            continue
        
        parts = re.findall(r'<([a-z]+)>(.*?)(?=<[a-z]+>|$)', row_str)
        
        if r_idx >= len(grid):
            grid.append([])
        
        col_idx = 0
        
        for tag, content in parts:
            while True:
                while len(grid[r_idx]) <= col_idx:
                    grid[r_idx].append(None)
                
                if grid[r_idx][col_idx] is not None:
                    col_idx += 1
                else:
                    break
            
            if tag == 'fcel' or tag == 'ecel':
                text = content.strip() if tag == 'fcel' else ""
                grid[r_idx][col_idx] = {
                    'text': text,
                    'rowspan': 1,
                    'colspan': 1,
                    'valid': True
                }
                col_idx += 1
                
            elif tag == 'lcel':
                search_c = col_idx - 1
                found = False
                while search_c >= 0:
                    if len(grid[r_idx]) > search_c:
                        cell = grid[r_idx][search_c]
                        if cell and cell.get('valid'):
                            cell['colspan'] += 1
                            found = True
                            break
                    search_c -= 1
                
                if found:
                    grid[r_idx][col_idx] = {'valid': False, 'type': 'lcel'}
                else:
                    grid[r_idx][col_idx] = {
                        'text': '',
                        'rowspan': 1,
                        'colspan': 1,
                        'valid': True
                    }
                col_idx += 1
                
            elif tag == 'ucel':
                search_r = r_idx - 1
                found = False
                while search_r >= 0:
                    if len(grid[search_r]) > col_idx:
                        cell = grid[search_r][col_idx]
                        if cell and cell.get('valid'):
                            cell['rowspan'] += 1
                            found = True
                            break
                    search_r -= 1
                
                if found:
                    grid[r_idx][col_idx] = {'valid': False, 'type': 'ucel'}
                else:
                    grid[r_idx][col_idx] = {
                        'text': '',
                        'rowspan': 1,
                        'colspan': 1,
                        'valid': True
                    }
                col_idx += 1
                
            elif tag == 'xcel':
                grid[r_idx][col_idx] = {'valid': False, 'type': 'xcel'}
                col_idx += 1
            else:
                col_idx += 1
    
    html_parts = ['<table>']
    
    for row in grid:
        html_parts.append('<tr>')
        for cell in row:
            if cell is None:
                continue
            elif cell.get('valid'):
                attrs = []
                if cell['rowspan'] > 1:
                    attrs.append(f'rowspan="{cell["rowspan"]}"')
                if cell['colspan'] > 1:
                    attrs.append(f'colspan="{cell["colspan"]}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                text = escape(cell['text'])
                html_parts.append(f'<td{attr_str}>{text}</td>')
        html_parts.append('</tr>')
    
    html_parts.append('</table>')
    return ''.join(html_parts)


def process_formula(content: str):
    content = content.strip('$').strip()
    # Collapse repeated \quad sequences (>=5).
    content = re.sub(
        r'(?:\\quad\s*){5,}',
        r'\\quad ',
        content
    )

    # Collapse repeated \qquad sequences (>=5).
    content = re.sub(
        r'(?:\\qquad\s*){5,}',
        r'\\qquad ',
        content
    ).strip()

    # Extract trailing (xxx). TODO: remove tag{}.
    match = re.search(
        r'(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$'
        r'|\\tag\{([^{}]*)\}\s*$',
        content
    )

    extracted = None
    if match:
        extracted = match.group(1)
        content = content[:match.start()].rstrip()

    begin_env = None
    has_end = False

    # Detect leading \begin{xx}.
    begin_match = re.match(r'^\\begin\{([^\}]+)\}', content)
    if begin_match:
        begin_env = begin_match.group(1)
        # Remove leading \begin{xx}.
        content = content[begin_match.end():].lstrip()

        # Detect whether the matching \end{xx} is at the end.
        end_pattern = rf'\\end\{{{re.escape(begin_env)}\}}\s*$'
        end_match = re.search(end_pattern, content)
        if end_match:
            has_end = True
            # Remove trailing \end{xx}.
            content = content[:end_match.start()].rstrip()


    # Extract trailing (xxx).
    match = re.search(
        r'(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$'
        r'|\\tag\{([^{}]*)\}\s*$',
        content
    )

    if match:
        extracted = match.group(1)
        content = content[:match.start()].rstrip()

    # ===== Restore begin/end =====

    if begin_env:
        content = f"\\begin{{{begin_env}}}\n{content}\n\\end{{{begin_env}}}"

    return content, extracted


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
):
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]

    for seq_len in range(1, window_size // 2 + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))

        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue

        while pos >= 0:
            if predicted_tokens[pos:pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break

        if repeat_count > max_repeats:
            return True

    return False


def _should_retry_repeat_output(raw: str) -> bool:
    raw = raw or ""
    return detect_repeat_token(raw) or (
        len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50)
    )


def batch_inference_with_repeat_retry(
    model: MonkeyOCRv2_Parsing,
    infer_images: list[Image.Image],
    infer_questions: list[str],
    max_tokens: int | None = 5000,
    max_retries: int | None = None,
) -> list[str]:
    if not infer_images:
        return []
    if max_retries is None:
        max_retries = int(os.getenv("MOCR2_REC_MAX_RETRIES", "3"))

    outputs = model.batch_inference(infer_images, infer_questions, max_tokens=max_tokens)
    retry_indices = [i for i, raw in enumerate(outputs) if _should_retry_repeat_output(raw)]

    retries = 0
    while retry_indices and retries < max_retries:
        retry_temperature = min(0.2 * (retries + 1), 0.8)
        print(
            f"Detected repeat token in {len(retry_indices)} outputs, "
            f"retrying batch (attempt {retries + 1})..."
        )
        retry_images = [infer_images[i] for i in retry_indices]
        retry_questions = [infer_questions[i] for i in retry_indices]
        retry_outputs = model.batch_inference(
            retry_images,
            retry_questions,
            max_tokens=max_tokens,
            temperature=retry_temperature,
            top_p=0.95,
        )

        next_retry_indices = []
        for src_idx, raw in zip(retry_indices, retry_outputs):
            outputs[src_idx] = raw
            if _should_retry_repeat_output(raw):
                next_retry_indices.append(src_idx)
        retry_indices = next_retry_indices
        retries += 1

    return outputs


def parse_images(
    model: MonkeyOCRv2_Parsing,
    images: list[Image.Image],
    pdf_pages=None,
    return_layouts: bool = False,
    doc_names: list[str] | None = None,
    use_base64: bool = True,
    image_dir: Path | None = None,
    enable_repeat_retry: bool = False,
    repeat_retry_max_retries: int | None = None,
):
    assert pdf_pages is None or sum(pdf_pages) == len(images)
    pdf_pages = pdf_pages or [1] * len(images)

    layouts_per_page = get_layout(model, images)

    page_to_pdf = []
    page_num_in_pdf = []
    for pdf_idx, pnum in enumerate(pdf_pages):
        for p in range(pnum):
            page_to_pdf.append(pdf_idx)
            page_num_in_pdf.append(p + 1)

    tasks = []
    for page_idx, items in enumerate(layouts_per_page):
        img = images[page_idx]
        w, h = img.size
        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            x1 = max(0, min(x1, w - 1 if w > 0 else 0))
            y1 = max(0, min(y1, h - 1 if h > 0 else 0))
            x2 = max(x1 + 1, min(int(round(x2)), w))
            y2 = max(y1 + 1, min(int(round(y2)), h))
            label = item["label"]
            crop = img.crop((x1, y1, x2, y2))
            tasks.append({
                "image": crop,
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "question": ALL_PROMPT.get(label, ""),
                "need_infer": label in ALL_PROMPT,
                "page_idx": page_idx,
                "page_num": page_num_in_pdf[page_idx],
            })

    infer_indices = [i for i, t in enumerate(tasks) if t["need_infer"]]
    infer_images = [tasks[i]["image"] for i in infer_indices]
    infer_questions = [tasks[i]["question"] for i in infer_indices]

    if infer_indices and enable_repeat_retry:
        infer_outputs = batch_inference_with_repeat_retry(
            model,
            infer_images,
            infer_questions,
            max_tokens=5000,
            max_retries=repeat_retry_max_retries,
        )
    else:
        infer_outputs = model.batch_inference(infer_images, infer_questions, max_tokens=5000) if infer_indices else []

    raw_outputs = [""] * len(tasks)
    for k, t_idx in enumerate(infer_indices):
        raw_outputs[t_idx] = infer_outputs[k]

    page_results = [[] for _ in images]
    picture_counts = [0] * len(pdf_pages)
    for t, raw in zip(tasks, raw_outputs):
        label = t["label"]
        content = (raw or "").strip()
        if label == "Formula":
            content, extracted = process_formula(content)
            content = "$$\n" + content + "\n$$\n"
            if extracted:
                content = content + extracted
        elif label == "Table":
            content = content if os.getenv('MOCR2_TABLE_HTML', '0') == "1" else otsl_to_html(content)
        elif label == "Picture":
            if use_base64:
                image_ref = image_to_png_data_uri(t["image"])
            else:
                if image_dir is None:
                    raise ValueError("image_dir is required when use_base64 is False")
                doc_idx = page_to_pdf[t["page_idx"]]
                doc_name = doc_names[doc_idx] if doc_names else f"doc_{doc_idx}"
                sub_idx = picture_counts[doc_idx]
                picture_counts[doc_idx] += 1
                image_ref = save_picture_block(t["image"], image_dir, doc_name, sub_idx)
            content = f"![image]({image_ref})"
        elif label == "Title":
            content = "# " + content.replace("\n", "\n# ")
        elif label == "Section-header":
            content = "## " + content.replace("\n", "\n## ")
        elif not t["need_infer"]:
            content = ""
        rec = {
            "bbox": t["bbox"],
            "label": label,
            "content": content,
            "page_num": t["page_num"],
        }
        page_results[t["page_idx"]].append(rec)

    result = [[] for _ in pdf_pages]
    for page_idx, recs in enumerate(page_results):
        result[page_to_pdf[page_idx]].extend(recs)

    if return_layouts:
        return result, layouts_per_page
    return result


def draw_layout_pdf(images: list[Image.Image], layouts_per_page: list[list[dict]], save_pdf_path: str):
    vis_pages = []
    for img, items in zip(images, layouts_per_page):
        canvas = img.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for i, it in enumerate(items):
            x1, y1, x2, y2 = it["bbox"]
            label = it.get("label", "")
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            ty = max(0, y1 - 12)
            draw.text((x1, ty), str(i)+': '+label, fill=(255, 0, 0))
        vis_pages.append(canvas)

    if not vis_pages:
        return
    os.makedirs(os.path.dirname(save_pdf_path), exist_ok=True)
    vis_pages[0].save(save_pdf_path, "PDF", resolution=100.0, save_all=True, append_images=vis_pages[1:])

def result2md(
    names: list[str],
    results: list[list[dict]],
    save_dir: str | None = None,
    keep_header_footer: bool = False,
):
    md_list = []
    out_dir = None
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, pdf_items in enumerate(results):
        lines = []
        for item in pdf_items:
            if not keep_header_footer and item.get("label") in {"Page-header", "Page-footer"}:
                continue
            content = (item.get("content") or "").strip()
            if content:
                lines.append(content)

        md = "\n\n".join(lines).strip() + ("\n" if lines else "")
        md = md.replace("�", '') # Remove invalid characters
        md_list.append(md)

        if out_dir is not None:
            (out_dir / f"{names[i]}.md").write_text(md, encoding="utf-8")

    return md_list


def load_pdf_images(pdf_path: str):
    try:
        import pypdfium2 as pdfium
    except Exception as e:
        raise ImportError("Reading PDF files requires pypdfium2") from e

    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bmp = page.render(scale=200/72)
        pil = bmp.to_pil().convert("RGB")
        pages.append(pil)
    return pages


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: Union[str, Image.Image], max_pixels: int = None, min_pixels: int = None, max_size: int = None, min_size: int = None, resize: int = None) -> Image.Image:
    """load image from url, local path or openai GPT4V."""
    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if isinstance(image_url, Image.Image):
            img = image_url
        elif image_url.startswith('http'):
            response = requests.get(image_url, headers=headers, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        elif image_url.startswith('data:image'):
            img = load_image_from_base64(image_url.split(',')[1])
        else:
            # Load image from local path
            img = Image.open(image_url)

        # check image valid
        img = img.convert('RGB')
        if resize:
            img = img.resize([img.size[0]*2,img.size[1]*2], Image.LANCZOS)

        # resize image if too small
        if min_pixels and img.size[0] * img.size[1] < min_pixels:
            scale = (min_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        if min_size and min(img.size) < min_size:
            scale = min_size / min(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # resize image if too large
        if max_pixels and img.size[0] * img.size[1] > max_pixels:
            scale = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        elif max_size and max(img.size) > max_size:
            scale = max_size / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        if max(img.size[0], img.size[1]) / min(img.size[0], img.size[1]) > 200:
            img = Image.new('RGB', (32, 32))
    except Exception as error:
        if isinstance(image_url, str) and len(image_url) > 100:
            image_url = image_url[:100] + ' ...'
        print(f'--------{error}, image_url={image_url}')
        # use dummy image
        img = Image.new('RGB', (32, 32))

    return img


def _load_input_documents(input_path: str):
    p = Path(input_path)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    docs = []

    files = [p] if p.is_file() else sorted([x for x in p.iterdir() if x.is_file()])
    for f in files:
        ext = f.suffix.lower()
        if ext == ".pdf":
            pages = load_pdf_images(str(f))
            docs.append({
                "name": f.stem,
                "image_name": f.name,
                "image_path": str(f),
                "images": pages,
                "pdf_pages": len(pages),
            })
        elif ext in image_ext:
            docs.append({
                "name": f.stem,
                "image_name": f.name,
                "image_path": str(f),
                "images": [load_image(str(f))],
                "pdf_pages": 1,
            })
    return docs


def _doc_image_size(images: list[Image.Image]):
    sizes = [[int(img.size[0]), int(img.size[1])] for img in images]
    return sizes[0] if len(sizes) == 1 else sizes


def build_result_record(doc: dict, layouts: list[dict]):
    return {
        "image_name": doc.get("image_name") or f"{doc.get('name', '')}",
        "image_path": doc.get("image_path") or "",
        "image_size": _doc_image_size(doc.get("images", [])),
        "layouts": layouts,
    }


def parse_images_e2e(
    model: MonkeyOCRv2_Parsing,
    images: list[Image.Image],
    pdf_pages=None,
    return_layouts: bool = False,
    enable_repeat_retry: bool = False,
    repeat_retry_max_retries: int | None = None,
):
    assert pdf_pages is None or sum(pdf_pages) == len(images)
    pdf_pages = pdf_pages or [1] * len(images)

    questions = [ALL_PROMPT["END2END"]] * len(images)
    if enable_repeat_retry:
        outputs = batch_inference_with_repeat_retry(
            model,
            images,
            questions,
            max_tokens=None,
            max_retries=repeat_retry_max_retries,
        )
    else:
        outputs = model.batch_inference(images, questions)

    def _safe_eval(text: str):
        return eval(text, {"__builtins__": {}}, {})

    def _normalize_item(item):
        if not isinstance(item, dict):
            return None
        if "bbox" not in item or "label" not in item:
            return None
        bbox = item["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            bbox = [float(v) for v in bbox]
        except Exception:
            return None
        label = item["label"] if isinstance(item["label"], str) else str(item["label"])
        content = item.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        return {"bbox": bbox, "label": label, "content": content}

    def _normalize_list(obj):
        if not isinstance(obj, list):
            return []
        out = []
        for x in obj:
            nx = _normalize_item(x)
            if nx is not None:
                out.append(nx)
        return out

    def _extract_balanced_blocks(text: str, lch: str, rch: str):
        res = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == lch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == rch and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    res.append(text[start:i + 1])
                    start = -1
        return res

    def _dedup_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _extract_tolerant_list_blocks(text: str):
        blocks = _extract_balanced_blocks(text, "[", "]")
        first = text.find("[")
        if first != -1:
            tail = text[first:].strip()
            if tail:
                lcnt, rcnt = tail.count("["), tail.count("]")
                if lcnt > rcnt:
                    tail = tail + ("]" * (lcnt - rcnt))
                blocks.append(tail)
        return _dedup_keep_order(blocks)

    def _extract_tolerant_dict_blocks(text: str):
        blocks = _extract_balanced_blocks(text, "{", "}")
        n = len(text)
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            end = None
            for j in range(i, n):
                cj = text[j]
                if cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                blk = text[i:] + ("}" * max(depth, 1))
            else:
                blk = text[i:end]
            blocks.append(blk)
        return _dedup_keep_order(blocks)

    def _parse_one_output(text: str):
        text = (text or "").strip()
        if not text:
            return []
        try:
            obj = _safe_eval(text)
            full = _normalize_list(obj)
            if full:
                return full
        except Exception:
            pass

        best = []
        for blk in _extract_tolerant_list_blocks(text):
            try:
                cur = _normalize_list(_safe_eval(blk))
                if len(cur) > len(best):
                    best = cur
            except Exception:
                continue

        dict_items = []
        for blk in _extract_tolerant_dict_blocks(text):
            try:
                nobj = _normalize_item(_safe_eval(blk))
                if nobj is not None:
                    dict_items.append(nobj)
            except Exception:
                continue
        if len(dict_items) > len(best):
            best = dict_items
        return best

    def _map_bbox_to_image(bbox, w, h):
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 / 1000.0 * w, x2 / 1000.0 * w
        y1, y2 = y1 / 1000.0 * h, y2 / 1000.0 * h
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = max(0, min(int(round(x1)), w - 1 if w > 0 else 0))
        y1 = max(0, min(int(round(y1)), h - 1 if h > 0 else 0))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        return [x1, y1, x2, y2]

    page_to_pdf, page_num_in_pdf = [], []
    for pdf_idx, pnum in enumerate(pdf_pages):
        for p in range(pnum):
            page_to_pdf.append(pdf_idx)
            page_num_in_pdf.append(p + 1)

    page_results = [[] for _ in images]
    layouts_per_page = []

    for page_idx, out in enumerate(outputs):
        parsed = _parse_one_output(out)
        w, h = images[page_idx].size
        page_layout = []
        for item in parsed:
            bbox = _map_bbox_to_image(item["bbox"], w, h)
            label = item["label"]
            content = (item.get("content") or "").strip()
            page_results[page_idx].append({
                "bbox": bbox,
                "label": label,
                "content": content,
                "page_num": page_num_in_pdf[page_idx],
            })
            page_layout.append({"bbox": bbox, "label": label})
        layouts_per_page.append(page_layout)

    result = [[] for _ in pdf_pages]
    for page_idx, recs in enumerate(page_results):
        result[page_to_pdf[page_idx]].extend(recs)

    if return_layouts:
        return result, layouts_per_page
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", default="../images_test", help="Input file or folder containing PDFs, images, or both")
    parser.add_argument("--model-path", "-m", default="../model_weight/MonkeyOCRv2-B-Parsing", help="Model path")
    parser.add_argument("--group-size", "-g", type=int, default=100, help="Number of documents to process per batch")
    parser.add_argument("--output-path", "-o", default="./output/test", help="Output directory")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--max-pixels", type=int, default=1003520, help="Maximum input image pixels; larger images are resized proportionally")
    parser.add_argument("--draw-layout", action="store_true", help="Save layout visualization PDFs to output_path/layout")
    parser.add_argument("--end2end", action="store_true", help="Enable end-to-end parsing and output bbox/label/content from the full image")
    parser.add_argument("--skip-processed", action="store_true", help="Skip documents whose markdown output already exists")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing and directly parse the original input documents; this may lead to worse accuracy but faster speed")
    parser.add_argument("--retry-repeat", action="store_true", help="Retry recognition outputs when repeated tokens are detected")
    parser.add_argument("--retry-repeat-max-retries", type=int, default=3, help="Maximum retry attempts for repeated-token outputs")
    parser.add_argument("--keep-header-footer", action="store_true", help="Keep Page-header and Page-footer blocks in markdown output; JSON output always keeps them")
    parser.add_argument("--use-base64", "--use_base64", action="store_true", default=False, help="Write Picture blocks as base64 in markdown; by default images are saved to output/images and referenced by relative path")
    args = parser.parse_args()

    if args.max_pixels:
        os.environ["MOCR2_MAX_PIXELS"] = str(args.max_pixels)

    if not args.skip_preprocess:
        preprocessor = Preprocessor(args.model_path, batch_size=32)
    model = MonkeyOCRv2_Parsing(args.model_path, tp=args.tp)

    t2 = time.time()
    docs = _load_input_documents(args.input_path)
    print(f"Loaded {len(docs)} documents from {args.input_path} in {time.time() - t2:.2f} s")
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir = out_dir / "markdowns"
    md_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    if not args.use_base64:
        image_dir.mkdir(parents=True, exist_ok=True)
    if args.draw_layout:
        layout_dir = out_dir / "layouts"
        layout_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_processed:
        before = len(docs)
        docs = [d for d in docs if not (md_dir / f"{d['name']}.md").exists()]
        skipped = before - len(docs)
        print(f"--skip-processed: skipped {skipped} already processed documents, {len(docs)} remaining.")
        if not docs:
            print("All documents already processed. Nothing to do.")
            return

    all_results = []
    all_names = []

    t1 = time.time()
    time_pre = 0
    time_parse = 0
    time_save = 0

    for i in range(0, len(docs), max(1, args.group_size)):
        chunk = docs[i:i + max(1, args.group_size)]
        print(f"Processing documents {i+1} to {i + len(chunk)}: {chunk[0]['name']} {'...' if len(chunk)>1 else ''} {chunk[-1]['name']  if len(chunk)>1 else ''}")
        if not args.skip_preprocess:
            t2 = time.time()
            chunk = preprocessor.preprocess_docs(chunk)
            time_pre += time.time() - t2
        chunk_images = []
        chunk_pdf_pages = []
        chunk_names = []

        for d in chunk:
            chunk_images.extend(d["images"])
            chunk_pdf_pages.append(d["pdf_pages"])
            chunk_names.append(d["name"])

        if args.end2end:
            chunk_results, chunk_layouts = parse_images_e2e(
                model,
                chunk_images,
                chunk_pdf_pages,
                return_layouts=True,
                enable_repeat_retry=args.retry_repeat,
                repeat_retry_max_retries=args.retry_repeat_max_retries,
            )
        else:
            t2 = time.time()
            chunk_results, chunk_layouts = parse_images(
                model,
                chunk_images,
                chunk_pdf_pages,
                return_layouts=True,
                doc_names=chunk_names,
                use_base64=args.use_base64,
                image_dir=image_dir,
                enable_repeat_retry=args.retry_repeat,
                repeat_retry_max_retries=args.retry_repeat_max_retries,
            )
            time_parse += time.time() - t2

        chunk_records = [
            build_result_record(doc, res)
            for doc, res in zip(chunk, chunk_results)
        ]
        all_results.extend(chunk_records)
        all_names.extend(chunk_names)

        if args.draw_layout:
            s = 0
            for d in chunk:
                n = d["pdf_pages"]
                doc_imgs = chunk_images[s:s+n]
                doc_layouts = chunk_layouts[s:s+n]
                draw_layout_pdf(doc_imgs, doc_layouts, str(layout_dir / f"{d['name']}_layout.pdf"))
                s += n

        t2 = time.time()
        for name, record in zip(chunk_names, chunk_records):
            (json_dir / f"{name}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1),
                encoding="utf-8"
            )
        result2md(
            chunk_names,
            chunk_results,
            save_dir=str(md_dir),
            keep_header_footer=args.keep_header_footer,
        )
        time_save += time.time() - t2


    (out_dir / "all_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=1),
        encoding="utf-8"
    )
    time_used = time.time() - t1
    print(f"Preprocessing time: {time_pre:.2f} s, parsing time: {time_parse:.2f} s, saving time: {time_save:.2f} s")
    print(f"Total time used: {time_used:.2f} s / {len(docs)} docs, avg {time_used/len(docs):.2f} s/doc.")
    print(f"Processing completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()

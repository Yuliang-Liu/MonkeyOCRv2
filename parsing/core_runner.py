from modeling import modeling_monkeyocrv2_vllm

import os
import json
import time
import torch
import base64
import requests
import warnings
import zipfile
import threading
import queue
import asyncio
import uuid
import shutil
from requests import exceptions as requests_exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Union
from urllib.parse import urlparse, urlunparse
from vllm import SamplingParams
try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
except Exception:
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
    except Exception:
        AsyncLLMEngine = None
        AsyncEngineArgs = None
from PIL import Image, ImageFile, ImageDraw, ImageOps

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from modeling.modeling_preprocessor import Preprocessor

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


def build_vllm_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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


def save_preprocessed_page(image: Image.Image, preprocessed_dir: Path, doc_name: str, page_idx: int) -> str:
    path = get_preprocessed_page_path(preprocessed_dir, doc_name, page_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")
    return str(path)


def get_preprocessed_page_path(preprocessed_dir: Path, doc_name: str, page_idx: int) -> Path:
    return preprocessed_dir / doc_name / f"page_{page_idx + 1:03}.png"


def normalize_server_url(server_url: str) -> str:
    server_url = (server_url or "").strip().rstrip("/")
    if not server_url:
        return ""
    if "://" not in server_url:
        server_url = "http://" + server_url
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported server URL scheme: {parsed.scheme}. Use http:// or https://.")
    if parsed.scheme == "https" and parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
        warnings.warn(
            f"Server URL {server_url} uses HTTPS for a local vLLM endpoint. "
            "vLLM serve defaults to plain HTTP; using http:// instead.",
            RuntimeWarning,
        )
        parsed = parsed._replace(scheme="http")
        server_url = urlunparse(parsed)
    return server_url.rstrip("/")


class MonkeyOCRv2_ServerParsing:
    def __init__(
        self,
        server_url: str,
        model_name: str = "MonkeyOCRv2",
        timeout: int = 300,
        http_max_retries: int = 5,
        http_retry_backoff: float = 1.0,
    ):
        self.server_url = normalize_server_url(server_url)
        if self.server_url.endswith("/v1"):
            self.api_base = self.server_url
        else:
            self.api_base = self.server_url + "/v1"
        self.model_name = model_name
        self.timeout = timeout
        self.http_max_retries = max(0, int(http_max_retries))
        self.http_retry_backoff = max(0.0, float(http_retry_backoff))
        self.max_inflight = max(1, int(os.getenv("MOCR2_SERVER_MAX_INFLIGHT", "1024")))
        self._inflight = threading.BoundedSemaphore(self.max_inflight)
        self._thread_local = threading.local()

    def _session(self):
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _reset_session(self):
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self._thread_local.session = requests.Session()
        return self._thread_local.session

    def _chat_completion(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        payload = {
            "model": self.model_name,
            "temperature": 0 if temperature is None else temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_png_data_uri(image)}},
                    {"type": "text", "text": question},
                ],
            }],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        url = f"{self.api_base}/chat/completions"
        last_exc = None
        with self._inflight:
            for attempt in range(self.http_max_retries + 1):
                try:
                    resp = self._session().post(
                        url,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        raise requests_exceptions.HTTPError(
                            f"retryable HTTP {resp.status_code}: {resp.text[:500]}",
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                except (
                    requests_exceptions.ConnectionError,
                    requests_exceptions.Timeout,
                    requests_exceptions.ChunkedEncodingError,
                    requests_exceptions.SSLError,
                    requests_exceptions.HTTPError,
                ) as exc:
                    last_exc = exc
                    if isinstance(exc, requests_exceptions.SSLError) or "WRONG_VERSION_NUMBER" in str(exc):
                        raise RuntimeError(
                            f"SSL protocol error when connecting to {url}. "
                            "vLLM serve usually runs plain HTTP, so use "
                            f"{self.api_base.replace('https://', 'http://', 1)} "
                            "instead of an https:// URL unless you configured TLS explicitly."
                        ) from exc
                    response = getattr(exc, "response", None)
                    if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
                        raise
                    self._reset_session()
                    if attempt >= self.http_max_retries:
                        break
                    sleep_s = self.http_retry_backoff * (2 ** attempt)
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 30.0))
        raise last_exc

    def batch_inference(
        self,
        images,
        questions,
        min_pixels=None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        concurrency: int | None = None,
    ):
        if not images:
            return []
        max_pixels = int(os.getenv("MOCR2_MAX_PIXELS")) if os.getenv("MOCR2_MAX_PIXELS") else None
        prepared = [
            load_image(img, max_pixels=max_pixels, min_pixels=min_pixels)
            for img in images
        ]
        if len(prepared) == 1:
            return [self._chat_completion(
                prepared[0],
                questions[0],
                max_tokens,
                temperature,
                top_p,
            )]
        concurrency = max(1, min(int(concurrency or len(prepared)), self.max_inflight))
        outputs = [None] * len(prepared)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    self._chat_completion,
                    prepared[i],
                    questions[i],
                    max_tokens,
                    temperature,
                    top_p,
                ): i
                for i in range(len(prepared))
            }
            for future in as_completed(future_to_idx):
                outputs[future_to_idx[future]] = future.result()
        return outputs


class MonkeyOCRv2_AsyncParsing:
    def __init__(self, model_path: str, tp: int = 1, max_inflight: int = 1024):
        if AsyncLLMEngine is None or AsyncEngineArgs is None:
            raise ImportError("AsyncLLMEngine is unavailable in this vLLM installation.")
        self.model_name = os.path.basename(model_path)
        self.max_inflight = max(1, int(max_inflight))
        self._inflight = threading.BoundedSemaphore(self.max_inflight)
        self.gen_config = SamplingParams(max_tokens=10000, temperature=0)
        self._engine_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "trust_remote_code": True,
            "max_model_len": 16384,
            "gpu_memory_utilization": self._auto_gpu_mem_ratio(0.5),
        }
        self.engine = None
        self._closed = False
        try:
            engine_kwargs = dict(self._engine_kwargs)
            engine_kwargs["mm_processor_kwargs"] = {"use_fast": True}
            AsyncEngineArgs(**engine_kwargs)
            self._engine_kwargs = engine_kwargs
        except TypeError:
            self._engine_kwargs.pop("mm_processor_kwargs", None)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._run_coro(self._init_engine())

    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        return ratio * mem_free / mem_total

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro, timeout: float | None = None):
        if self._closed:
            raise RuntimeError("Async vLLM engine has already been closed.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _init_engine(self):
        engine_args = AsyncEngineArgs(**self._engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _generate_one(
        self,
        image: Image.Image,
        question: str,
        min_pixels=None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        max_pixels = int(os.getenv("MOCR2_MAX_PIXELS")) if os.getenv("MOCR2_MAX_PIXELS") else None
        gen_config = self.gen_config.clone()
        if max_tokens is not None:
            gen_config.max_tokens = max_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        inputs = {
            "prompt": build_vllm_prompt(question),
            "multi_modal_data": {
                "image": load_image(image, max_pixels=max_pixels, min_pixels=min_pixels),
            },
        }
        final_output = None
        if self.engine is None:
            raise RuntimeError("Async vLLM engine is not initialized.")
        async for output in self.engine.generate(inputs, gen_config, request_id=str(uuid.uuid4())):
            final_output = output
        return final_output.outputs[0].text if final_output is not None else ""

    def _infer_one(
        self,
        image: Image.Image,
        question: str,
        min_pixels=None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        with self._inflight:
            return self._run_coro(
                self._generate_one(image, question, min_pixels, max_tokens, temperature, top_p)
            )

    def batch_inference(
        self,
        images,
        questions,
        min_pixels=None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        concurrency: int | None = None,
    ):
        if not images:
            return []
        concurrency = max(1, min(int(concurrency or len(images)), self.max_inflight))
        outputs = [None] * len(images)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    self._infer_one,
                    images[i],
                    questions[i],
                    min_pixels,
                    max_tokens,
                    temperature,
                    top_p,
                ): i
                for i in range(len(images))
            }
            for future in as_completed(future_to_idx):
                outputs[future_to_idx[future]] = future.result()
        return outputs

    async def _shutdown_engine(self):
        engine = self.engine
        self.engine = None
        if engine is None:
            return
        shutdown = getattr(engine, "shutdown", None)
        close = getattr(engine, "close", None)
        if callable(shutdown):
            result = shutdown()
            if asyncio.iscoroutine(result):
                await result
        elif callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result
        else:
            engine_core = getattr(engine, "engine_core", None)
            engine_core_shutdown = getattr(engine_core, "shutdown", None)
            if callable(engine_core_shutdown):
                engine_core_shutdown()

    async def _cancel_loop_tasks(self):
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks(self._loop) if task is not current and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown_engine(), self._loop)
                future.result(timeout=30)
            except Exception as exc:
                warnings.warn(f"Failed to shutdown Async vLLM engine cleanly: {exc}", RuntimeWarning)
            try:
                future = asyncio.run_coroutine_threadsafe(self._cancel_loop_tasks(), self._loop)
                future.result(timeout=10)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if not self._loop.is_closed():
            self._loop.close()
    

def get_layout(model, images: list[Image.Image]):
    outputs = model.batch_inference(
        images,
        [ALL_PROMPT["LAYOUT"]] * len(images),
        min_pixels=1003520,
        max_tokens=4096,
    )

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


def parse_end2end_output(text: str, image_size: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    def _safe_eval(src: str):
        return eval(src, {"__builtins__": {}}, {})

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

    def _extract_balanced_blocks(src: str, lch: str, rch: str):
        res = []
        depth = 0
        start = -1
        for i, ch in enumerate(src):
            if ch == lch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == rch and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    res.append(src[start:i + 1])
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

    def _extract_tolerant_list_blocks(src: str):
        blocks = _extract_balanced_blocks(src, "[", "]")
        first = src.find("[")
        if first != -1:
            tail = src[first:].strip()
            if tail:
                lcnt, rcnt = tail.count("["), tail.count("]")
                if lcnt > rcnt:
                    tail = tail + ("]" * (lcnt - rcnt))
                blocks.append(tail)
        return _dedup_keep_order(blocks)

    def _extract_tolerant_dict_blocks(src: str):
        blocks = _extract_balanced_blocks(src, "{", "}")
        n = len(src)
        for i, ch in enumerate(src):
            if ch != "{":
                continue
            depth = 0
            end = None
            for j in range(i, n):
                cj = src[j]
                if cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            blocks.append(src[i:end] if end is not None else src[i:] + ("}" * max(depth, 1)))
        return _dedup_keep_order(blocks)

    def _parse_items(src: str):
        src = (src or "").strip()
        if not src:
            return []
        try:
            full = _normalize_list(_safe_eval(src))
            if full:
                return full
        except Exception:
            pass
        best = []
        for blk in _extract_tolerant_list_blocks(src):
            try:
                cur = _normalize_list(_safe_eval(blk))
                if len(cur) > len(best):
                    best = cur
            except Exception:
                continue
        dict_items = []
        for blk in _extract_tolerant_dict_blocks(src):
            try:
                nobj = _normalize_item(_safe_eval(blk))
                if nobj is not None:
                    dict_items.append(nobj)
            except Exception:
                continue
        return dict_items if len(dict_items) > len(best) else best

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

    w, h = image_size
    recs = []
    layouts = []
    for block_idx, item in enumerate(_parse_items(text)):
        bbox = _map_bbox_to_image(item["bbox"], w, h)
        label = item["label"]
        recs.append({
            "bbox": bbox,
            "label": label,
            "content": (item.get("content") or "").strip(),
            "_block_idx": block_idx,
        })
        layouts.append({"bbox": bbox, "label": label})
    return recs, layouts

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
    model,
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


def _format_block_content(
    task: dict,
    raw: str,
    page_to_pdf: list[int],
    doc_names: list[str] | None,
    picture_counts: list[int],
    use_base64: bool,
    image_dir: Path | None,
) -> str:
    label = task["label"]
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
            image_ref = image_to_png_data_uri(task["image"])
        else:
            if image_dir is None:
                raise ValueError("image_dir is required when use_base64 is False")
            doc_idx = page_to_pdf[task["page_idx"]]
            doc_name = doc_names[doc_idx] if doc_names else f"doc_{doc_idx}"
            sub_idx = picture_counts[doc_idx]
            picture_counts[doc_idx] += 1
            image_ref = save_picture_block(task["image"], image_dir, doc_name, sub_idx)
        content = f"![image]({image_ref})"
    elif label == "Title":
        content = "# " + content.replace("\n", "\n# ")
    elif label == "Section-header":
        content = "## " + content.replace("\n", "\n## ")
    elif not task["need_infer"]:
        content = ""
    return content


def _recognize_one_block(
    model,
    task: dict,
    enable_repeat_retry: bool,
    repeat_retry_max_retries: int | None,
) -> str:
    if not task["need_infer"]:
        return ""
    raw = model.batch_inference(
        [task["image"]],
        [task["question"]],
        max_tokens=5000,
        concurrency=1,
    )[0]
    if not enable_repeat_retry:
        return raw

    max_retries = repeat_retry_max_retries
    if max_retries is None:
        max_retries = int(os.getenv("MOCR2_REC_MAX_RETRIES", "3"))
    retries = 0
    while _should_retry_repeat_output(raw) and retries < max_retries:
        retry_temperature = min(0.2 * (retries + 1), 0.8)
        raw = model.batch_inference(
            [task["image"]],
            [task["question"]],
            max_tokens=5000,
            temperature=retry_temperature,
            top_p=0.95,
            concurrency=1,
        )[0]
        retries += 1
    return raw


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


def _is_jpeg_source(source) -> bool:
    if source is None:
        return False
    source = str(source).lower()
    return source.endswith((".jpg", ".jpeg")) or ".jpg?" in source or ".jpeg?" in source


def _apply_jpeg_orientation(img: Image.Image, source=None) -> Image.Image:
    if _is_jpeg_source(source) or (getattr(img, "format", None) or "").upper() == "JPEG":
        return ImageOps.exif_transpose(img)
    return img


def open_oriented_image(image_path: Union[str, Path]) -> Image.Image:
    img = Image.open(image_path)
    return _apply_jpeg_orientation(img, image_path)


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: Union[str, Path, Image.Image], max_pixels: int = None, min_pixels: int = None, max_size: int = None, min_size: int = None, resize: int = None) -> Image.Image:
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
            img = _apply_jpeg_orientation(image_url)
        else:
            image_source = str(image_url)
        if isinstance(image_url, Image.Image):
            pass
        elif image_source.startswith('http'):
            response = requests.get(image_url, headers=headers, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = _apply_jpeg_orientation(img, image_source)
        elif image_source.startswith('data:image'):
            img = load_image_from_base64(image_source.split(',')[1])
            img = _apply_jpeg_orientation(img, image_source)
        else:
            # Load image from local path
            img = open_oriented_image(image_source)

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


def _list_input_files(input_path: str):
    p = Path(input_path)
    return [p] if p.is_file() else sorted([x for x in p.iterdir() if x.is_file()])


def _count_pending_documents(input_path: str, md_dir: Path, skip_processed: bool) -> int:
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}
    total = 0
    for f in _list_input_files(input_path):
        if f.suffix.lower() not in image_ext:
            continue
        if skip_processed and (md_dir / f"{f.stem}.md").exists():
            continue
        total += 1
    return total


def _count_pending_pages(input_path: str, md_dir: Path, skip_processed: bool) -> int:
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    total = 0
    for f in _list_input_files(input_path):
        ext = f.suffix.lower()
        if ext not in image_ext and ext != ".pdf":
            continue
        if skip_processed and (md_dir / f"{f.stem}.md").exists():
            continue
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError:
                total += 1
                continue
            pdf = pdfium.PdfDocument(str(f))
            try:
                total += len(pdf)
            finally:
                close = getattr(pdf, "close", None)
                if close is not None:
                    close()
        else:
            total += 1
    return total


def _iter_input_documents(input_path: str):
    p = Path(input_path)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p] if p.is_file() else sorted([x for x in p.iterdir() if x.is_file()])
    for f in files:
        ext = f.suffix.lower()
        if ext == ".pdf":
            pages = load_pdf_images(str(f))
            yield {
                "name": f.stem,
                "image_name": f.name,
                "image_path": str(f),
                "images": pages,
                "pdf_pages": len(pages),
            }
        elif ext in image_ext:
            yield {
                "name": f.stem,
                "image_name": f.name,
                "image_path": str(f),
                "images": [load_image(str(f))],
                "pdf_pages": 1,
            }


def _doc_image_size(images: list[Image.Image]):
    sizes = [[int(img.size[0]), int(img.size[1])] for img in images]
    return sizes[0] if len(sizes) == 1 else sizes


def build_result_record(doc: dict, layouts: list[dict]):
    return {
        "image_name": doc.get("image_name") or f"{doc.get('name', '')}",
        "image_path": doc.get("image_path") or "",
        "image_size": doc.get("image_size") or _doc_image_size(doc.get("images", [])),
        "layouts": layouts,
    }


def run_streaming_pipeline(
    args,
    preprocessor,
    model,
    out_dir: Path,
    json_dir: Path,
    md_dir: Path,
    image_dir: Path,
    show_progress_bar: bool = False,
    verbose: bool = True,
):
    sentinel = object()
    page_window = max(1, int(args.page_max_inflight))
    server_window = max(1, int(args.server_max_inflight))
    preprocess_q = queue.Queue(maxsize=page_window)
    layout_q = queue.Queue(maxsize=page_window)
    rec_q = queue.Queue(maxsize=max(page_window * 8, server_window * 2))
    layout_workers = max(1, min(page_window, server_window))
    rec_workers = max(1, server_window)
    done_q = queue.Queue()
    error_q = queue.Queue()
    stop_event = threading.Event()
    lock = threading.Lock()
    states = {}
    completed_records = []
    stats = {
        "submitted_docs": 0,
        "skipped_docs": 0,
        "submitted_pages": 0,
        "time_pre": 0.0,
        "time_parse_requests": 0.0,
        "parse_started_at": None,
        "parse_finished_at": None,
    }

    layout_dir = out_dir / "layouts" if args.draw_layout else None
    total_docs = _count_pending_documents(args.input_path, md_dir, args.skip_processed)
    pbar = None
    pre_pbar = None
    if show_progress_bar and tqdm is not None and total_docs > 0:
        total_pages = _count_pending_pages(args.input_path, md_dir, args.skip_processed)
        pre_pbar = tqdm(
            total=total_pages,
            dynamic_ncols=True,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}",
            position=0,
            leave=True,
        )
        pbar = tqdm(
            total=total_docs,
            dynamic_ncols=True,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}",
            position=1,
            leave=True,
        )

    def maybe_complete(state):
        if state["pending_pages"] == 0 and state["pending_recs"] == 0 and not state["done"]:
            state["done"] = True
            done_q.put(state["doc_id"])

    def add_page_result(state, page_idx, rec):
        state["page_results"][page_idx].append(rec)

    def mark_parse_started():
        if stats["parse_started_at"] is None:
            stats["parse_started_at"] = time.time()

    def raise_if_worker_failed():
        with error_q.mutex:
            if error_q.queue:
                raise error_q.queue[0]

    def put_checked(q, item):
        while not stop_event.is_set():
            raise_if_worker_failed()
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue
        raise_if_worker_failed()
        raise RuntimeError("Streaming pipeline stopped before item could be queued.")

    def put_sentinels(q, count):
        sent = 0
        while sent < count and not stop_event.is_set():
            raise_if_worker_failed()
            try:
                q.put(sentinel, timeout=0.2)
                sent += 1
            except queue.Full:
                continue

    def get_checked(q):
        while not stop_event.is_set():
            try:
                return q.get(timeout=0.2)
            except queue.Empty:
                raise_if_worker_failed()
                continue
        return sentinel

    def record_worker_error(exc):
        stop_event.set()
        error_q.put(exc)

    def preprocess_worker():
        try:
            batch = []
            while True:
                item = get_checked(preprocess_q)
                if item is sentinel:
                    break
                batch.append(item)
                saw_sentinel = False
                while len(batch) < max(1, args.preprocess_batch_size):
                    if stop_event.is_set():
                        break
                    try:
                        nxt = preprocess_q.get(timeout=0.05)
                    except queue.Empty:
                        break
                    if nxt is sentinel:
                        saw_sentinel = True
                        break
                    batch.append(nxt)

                if not batch:
                    if saw_sentinel:
                        break
                    continue
                t0 = time.time()
                processed = preprocessor.preprocess_images(
                    [x["image"] for x in batch],
                    batch_size=args.preprocess_batch_size,
                )
                with lock:
                    stats["time_pre"] += time.time() - t0
                for item, image in zip(batch, processed):
                    item["image"] = image
                    with lock:
                        state = states[item["doc_id"]]
                        state["doc"]["images"][item["page_idx"]] = image
                        doc_name = state["doc"]["name"]
                    item["image_path"] = save_preprocessed_page(
                        image,
                        out_dir / "preprocessed",
                        doc_name,
                        item["page_idx"],
                    )
                    put_checked(layout_q, item)
                    if pre_pbar is not None:
                        pre_pbar.set_description_str(f"Preprocessing {doc_name}")
                        pre_pbar.update(1)
                batch = []
                if saw_sentinel:
                    break
        except Exception as exc:
            record_worker_error(exc)
        finally:
            if not stop_event.is_set():
                put_sentinels(layout_q, layout_workers)

    def layout_worker():
        try:
            while True:
                page = get_checked(layout_q)
                if page is sentinel:
                    break
                t0 = time.time()
                with lock:
                    mark_parse_started()
                img = page["image"]
                if args.end2end:
                    if args.retry_repeat:
                        raw = batch_inference_with_repeat_retry(
                            model,
                            [img],
                            [ALL_PROMPT["END2END"]],
                            max_tokens=None,
                            max_retries=args.retry_repeat_max_retries,
                        )[0]
                    else:
                        raw = model.batch_inference(
                            [img],
                            [ALL_PROMPT["END2END"]],
                            max_tokens=None,
                        )[0]
                    page_recs, page_layout = parse_end2end_output(raw, img.size)
                    for rec in page_recs:
                        rec["page_num"] = page["page_idx"] + 1
                    with lock:
                        stats["time_parse_requests"] += time.time() - t0
                        state = states[page["doc_id"]]
                        state["layouts"][page["page_idx"]] = page_layout
                        state["pending_pages"] -= 1
                        for rec in page_recs:
                            add_page_result(state, page["page_idx"], rec)
                        maybe_complete(state)
                    continue

                items = get_layout(model, [img])[0]
                with lock:
                    stats["time_parse_requests"] += time.time() - t0
                    state = states[page["doc_id"]]
                    state["layouts"][page["page_idx"]] = items

                w, h = img.size
                created_rec = 0
                no_infer_records = []
                rec_tasks = []
                for block_idx, item in enumerate(items):
                    x1, y1, x2, y2 = item["bbox"]
                    x1 = max(0, min(x1, w - 1 if w > 0 else 0))
                    y1 = max(0, min(y1, h - 1 if h > 0 else 0))
                    x2 = max(x1 + 1, min(int(round(x2)), w))
                    y2 = max(y1 + 1, min(int(round(y2)), h))
                    label = item["label"]
                    task = {
                        "image": img.crop((x1, y1, x2, y2)),
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "question": ALL_PROMPT.get(label, ""),
                        "need_infer": label in ALL_PROMPT,
                        "page_idx": page["page_idx"],
                        "page_num": page["page_idx"] + 1,
                        "block_idx": block_idx,
                        "doc_id": page["doc_id"],
                    }
                    if task["need_infer"]:
                        created_rec += 1
                        rec_tasks.append(task)
                    else:
                        no_infer_records.append(task)

                with lock:
                    state = states[page["doc_id"]]
                    state["pending_pages"] -= 1
                    state["pending_recs"] += created_rec
                    page_to_pdf = [0] * state["doc"]["pdf_pages"]
                    for task in no_infer_records:
                        content = _format_block_content(
                            task,
                            "",
                            page_to_pdf,
                            [state["doc"]["name"]],
                            state["picture_counts"],
                            args.use_base64,
                            image_dir,
                        )
                        add_page_result(state, task["page_idx"], {
                            "bbox": task["bbox"],
                            "label": task["label"],
                            "content": content,
                            "page_num": task["page_num"],
                            "_block_idx": task["block_idx"],
                        })
                    maybe_complete(state)
                for task in rec_tasks:
                    put_checked(rec_q, task)
        except Exception as exc:
            record_worker_error(exc)

    def recognition_worker():
        try:
            while True:
                task = get_checked(rec_q)
                if task is sentinel:
                    break
                t0 = time.time()
                with lock:
                    mark_parse_started()
                raw = _recognize_one_block(
                    model,
                    task,
                    args.retry_repeat,
                    args.retry_repeat_max_retries,
                )
                elapsed = time.time() - t0
                with lock:
                    stats["time_parse_requests"] += elapsed
                    state = states[task["doc_id"]]
                    page_to_pdf = [0] * state["doc"]["pdf_pages"]
                    content = _format_block_content(
                        task,
                        raw,
                        page_to_pdf,
                        [state["doc"]["name"]],
                        state["picture_counts"],
                        args.use_base64,
                        image_dir,
                    )
                    add_page_result(state, task["page_idx"], {
                        "bbox": task["bbox"],
                        "label": task["label"],
                        "content": content,
                        "page_num": task["page_num"],
                        "_block_idx": task["block_idx"],
                    })
                    state["pending_recs"] -= 1
                    maybe_complete(state)
        except Exception as exc:
            record_worker_error(exc)

    def writer_worker():
        try:
            while True:
                doc_id = get_checked(done_q)
                if doc_id is sentinel:
                    break
                with lock:
                    state = states[doc_id]
                    doc_results = []
                    for recs in state["page_results"]:
                        recs = sorted(recs, key=lambda x: x.pop("_block_idx", 0))
                        doc_results.extend(recs)
                    record = build_result_record(state["doc"], doc_results)

                name = state["doc"]["name"]
                if pbar is not None:
                    pbar.set_description_str(f"Parsing {name}")
                if args.draw_layout and layout_dir is not None:
                    draw_layout_pdf(
                        state["doc"]["images"],
                        state["layouts"],
                        str(layout_dir / f"{name}_layout.pdf"),
                    )
                (json_dir / f"{name}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                result2md(
                    [name],
                    [doc_results],
                    save_dir=str(md_dir),
                    keep_header_footer=args.keep_header_footer,
                )
                with lock:
                    completed_records.append((state["doc_idx"], record))
                    states.pop(doc_id, None)
                if pbar is not None:
                    pbar.update(1)
        except Exception as exc:
            record_worker_error(exc)

    t_start = time.time()
    writer = threading.Thread(target=writer_worker, name="mocr2-writer", daemon=True)
    writer.start()
    pre_thread = None
    if not args.skip_preprocess:
        pre_thread = threading.Thread(target=preprocess_worker, name="mocr2-preprocess", daemon=True)
        pre_thread.start()

    layout_threads = [
        threading.Thread(target=layout_worker, name=f"mocr2-layout-{i}", daemon=True)
        for i in range(layout_workers)
    ]
    rec_threads = [
        threading.Thread(target=recognition_worker, name=f"mocr2-rec-{i}", daemon=True)
        for i in range(rec_workers)
    ]
    for th in layout_threads + rec_threads:
        th.start()

    def join_checked(th):
        while th.is_alive():
            th.join(timeout=0.2)
            raise_if_worker_failed()

    def join_best_effort(th, timeout=2.0):
        deadline = time.time() + timeout
        while th.is_alive() and time.time() < deadline:
            th.join(timeout=0.2)

    pipeline_error = None
    try:
        doc_idx = 0
        for doc in _iter_input_documents(args.input_path):
            if args.skip_processed and (md_dir / f"{doc['name']}.md").exists():
                stats["skipped_docs"] += 1
                continue
            doc_id = doc_idx
            doc_idx += 1
            page_count = doc["pdf_pages"]
            if pbar is not None:
                pbar.set_description_str(f"Parsing {doc['name']}")
            elif verbose:
                print(f"Streaming document {doc_idx}: {doc['name']} ({page_count} pages)")
            with lock:
                states[doc_id] = {
                    "doc_id": doc_id,
                    "doc_idx": doc_id,
                    "doc": doc,
                    "layouts": [[] for _ in range(page_count)],
                    "page_results": [[] for _ in range(page_count)],
                    "picture_counts": [0],
                    "pending_pages": page_count,
                    "pending_recs": 0,
                    "done": False,
                }
                stats["submitted_docs"] += 1
                stats["submitted_pages"] += page_count
            for page_idx, image in enumerate(doc["images"]):
                page = {
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "image": image,
                }
                cached_preprocessed = None
                if not args.skip_preprocess and args.skip_processed:
                    cached_path = get_preprocessed_page_path(out_dir / "preprocessed", doc["name"], page_idx)
                    if cached_path.exists():
                        cached_preprocessed = load_image(str(cached_path))

                if cached_preprocessed is not None:
                    page["image"] = cached_preprocessed
                    with lock:
                        states[doc_id]["doc"]["images"][page_idx] = cached_preprocessed
                    put_checked(layout_q, page)
                    if pre_pbar is not None:
                        pre_pbar.set_description_str(f"Preprocessing {doc['name']}")
                        pre_pbar.update(1)
                elif args.skip_preprocess:
                    put_checked(layout_q, page)
                    if pre_pbar is not None:
                        pre_pbar.set_description_str(f"Preprocessing {doc['name']}")
                        pre_pbar.update(1)
                else:
                    put_checked(preprocess_q, page)

        if args.skip_preprocess:
            put_sentinels(layout_q, len(layout_threads))
        else:
            put_checked(preprocess_q, sentinel)

        if pre_thread is not None:
            join_checked(pre_thread)
        for th in layout_threads:
            join_checked(th)
        put_sentinels(rec_q, len(rec_threads))
        for th in rec_threads:
            join_checked(th)
    except Exception as exc:
        pipeline_error = exc
        stop_event.set()
        for q in (preprocess_q, layout_q, rec_q, done_q):
            for _ in range(max(1, layout_workers + rec_workers + 2)):
                try:
                    q.put_nowait(sentinel)
                except queue.Full:
                    break
    finally:
        if pipeline_error is None and error_q.empty():
            if pre_thread is not None:
                join_checked(pre_thread)
            for th in layout_threads:
                join_checked(th)
            for th in rec_threads:
                join_checked(th)
        else:
            if pre_thread is not None:
                join_best_effort(pre_thread)
            for th in layout_threads:
                join_best_effort(th)
            for th in rec_threads:
                join_best_effort(th)
        with lock:
            if stats["parse_started_at"] is not None:
                stats["parse_finished_at"] = time.time()
        try:
            done_q.put_nowait(sentinel)
        except queue.Full:
            pass
        if pipeline_error is None and error_q.empty():
            join_checked(writer)
        else:
            join_best_effort(writer)
        if pbar is not None:
            pbar.close()
        if pre_pbar is not None:
            pre_pbar.close()

    if pipeline_error is not None:
        raise pipeline_error

    if not error_q.empty():
        raise error_q.get()

    all_results = [
        record for _, record in sorted(completed_records, key=lambda x: x[0])
    ]
    (out_dir / "all_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    time_used = time.time() - t_start
    time_parse = 0.0
    if stats["parse_started_at"] is not None and stats["parse_finished_at"] is not None:
        time_parse = stats["parse_finished_at"] - stats["parse_started_at"]
    if verbose:
        print(
            f"Preprocessing time: {stats['time_pre']:.2f} s, "
            f"parsing time: {time_parse:.2f} s"
        )
    if verbose and stats["skipped_docs"]:
        print(f"--skip-processed: skipped {stats['skipped_docs']} already processed documents.")
    avg = time_used / max(1, stats["submitted_docs"])
    if verbose:
        print(
            f"Total time used: {time_used:.2f} s / {stats['submitted_docs']} docs, "
            f"{stats['submitted_pages']} pages, avg {avg:.2f} s/doc."
        )
        print(f"Processing completed. Results saved to {out_dir}")

    preprocessed_dir = out_dir / "preprocessed"
    if preprocessed_dir.exists():
        shutil.rmtree(preprocessed_dir, ignore_errors=True)


@dataclass(frozen=True)
class BackendConfig:
    model_path: str
    server_url: str = ""
    served_model_name: str = "MonkeyOCRv2"
    tp: int = 1
    max_pixels: int = 1003520
    request_timeout: int = 300
    http_max_retries: int = 5
    http_retry_backoff: float = 1.0
    server_max_inflight: int = 1024
    preprocess_batch_size: int = 32
    skip_preprocess: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    output_path: str
    backend: BackendConfig
    page_max_inflight: int = 64
    draw_layout: bool = False
    end2end: bool = False
    skip_processed: bool = False
    retry_repeat: bool = False
    retry_repeat_max_retries: int = 3
    keep_header_footer: bool = False
    use_base64: bool = False
    show_progress_bar: bool = False
    verbose: bool = True


@dataclass(frozen=True)
class OutputDirs:
    out_dir: Path
    json_dir: Path
    md_dir: Path
    image_dir: Path
    preprocessed_dir: Path
    layout_dir: Path | None


class BackendManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}

    def _close_cached_unlocked(self):
        for preprocessor, model in self._cache.values():
            close_preprocessor = getattr(preprocessor, "close", None)
            if callable(close_preprocessor):
                try:
                    close_preprocessor()
                except Exception:
                    pass
            close_model = getattr(model, "close", None)
            if callable(close_model):
                try:
                    close_model()
                except Exception:
                    pass
        self._cache.clear()

    def close(self):
        with self._lock:
            self._close_cached_unlocked()

    def get(self, config: BackendConfig):
        key = (
            "server" if config.server_url else "async",
            config.server_url,
            config.served_model_name,
            str(Path(config.model_path).expanduser().resolve()),
            int(config.tp),
            int(config.server_max_inflight),
            int(config.preprocess_batch_size),
            bool(config.skip_preprocess),
        )
        with self._lock:
            if key not in self._cache:
                self._close_cached_unlocked()
                configure_runtime(config)
                if config.server_url:
                    model = MonkeyOCRv2_ServerParsing(
                        config.server_url,
                        model_name=config.served_model_name,
                        timeout=config.request_timeout,
                        http_max_retries=config.http_max_retries,
                        http_retry_backoff=config.http_retry_backoff,
                    )
                    print(f"Using vLLM server backend: {config.server_url} model={config.served_model_name}")
                else:
                    warnings.warn(
                        "--server-url was not provided; using local vLLM AsyncLLMEngine as the "
                        f"fallback inference backend with model: {config.model_path}",
                        RuntimeWarning,
                    )
                    model = MonkeyOCRv2_AsyncParsing(
                        config.model_path,
                        tp=config.tp,
                        max_inflight=config.server_max_inflight,
                    )
                preprocessor = None
                if not config.skip_preprocess:
                    preprocessor = Preprocessor(config.model_path, batch_size=config.preprocess_batch_size)
                self._cache[key] = (preprocessor, model)
            return self._cache[key]


DEFAULT_BACKEND_MANAGER = BackendManager()
TASK_PROMPTS = {
    "text": ALL_PROMPT["Text"],
    "formula": ALL_PROMPT["Formula"],
    "table": ALL_PROMPT["Table"],
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}


def configure_runtime(config: BackendConfig):
    os.environ["MOCR2_MAX_PIXELS"] = str(config.max_pixels)
    os.environ["MOCR2_SERVER_MAX_INFLIGHT"] = str(config.server_max_inflight)


def prepare_output_dirs(
    output_path: str | Path,
    *,
    skip_preprocess: bool,
    draw_layout: bool = False,
    use_base64: bool = False,
) -> OutputDirs:
    out_dir = Path(output_path).expanduser().resolve()
    json_dir = out_dir / "jsons"
    md_dir = out_dir / "markdowns"
    image_dir = out_dir / "images"
    preprocessed_dir = out_dir / "preprocessed"
    layout_dir = out_dir / "layouts" if draw_layout else None

    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    if not use_base64:
        image_dir.mkdir(parents=True, exist_ok=True)
    if not skip_preprocess:
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
    if layout_dir is not None:
        layout_dir.mkdir(parents=True, exist_ok=True)

    return OutputDirs(out_dir, json_dir, md_dir, image_dir, preprocessed_dir, layout_dir)


def build_pipeline_args(config: PipelineConfig):
    backend = config.backend
    return SimpleNamespace(
        input_path=str(config.input_path),
        model_path=backend.model_path,
        tp=backend.tp,
        max_pixels=backend.max_pixels,
        server_url=backend.server_url,
        served_model_name=backend.served_model_name,
        request_timeout=backend.request_timeout,
        http_max_retries=backend.http_max_retries,
        http_retry_backoff=backend.http_retry_backoff,
        server_max_inflight=backend.server_max_inflight,
        page_max_inflight=config.page_max_inflight,
        preprocess_batch_size=backend.preprocess_batch_size,
        draw_layout=config.draw_layout,
        end2end=config.end2end,
        skip_processed=config.skip_processed,
        skip_preprocess=backend.skip_preprocess,
        retry_repeat=config.retry_repeat,
        retry_repeat_max_retries=config.retry_repeat_max_retries,
        keep_header_footer=config.keep_header_footer,
        use_base64=config.use_base64,
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    backend_manager: BackendManager = DEFAULT_BACKEND_MANAGER,
    parse_semaphore=None,
):
    configure_runtime(config.backend)
    dirs = prepare_output_dirs(
        config.output_path,
        skip_preprocess=config.backend.skip_preprocess,
        draw_layout=config.draw_layout,
        use_base64=config.use_base64,
    )
    preprocessor, model = backend_manager.get(config.backend)
    args = build_pipeline_args(config)

    started = time.time()
    if parse_semaphore is None:
        run_streaming_pipeline(
            args,
            preprocessor,
            model,
            dirs.out_dir,
            dirs.json_dir,
            dirs.md_dir,
            dirs.image_dir,
            show_progress_bar=config.show_progress_bar,
            verbose=config.verbose,
        )
    else:
        with parse_semaphore:
            run_streaming_pipeline(
                args,
                preprocessor,
                model,
                dirs.out_dir,
                dirs.json_dir,
                dirs.md_dir,
                dirs.image_dir,
                show_progress_bar=config.show_progress_bar,
                verbose=config.verbose,
            )

    return {
        "out_dir": dirs.out_dir,
        "json_dir": dirs.json_dir,
        "md_dir": dirs.md_dir,
        "image_dir": dirs.image_dir,
        "elapsed": time.time() - started,
        "all_results_path": dirs.out_dir / "all_results.json",
    }


def load_all_results(out_dir: str | Path):
    path = Path(out_dir) / "all_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_markdowns(md_dir: str | Path):
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(Path(md_dir).glob("*.md"))
    ]


def zip_dir(src_dir: str | Path, zip_path: str | Path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in src_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zipf.write(path, path.relative_to(src_dir))


def _list_single_task_inputs(input_path: str | Path):
    p = Path(input_path)
    files = [p] if p.is_file() else sorted([x for x in p.iterdir() if x.is_file()])
    return [x for x in files if x.suffix.lower() in INPUT_EXTS]


def _load_task_images(input_file: str | Path):
    input_file = Path(input_file)
    suffix = input_file.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_images(str(input_file))
    if suffix in IMAGE_EXTS:
        return [load_image(str(input_file))]
    raise ValueError(f"Unsupported file type for single task recognition: {input_file}")


def _format_single_task_markdown(task: str, doc_name: str, outputs: list[str]):
    if not outputs:
        return ""
    if len(outputs) == 1:
        content = (outputs[0] or "").strip()
    else:
        parts = []
        for idx, raw in enumerate(outputs, 1):
            parts.append(f"## Page {idx}\n\n{(raw or '').strip()}")
        content = "\n\n".join(parts).strip()
    if task == "formula" and content and len(outputs) == 1 and not content.lstrip().startswith("$$"):
        content = "$$\n" + content + "\n$$"
    return content + ("\n" if content else "")


def run_single_task_recognition(
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    backend_config: BackendConfig,
    *,
    backend_manager: BackendManager = DEFAULT_BACKEND_MANAGER,
    parse_semaphore=None,
):
    task = task.lower()
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unsupported task: {task}. Choose from: {', '.join(TASK_PROMPTS)}")

    configure_runtime(backend_config)
    out_dir = Path(output_path).expanduser().resolve()
    md_dir = out_dir / "markdowns"
    json_dir = out_dir / "jsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    _, model = backend_manager.get(backend_config)
    files = _list_single_task_inputs(input_path)
    if not files:
        raise ValueError(f"No supported input files found: {input_path}")

    def infer_one_file(file_path: Path):
        images = _load_task_images(file_path)
        outputs = model.batch_inference(
            images,
            [TASK_PROMPTS[task]] * len(images),
            min_pixels=1003520,
            max_tokens=4096 if task == "table" else None,
        )
        md_text = _format_single_task_markdown(task, file_path.stem, outputs)
        md_path = md_dir / f"{file_path.stem}_{task}_result.md"
        json_path = json_dir / f"{file_path.stem}_{task}_result.json"
        md_path.write_text(md_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "image_name": file_path.name,
                    "image_path": str(file_path),
                    "task": task,
                    "outputs": outputs,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return {
            "input_path": str(file_path),
            "task": task,
            "outputs": outputs,
            "markdown_path": str(md_path),
            "json_path": str(json_path),
        }

    started = time.time()
    results = []
    if parse_semaphore is None:
        for file_path in files:
            results.append(infer_one_file(file_path))
    else:
        with parse_semaphore:
            for file_path in files:
                results.append(infer_one_file(file_path))

    all_results_path = out_dir / f"single_task_{task}_results.json"
    all_results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return {
        "out_dir": out_dir,
        "json_dir": json_dir,
        "md_dir": md_dir,
        "elapsed": time.time() - started,
        "results": results,
        "all_results_path": all_results_path,
    }

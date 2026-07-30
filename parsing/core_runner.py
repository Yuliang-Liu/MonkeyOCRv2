from modeling import modeling_monkeyocrv2_vllm  # noqa: F401 - register vLLM model

import ast
import hashlib
import os
import json
import re
import time
import torch
import base64
import requests
import warnings
import zipfile
import traceback
import threading
import queue
import asyncio
import uuid
import shutil
from collections import OrderedDict, deque
from requests import exceptions as requests_exceptions
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from io import BytesIO
from html import escape
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}
MAX_FILENAME_BYTES = 255
# PDFium is not thread-safe. Every in-process PDFium call must share this lock.
PDFIUM_LOCK = threading.RLock()


def make_artifact_filename(stem: str, suffix: str, max_bytes: int = MAX_FILENAME_BYTES) -> str:
    """Build a deterministic filesystem-safe name without changing names that already fit."""
    stem = str(stem)
    suffix = str(suffix)
    candidate = stem + suffix
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate

    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:10]
    trailer = f"_{digest}{suffix}"
    budget = max_bytes - len(trailer.encode("utf-8"))
    if budget <= 0:
        raise ValueError(f"Artifact suffix is too long: {suffix!r}")
    shortened = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{shortened or 'artifact'}{trailer}"


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
    image_name = make_artifact_filename(doc_name, f"_sub{sub_idx}.jpg")
    image.convert("RGB").save(image_dir / image_name, format="JPEG", quality=95)
    return f"../images/{image_name}"


def save_preprocessed_page(image: Image.Image, preprocessed_dir: Path, doc_name: str, page_idx: int) -> str:
    path = get_preprocessed_page_path(preprocessed_dir, doc_name, page_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", compress_level=1)
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
        default_workers = min(self.max_inflight, 256)
        self.worker_limit = max(
            1,
            min(int(os.getenv("MOCR2_HTTP_WORKERS", str(default_workers))), self.max_inflight),
        )
        self._inflight = threading.BoundedSemaphore(self.max_inflight)
        self._thread_local = threading.local()
        self._sessions = set()
        self._sessions_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_limit,
            thread_name_prefix="mocr2-http",
        )
        self._closed = False

    def _session(self):
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
            with self._sessions_lock:
                self._sessions.add(session)
        return session

    def _reset_session(self):
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
            with self._sessions_lock:
                self._sessions.discard(session)
        self._thread_local.session = requests.Session()
        with self._sessions_lock:
            self._sessions.add(self._thread_local.session)
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
        if len(images) != len(questions):
            raise ValueError("images and questions must contain the same number of items.")
        if self._closed:
            raise RuntimeError("vLLM server backend has already been closed.")
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
        concurrency = max(1, min(int(concurrency or len(prepared)), self.worker_limit))
        outputs = [None] * len(prepared)
        pending = {}
        next_idx = 0
        try:
            while next_idx < len(prepared) or pending:
                while next_idx < len(prepared) and len(pending) < concurrency:
                    future = self._executor.submit(
                        self._chat_completion,
                        prepared[next_idx],
                        questions[next_idx],
                        max_tokens,
                        temperature,
                        top_p,
                    )
                    pending[future] = next_idx
                    next_idx += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    outputs[pending.pop(future)] = future.result()
        except Exception:
            for future in pending:
                future.cancel()
            raise
        return outputs

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._sessions_lock:
            sessions = list(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()


class MonkeyOCRv2_AsyncParsing:
    def __init__(self, model_path: str, tp: int = 1, max_inflight: int = 1024):
        if AsyncLLMEngine is None or AsyncEngineArgs is None:
            raise ImportError("AsyncLLMEngine is unavailable in this vLLM installation.")
        self.model_name = os.path.basename(model_path)
        self.max_inflight = max(1, int(max_inflight))
        self.gen_config = SamplingParams(max_tokens=10000, temperature=0)
        self._engine_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "trust_remote_code": True,
            "max_model_len": 16384,
            "gpu_memory_utilization": self._auto_gpu_mem_ratio(0.5),
        }
        self.engine = None
        self._async_inflight = None
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
        self._async_inflight = asyncio.Semaphore(self.max_inflight)

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

    async def _generate_many(
        self,
        images,
        questions,
        min_pixels,
        max_tokens,
        temperature,
        top_p,
        concurrency,
    ):
        if self._async_inflight is None:
            raise RuntimeError("Async vLLM engine is not initialized.")
        batch_limit = asyncio.Semaphore(concurrency)

        async def generate_one(index):
            async with batch_limit, self._async_inflight:
                return await self._generate_one(
                    images[index],
                    questions[index],
                    min_pixels,
                    max_tokens,
                    temperature,
                    top_p,
                )

        results = await asyncio.gather(
            *(generate_one(i) for i in range(len(images))),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

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
        if len(images) != len(questions):
            raise ValueError("images and questions must contain the same number of items.")
        concurrency = max(1, min(int(concurrency or len(images)), self.max_inflight))
        return self._run_coro(self._generate_many(
            images,
            questions,
            min_pixels,
            max_tokens,
            temperature,
            top_p,
            concurrency,
        ))

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
    

def _extract_balanced_blocks(text: str, left: str, right: str):
    blocks = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == left:
            if depth == 0:
                start = index
            depth += 1
        elif char == right and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:index + 1])
                start = -1
    return blocks


def _deduplicate_strings(values):
    return list(dict.fromkeys(values))


def _extract_tolerant_list_blocks(text: str):
    blocks = _extract_balanced_blocks(text, "[", "]")
    first = text.find("[")
    if first != -1:
        tail = text[first:].strip()
        missing = tail.count("[") - tail.count("]")
        if tail and missing > 0:
            blocks.append(tail + ("]" * missing))
    return _deduplicate_strings(blocks)


def _extract_tolerant_dict_blocks(text: str):
    blocks = _extract_balanced_blocks(text, "{", "}")
    for index, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        end = None
        for cursor in range(index, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
        blocks.append(
            text[index:end]
            if end is not None
            else text[index:] + ("}" * max(depth, 1))
        )
    return _deduplicate_strings(blocks)


def _parse_tolerant_items(text: str, normalize_item):
    text = (text or "").strip()
    if not text:
        return []

    def normalize_list(value):
        if not isinstance(value, list):
            return []
        return [item for raw in value if (item := normalize_item(raw)) is not None]

    try:
        complete = normalize_list(ast.literal_eval(text))
        if complete:
            return complete
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        pass

    best = []
    for block in _extract_tolerant_list_blocks(text):
        try:
            current = normalize_list(ast.literal_eval(block))
            if len(current) > len(best):
                best = current
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue

    dict_items = []
    for block in _extract_tolerant_dict_blocks(text):
        try:
            item = normalize_item(ast.literal_eval(block))
            if item is not None:
                dict_items.append(item)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue
    return dict_items if len(dict_items) > len(best) else best


def _map_bbox_to_image(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
    y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), width - 1 if width > 0 else 0))
    y1 = max(0, min(int(round(y1)), height - 1 if height > 0 else 0))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def _normalize_model_item(item, include_content: bool):
    if not isinstance(item, dict) or "bbox" not in item or "label" not in item:
        return None
    bbox = item["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    normalized = {
        "bbox": bbox,
        "label": item["label"] if isinstance(item["label"], str) else str(item["label"]),
    }
    if include_content:
        content = item.get("content", "")
        normalized["content"] = content if isinstance(content, str) else str(content or "")
    return normalized


def get_layout(model, images: list[Image.Image]):
    outputs = model.batch_inference(
        images,
        [ALL_PROMPT["LAYOUT"]] * len(images),
        min_pixels=1003520,
        max_tokens=4096,
    )
    page_layouts = []
    for image, output in zip(images, outputs):
        width, height = image.size
        items = _parse_tolerant_items(
            output,
            lambda item: _normalize_model_item(item, include_content=False),
        )
        page_layouts.append([{
            "bbox": _map_bbox_to_image(item["bbox"], width, height),
            "label": item["label"],
        } for item in items])
    return page_layouts


def parse_end2end_output(text: str, image_size: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    width, height = image_size
    records = []
    layouts = []
    items = _parse_tolerant_items(
        text,
        lambda item: _normalize_model_item(item, include_content=True),
    )
    for block_idx, item in enumerate(items):
        bbox = _map_bbox_to_image(item["bbox"], width, height)
        label = item["label"]
        records.append({
            "bbox": bbox,
            "label": label,
            "content": (item.get("content") or "").strip(),
            "_block_idx": block_idx,
        })
        layouts.append({"bbox": bbox, "label": label})
    return records, layouts


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
    doc_name: str | None,
    picture_count: list[int] | None,
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
            if picture_count is None:
                raise ValueError("picture_count is required for Picture blocks")
            sub_idx = picture_count[0]
            picture_count[0] += 1
            image_ref = save_picture_block(task["image"], image_dir, doc_name, sub_idx)
        content = f"![image]({image_ref})"
    elif label == "Title":
        content = "# " + content.replace("\n", "\n# ")
    elif label == "Section-header":
        content = "## " + content.replace("\n", "\n## ")
    elif not task["need_infer"]:
        content = ""
    return content


def _build_page_tasks(page_idx, image, layouts, doc_id=None):
    width, height = image.size
    tasks = []
    for block_idx, item in enumerate(layouts):
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0, min(int(round(x1)), max(0, width - 1)))
        y1 = max(0, min(int(round(y1)), max(0, height - 1)))
        x2 = max(x1 + 1, min(int(round(x2)), width))
        y2 = max(y1 + 1, min(int(round(y2)), height))
        label = item["label"]
        task = {
            "image": image.crop((x1, y1, x2, y2)),
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "question": ALL_PROMPT.get(label, ""),
            "need_infer": label in ALL_PROMPT,
            "page_idx": page_idx,
            "page_num": page_idx + 1,
            "block_idx": block_idx,
        }
        if doc_id is not None:
            task["doc_id"] = doc_id
        tasks.append(task)
    return tasks


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
            (out_dir / make_artifact_filename(names[i], ".md")).write_text(md, encoding="utf-8")

    return md_list


def _render_pdf_page(pdf, page_idx: int) -> Image.Image:
    with PDFIUM_LOCK:
        page = pdf[page_idx]
        try:
            bitmap = page.render(scale=200 / 72)
            try:
                return bitmap.to_pil().convert("RGB")
            finally:
                close_bitmap = getattr(bitmap, "close", None)
                if callable(close_bitmap):
                    close_bitmap()
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()


def load_pdf_images(pdf_path: str):
    try:
        import pypdfium2 as pdfium
    except Exception as e:
        raise ImportError("Reading PDF files requires pypdfium2") from e

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            return [_render_pdf_page(pdf, page_idx) for page_idx in range(len(pdf))]
        finally:
            close_pdf = getattr(pdf, "close", None)
            if callable(close_pdf):
                close_pdf()


class _PdfRenderer:
    def __init__(self, max_open_documents: int = 16):
        self.max_open_documents = max(1, int(max_open_documents))
        self._cache = OrderedDict()

    @staticmethod
    def _close_handle(pdf):
        close_pdf = getattr(pdf, "close", None)
        if callable(close_pdf):
            close_pdf()

    def render(self, pdf_path: str | Path, page_idx: int) -> Image.Image:
        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise ImportError("Reading PDF files requires pypdfium2") from exc

        with PDFIUM_LOCK:
            key = str(Path(pdf_path).resolve())
            pdf = self._cache.pop(key, None)
            if pdf is None:
                pdf = pdfium.PdfDocument(key)
            self._cache[key] = pdf
            while len(self._cache) > self.max_open_documents:
                _, stale_pdf = self._cache.popitem(last=False)
                self._close_handle(stale_pdf)
            return _render_pdf_page(pdf, page_idx)

    def close(self):
        with PDFIUM_LOCK:
            handles = list(self._cache.values())
            self._cache.clear()
            for pdf in handles:
                try:
                    self._close_handle(pdf)
                except Exception:
                    pass


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
            response = requests.get(image_source, headers=headers, timeout=FETCH_TIMEOUT)
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
    total = 0
    for f in _list_input_files(input_path):
        if f.suffix.lower() not in INPUT_EXTS:
            continue
        if skip_processed and (md_dir / make_artifact_filename(f.stem, ".md")).exists():
            continue
        total += 1
    return total


def _count_pending_pages(input_path: str, md_dir: Path, skip_processed: bool) -> int:
    total = 0
    for f in _list_input_files(input_path):
        ext = f.suffix.lower()
        if ext not in INPUT_EXTS:
            continue
        if skip_processed and (md_dir / make_artifact_filename(f.stem, ".md")).exists():
            continue
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError:
                total += 1
                continue
            with PDFIUM_LOCK:
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


def _iter_input_page_events(
    input_path: str,
    md_dir: Path,
    skip_processed: bool,
    acquire_page_slot,
    release_page_slot,
    preprocessed_dir: Path | None = None,
):
    doc_id = 0
    for input_file in _list_input_files(input_path):
        ext = input_file.suffix.lower()
        if ext not in INPUT_EXTS:
            continue
        if skip_processed and (
            md_dir / make_artifact_filename(input_file.stem, ".md")
        ).exists():
            yield ("skipped",)
            continue

        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except Exception as exc:
                raise ImportError("Reading PDF files requires pypdfium2") from exc
            with PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(str(input_file))
            try:
                with PDFIUM_LOCK:
                    page_count = len(pdf)
                yield ("doc", doc_id, {
                    "name": input_file.stem,
                    "image_name": input_file.name,
                    "image_path": str(input_file),
                    "pdf_pages": page_count,
                })
                for page_idx in range(page_count):
                    cached_path = None
                    if preprocessed_dir is not None:
                        cached_path = get_preprocessed_page_path(
                            preprocessed_dir, input_file.stem, page_idx
                        )
                    if cached_path is not None and skip_processed and cached_path.exists():
                        yield ("page", doc_id, page_idx, None, str(cached_path), False)
                        continue
                    if not acquire_page_slot():
                        return
                    try:
                        image = _render_pdf_page(pdf, page_idx)
                    except Exception:
                        release_page_slot()
                        raise
                    yield (
                        "page", doc_id, page_idx, image,
                        str(cached_path) if cached_path is not None else None,
                        True,
                    )
            finally:
                with PDFIUM_LOCK:
                    close_pdf = getattr(pdf, "close", None)
                    if callable(close_pdf):
                        close_pdf()
        else:
            yield ("doc", doc_id, {
                "name": input_file.stem,
                "image_name": input_file.name,
                "image_path": str(input_file),
                "pdf_pages": 1,
            })
            cached_path = None
            if preprocessed_dir is not None:
                cached_path = get_preprocessed_page_path(preprocessed_dir, input_file.stem, 0)
            if cached_path is not None and skip_processed and cached_path.exists():
                yield ("page", doc_id, 0, None, str(cached_path), False)
            else:
                if not acquire_page_slot():
                    return
                try:
                    image = load_image(str(input_file))
                except Exception:
                    release_page_slot()
                    raise
                yield (
                    "page", doc_id, 0, image,
                    str(cached_path) if cached_path is not None else None,
                    True,
                )
        doc_id += 1


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
    layout_q = queue.Queue(maxsize=page_window)
    rec_q = queue.Queue(maxsize=max(page_window * 8, server_window * 2))
    layout_workers = max(1, min(page_window, server_window))
    rec_workers = max(1, min(server_window, max(32, page_window * 4), 256))
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
        "time_pre_stage": 0.0,
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
        if not args.skip_preprocess:
            pre_pbar = tqdm(
                total=total_pages,
                dynamic_ncols=True,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}",
                position=0,
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

    def release_input_page(page):
        if not isinstance(page, dict):
            return
        page.pop("image", None)
        release_slot = page.pop("_release_input_slot", None)
        if release_slot is not None:
            release_slot()

    def layout_worker():
        page = None
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
                    release_input_page(page)
                    continue

                items = get_layout(model, [img])[0]
                with lock:
                    stats["time_parse_requests"] += time.time() - t0
                    state = states[page["doc_id"]]
                    state["layouts"][page["page_idx"]] = items

                created_rec = 0
                no_infer_records = []
                rec_tasks = []
                for task in _build_page_tasks(
                    page["page_idx"], img, items, doc_id=page["doc_id"]
                ):
                    if task["need_infer"]:
                        created_rec += 1
                        rec_tasks.append(task)
                    else:
                        no_infer_records.append(task)

                with lock:
                    state = states[page["doc_id"]]
                    state["pending_pages"] -= 1
                    state["pending_recs"] += created_rec
                    for task in no_infer_records:
                        content = _format_block_content(
                            task,
                            "",
                            state["doc"]["name"],
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
                release_input_page(page)
                for task in rec_tasks:
                    put_checked(rec_q, task)
        except Exception as exc:
            release_input_page(page)
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
                    content = _format_block_content(
                        task,
                        raw,
                        state["doc"]["name"],
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
                        str(layout_dir / make_artifact_filename(name, "_layout.pdf")),
                    )
                (json_dir / make_artifact_filename(name, ".json")).write_text(
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
    writer = None
    reader_thread = None
    preprocess_save_threads = []
    layout_threads = []
    rec_threads = []

    def join_checked(th):
        while th.is_alive():
            th.join(timeout=0.2)
            raise_if_worker_failed()

    def join_best_effort(th, timeout=2.0):
        deadline = time.time() + timeout
        while th.is_alive() and time.time() < deadline:
            th.join(timeout=0.2)

    def start_input_reader(preprocessed_dir=None):
        input_q = queue.Queue(maxsize=page_window)
        raw_page_slots = threading.BoundedSemaphore(page_window)

        def acquire_page_slot():
            while not stop_event.is_set():
                if raw_page_slots.acquire(timeout=0.2):
                    return True
                raise_if_worker_failed()
            return False

        def reader_worker():
            try:
                events = _iter_input_page_events(
                    args.input_path,
                    md_dir,
                    args.skip_processed,
                    acquire_page_slot,
                    raw_page_slots.release,
                    preprocessed_dir,
                )
                for event in events:
                    try:
                        put_checked(input_q, event)
                    except Exception:
                        if event[0] == "page" and event[5]:
                            raw_page_slots.release()
                        raise
            except Exception as exc:
                record_worker_error(exc)
            finally:
                if not stop_event.is_set():
                    put_checked(input_q, sentinel)

        thread = threading.Thread(
            target=reader_worker,
            name="mocr2-input-reader",
            daemon=True,
        )
        thread.start()
        return input_q, raw_page_slots, thread

    pipeline_error = None
    try:
        prepared_docs = None
        if not args.skip_preprocess:
            preprocess_stage_started = time.time()
            prepared_docs = []
            prepared_docs_by_id = {}
            preprocess_batch = []
            preprocess_target = min(max(1, args.preprocess_batch_size), page_window)
            warm_pages = 0
            preprocess_save_q = queue.Queue(maxsize=page_window)
            preprocess_save_workers = max(1, min(4, page_window, os.cpu_count() or 1))
            progress_lock = threading.Lock()

            def update_preprocess_progress(doc_name):
                if pre_pbar is not None:
                    with progress_lock:
                        pre_pbar.set_description_str(f"Preprocessing {doc_name}")
                        pre_pbar.update(1)

            def preprocess_save_worker():
                try:
                    while True:
                        item = get_checked(preprocess_save_q)
                        if item is sentinel:
                            break
                        save_preprocessed_page(
                            item["image"],
                            out_dir / "preprocessed",
                            item["doc_name"],
                            item["page_idx"],
                        )
                        update_preprocess_progress(item["doc_name"])
                except Exception as exc:
                    record_worker_error(exc)

            def flush_preprocess_batch():
                nonlocal preprocess_batch, warm_pages
                if not preprocess_batch:
                    return
                current_batch = preprocess_batch
                preprocess_batch = []
                try:
                    t0 = time.time()
                    output_batch = preprocessor.preprocess_images(
                        [item["image"] for item in current_batch],
                        batch_size=args.preprocess_batch_size,
                    )
                    stats["time_pre"] += time.time() - t0
                    if len(output_batch) != len(current_batch):
                        raise RuntimeError(
                            "Preprocessor returned a different number of images than it received."
                        )
                    for item, image in zip(current_batch, output_batch):
                        if item["slot_acquired"]:
                            raw_page_slots.release()
                            item["slot_acquired"] = False
                        item.pop("image", None)
                        item["doc"]["_image_sizes"][item["page_idx"]] = [
                            int(image.size[0]), int(image.size[1])
                        ]
                        if warm_pages < page_window:
                            item["doc"]["preprocessed_paths"][item["page_idx"]] = image
                            warm_pages += 1
                        put_checked(preprocess_save_q, {
                            "image": image,
                            "doc_name": item["doc"]["name"],
                            "page_idx": item["page_idx"],
                        })
                finally:
                    for item in current_batch:
                        if item["slot_acquired"]:
                            raw_page_slots.release()

            preprocess_save_threads = [
                threading.Thread(
                    target=preprocess_save_worker,
                    name=f"mocr2-preprocess-writer-{i}",
                    daemon=True,
                )
                for i in range(preprocess_save_workers)
            ]
            for thread in preprocess_save_threads:
                thread.start()
            input_q, raw_page_slots, reader_thread = start_input_reader(
                out_dir / "preprocessed"
            )
            while True:
                event = get_checked(input_q)
                if event is sentinel:
                    break
                if event[0] == "skipped":
                    stats["skipped_docs"] += 1
                    continue
                if event[0] == "doc":
                    _, doc_id, doc = event
                    prepared_doc = {
                        "name": doc["name"],
                        "image_name": doc["image_name"],
                        "image_path": doc["image_path"],
                        "images": [None] * doc["pdf_pages"],
                        "preprocessed_paths": [None] * doc["pdf_pages"],
                        "_image_sizes": [None] * doc["pdf_pages"],
                        "pdf_pages": doc["pdf_pages"],
                    }
                    prepared_docs.append(prepared_doc)
                    prepared_docs_by_id[doc_id] = prepared_doc
                    continue

                _, doc_id, page_idx, image, cached_path, slot_acquired = event
                prepared_doc = prepared_docs_by_id[doc_id]
                prepared_doc["preprocessed_paths"][page_idx] = cached_path
                if image is None:
                    cached_image = load_image(cached_path)
                    prepared_doc["_image_sizes"][page_idx] = [
                        int(cached_image.size[0]), int(cached_image.size[1])
                    ]
                    if warm_pages < page_window:
                        prepared_doc["preprocessed_paths"][page_idx] = cached_image
                        warm_pages += 1
                    update_preprocess_progress(prepared_doc["name"])
                    continue

                preprocess_batch.append({
                    "doc": prepared_doc,
                    "page_idx": page_idx,
                    "image": image,
                    "slot_acquired": slot_acquired,
                })
                if len(preprocess_batch) >= preprocess_target:
                    flush_preprocess_batch()

            flush_preprocess_batch()
            join_checked(reader_thread)
            reader_thread = None
            put_sentinels(preprocess_save_q, len(preprocess_save_threads))
            for thread in preprocess_save_threads:
                join_checked(thread)
            preprocess_save_threads = []
            stats["time_pre_stage"] = time.time() - preprocess_stage_started

            for doc in prepared_docs:
                image_sizes = doc.pop("_image_sizes")
                if any(size is None for size in image_sizes):
                    raise RuntimeError(f"Preprocessing did not produce every page for {doc['name']}.")
                doc["image_size"] = image_sizes[0] if len(image_sizes) == 1 else image_sizes

            if pre_pbar is not None:
                pre_pbar.set_description_str("Preprocessing complete")
                pre_pbar.close()
                pre_pbar = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if show_progress_bar and tqdm is not None and total_docs > 0:
            pbar = tqdm(
                total=total_docs,
                dynamic_ncols=True,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}",
                position=0,
                leave=True,
            )

        writer = threading.Thread(target=writer_worker, name="mocr2-writer", daemon=True)
        writer.start()
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

        def register_document(doc_id, doc, doc_idx):
            page_count = doc["pdf_pages"]
            if pbar is not None:
                pbar.set_description_str(f"Parsing {doc['name']}")
            elif verbose:
                print(f"Streaming document {doc_idx + 1}: {doc['name']} ({page_count} pages)")
            with lock:
                states[doc_id] = {
                    "doc_id": doc_id,
                    "doc_idx": doc_idx,
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

        if prepared_docs is not None:
            for doc_idx, doc in enumerate(prepared_docs):
                doc_id = doc_idx
                register_document(doc_id, doc, doc_idx)
                for page_idx, source in enumerate(doc["preprocessed_paths"]):
                    image = load_image(source) if isinstance(source, str) else source
                    if args.draw_layout:
                        with lock:
                            states[doc_id]["doc"]["images"][page_idx] = image
                    put_checked(layout_q, {
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "image": image,
                    })
        else:
            input_q, raw_page_slots, reader_thread = start_input_reader()
            direct_docs = {}
            while True:
                event = get_checked(input_q)
                if event is sentinel:
                    break
                if event[0] == "skipped":
                    stats["skipped_docs"] += 1
                    continue
                if event[0] == "doc":
                    _, doc_id, doc_meta = event
                    page_count = doc_meta["pdf_pages"]
                    doc = {
                        **doc_meta,
                        "images": [None] * page_count,
                        "_image_sizes": [None] * page_count,
                    }
                    direct_docs[doc_id] = doc
                    register_document(doc_id, doc, len(direct_docs) - 1)
                    continue

                _, doc_id, page_idx, image, _, slot_acquired = event
                doc = direct_docs[doc_id]
                doc["_image_sizes"][page_idx] = [int(image.size[0]), int(image.size[1])]
                if args.draw_layout:
                    with lock:
                        states[doc_id]["doc"]["images"][page_idx] = image
                if all(size is not None for size in doc["_image_sizes"]):
                    image_sizes = doc.pop("_image_sizes")
                    doc["image_size"] = image_sizes[0] if len(image_sizes) == 1 else image_sizes
                page = {
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "image": image,
                }
                if slot_acquired:
                    page["_release_input_slot"] = raw_page_slots.release
                try:
                    put_checked(layout_q, page)
                except Exception:
                    release_input_page(page)
                    raise
            join_checked(reader_thread)
            reader_thread = None

        put_sentinels(layout_q, len(layout_threads))
        for th in layout_threads:
            join_checked(th)
        put_sentinels(rec_q, len(rec_threads))
        for th in rec_threads:
            join_checked(th)
    except Exception as exc:
        pipeline_error = exc
        stop_event.set()
        for q in (layout_q, rec_q, done_q):
            for _ in range(max(1, layout_workers + rec_workers + 2)):
                try:
                    q.put_nowait(sentinel)
                except queue.Full:
                    break
    finally:
        if reader_thread is not None:
            join_best_effort(reader_thread)
        for thread in preprocess_save_threads:
            join_best_effort(thread)
        if pipeline_error is None and error_q.empty():
            for th in layout_threads:
                join_checked(th)
            for th in rec_threads:
                join_checked(th)
        else:
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
        if writer is not None and pipeline_error is None and error_q.empty():
            join_checked(writer)
        elif writer is not None:
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
            f"Preprocess time: {stats['time_pre_stage']:.2f} s, "
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
            int(config.max_pixels),
            int(config.request_timeout),
            int(config.http_max_retries),
            float(config.http_retry_backoff),
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


class _BatchCompletion:
    def __init__(self, size: int):
        self._remaining = size
        self._lock = threading.Lock()
        self._event = threading.Event()
        if size == 0:
            self._event.set()

    def done(self):
        with self._lock:
            self._remaining -= 1
            if self._remaining <= 0:
                self._event.set()

    def wait(self, stop_event: threading.Event):
        while not self._event.wait(0.2):
            if stop_event.is_set():
                raise RuntimeError("Service pipeline stopped while waiting for a parse batch.")


class _ServiceJob:
    def __init__(
        self,
        config: PipelineConfig,
        dirs: OutputDirs,
        single_task: str | None = None,
    ):
        self.config = config
        self.dirs = dirs
        self.single_task = single_task
        self.skip_preprocess = single_task is not None or config.backend.skip_preprocess
        self.future = Future()
        self.lock = threading.Lock()
        self.doc = None
        self.page_results = []
        self.single_outputs = []
        self.pending_pages = 0
        self.picture_counts = [0]
        self.failed = False
        self.started_at = time.time()

    def initialize(self, input_path: Path, page_count: int):
        with self.lock:
            self.doc = {
                "name": input_path.stem,
                "image_name": input_path.name,
                "image_path": input_path.name,
                "image_size": [None] * page_count,
                "pdf_pages": page_count,
            }
            self.page_results = [[] for _ in range(page_count)]
            self.single_outputs = [None] * page_count
            self.pending_pages = page_count

    def fail(self, exc: Exception):
        with self.lock:
            if self.failed or self.future.done():
                return False
            self.failed = True
            self.future.set_exception(exc)
            return True


class ServicePipelinePool:
    """Shared request scheduler used by the demo and API services."""

    def __init__(
        self,
        backend_config: BackendConfig,
        page_max_inflight: int,
        *,
        backend_manager: BackendManager = DEFAULT_BACKEND_MANAGER,
        batch_wait_seconds: float = 1.0,
        debug: bool = False,
    ):
        configure_runtime(backend_config)
        self.backend_config = backend_config
        self.page_window = max(1, int(page_max_inflight))
        self.batch_wait_seconds = max(0.0, float(batch_wait_seconds))
        self.debug = bool(debug)
        self.preprocessor, self.model = backend_manager.get(backend_config)
        self.stop_event = threading.Event()
        self._jobs_lock = threading.Lock()
        self._active_jobs = set()
        self._accepting_jobs = True
        self._closed = False
        self.request_q = queue.Queue()
        self.preprocess_q = queue.Queue(maxsize=self.page_window)
        self.parse_q = queue.Queue(maxsize=self.page_window)
        self.preprocess_slots = threading.BoundedSemaphore(self.page_window)
        self.parse_slots = threading.BoundedSemaphore(self.page_window)
        self.pdf_renderer = _PdfRenderer()
        self.page_executor = ThreadPoolExecutor(max_workers=self.page_window)
        self.output_executor = ThreadPoolExecutor(max_workers=max(1, min(4, self.page_window)))
        self._sentinel = object()
        self._threads = [
            threading.Thread(target=self._request_worker, name="mocr2-service-reader", daemon=True),
            threading.Thread(target=self._preprocess_worker, name="mocr2-service-preprocess", daemon=True),
            threading.Thread(target=self._parse_worker, name="mocr2-service-parse", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _report_error(self, stage: str, exc: Exception):
        if not self.debug:
            return
        print(
            f"[ServicePipelinePool:{stage}] {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    def _fail_job(self, job: _ServiceJob, stage: str, exc: Exception):
        if job.fail(exc):
            self._report_error(stage, exc)

    def _remove_job(self, job: _ServiceJob):
        with self._jobs_lock:
            self._active_jobs.discard(job)

    def _register_job(self, job: _ServiceJob):
        with self._jobs_lock:
            if not self._accepting_jobs:
                raise RuntimeError("Service pipeline is shutting down.")
            self._active_jobs.add(job)
        job.future.add_done_callback(lambda _future: self._remove_job(job))

    def _put(self, target_q, item):
        while not self.stop_event.is_set():
            try:
                target_q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue
        raise RuntimeError("Service pipeline is shutting down.")

    def _acquire_slot(self, slot):
        while not self.stop_event.is_set():
            if slot.acquire(timeout=0.2):
                return
        raise RuntimeError("Service pipeline is shutting down.")

    def _enqueue_parse(self, page, *, slot_reserved=False):
        if not slot_reserved:
            self._acquire_slot(self.parse_slots)
        try:
            self._put(self.parse_q, page)
        except Exception:
            self.parse_slots.release()
            raise

    def run(self, config: PipelineConfig):
        job = None
        try:
            if config.backend != self.backend_config:
                raise ValueError("ServicePipelinePool backend configuration cannot change per request.")
            if int(config.page_max_inflight) != self.page_window:
                raise ValueError(
                    "PipelineConfig.page_max_inflight must match the service pool page window: "
                    f"{config.page_max_inflight} != {self.page_window}"
                )
            dirs = prepare_output_dirs(
                config.output_path,
                # Service requests pass preprocessed PIL images directly to parsing.
                skip_preprocess=True,
                draw_layout=config.draw_layout,
                use_base64=config.use_base64,
            )
            job = _ServiceJob(config, dirs)
            self._register_job(job)
            self._put(self.request_q, job)
            return job.future.result()
        except Exception as exc:
            if job is not None and not job.future.done():
                job.fail(exc)
            if job is None or not job.failed:
                self._report_error("submit", exc)
            raise
    def run_single_task(self, input_path, output_path, task):
        task = task.lower()
        if task not in TASK_PROMPTS:
            raise ValueError(f"Unsupported task: {task}. Choose from: {', '.join(TASK_PROMPTS)}")
        config = PipelineConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            backend=self.backend_config,
            page_max_inflight=self.page_window,
        )
        dirs = prepare_output_dirs(
            output_path,
            skip_preprocess=True,
            use_base64=True,
        )
        job = _ServiceJob(config, dirs, single_task=task)
        try:
            self._register_job(job)
            self._put(self.request_q, job)
            return job.future.result()
        except Exception as exc:
            if not job.future.done():
                job.fail(exc)
            raise

    def _create_read_state(self, job: _ServiceJob):
        input_path = Path(job.config.input_path)
        if not input_path.is_file() or input_path.suffix.lower() not in INPUT_EXTS:
            raise ValueError(f"Service parsing expects one PDF or image file: {input_path}")
        if input_path.suffix.lower() == ".pdf":
            try:
                import pypdfium2 as pdfium
            except Exception as exc:
                raise ImportError("Reading PDF files requires pypdfium2") from exc
            with PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(str(input_path))
                try:
                    page_count = len(pdf)
                finally:
                    close_pdf = getattr(pdf, "close", None)
                    if callable(close_pdf):
                        close_pdf()
        else:
            page_count = 1
        if page_count == 0:
            raise ValueError(f"PDF contains no pages: {input_path}")
        job.initialize(input_path, page_count)
        return {
            "job": job,
            "input_path": input_path,
            "is_pdf": input_path.suffix.lower() == ".pdf",
            "page_count": page_count,
            "next_page": 0,
        }

    @staticmethod
    def _try_acquire_slot(slot) -> bool:
        return slot.acquire(blocking=False)

    def _submit_read(self, state) -> bool:
        if state["next_page"] >= state["page_count"]:
            return False
        job = state["job"]
        slot = self.parse_slots if job.skip_preprocess else self.preprocess_slots
        if not self._try_acquire_slot(slot):
            return False
        page_idx = state["next_page"]
        state["next_page"] += 1
        try:
            if state["is_pdf"]:
                image = self.pdf_renderer.render(state["input_path"], page_idx)
            else:
                image = load_image(str(state["input_path"]))
        except Exception:
            slot.release()
            raise

        page = {"job": job, "page_idx": page_idx, "image": image}
        if job.skip_preprocess:
            self._enqueue_parse(page, slot_reserved=True)
        else:
            try:
                self._put(self.preprocess_q, page)
            except Exception:
                self.preprocess_slots.release()
                raise
        return True

    def _request_worker(self):
        active = deque()
        while not self.stop_event.is_set():
            made_progress = False
            while True:
                try:
                    job = self.request_q.get_nowait()
                except queue.Empty:
                    break
                if job is self._sentinel:
                    continue
                try:
                    state = self._create_read_state(job)
                    active.append(state)
                    made_progress = True
                except Exception as exc:
                    self._fail_job(job, "request-reader", exc)

            for _ in range(len(active)):
                state = active.popleft()
                job = state["job"]
                try:
                    if job.failed:
                        continue
                    # One page per request per round keeps
                    # large PDFs from monopolizing the shared page window.
                    made_progress = self._submit_read(state) or made_progress
                    if state["next_page"] < state["page_count"]:
                        active.append(state)
                except Exception as exc:
                    self._fail_job(job, "request-reader", exc)

            if not made_progress:
                try:
                    job = self.request_q.get(timeout=0.02)
                except queue.Empty:
                    continue
                if job is not self._sentinel:
                    try:
                        state = self._create_read_state(job)
                        active.append(state)
                    except Exception as exc:
                        self._fail_job(job, "request-reader", exc)

    def _preprocess_worker(self):
        if self.preprocessor is None:
            return
        saw_sentinel = False
        while not self.stop_event.is_set() and not saw_sentinel:
            try:
                first = self.preprocess_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if first is self._sentinel:
                break
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_seconds
            while len(batch) < self.page_window:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.preprocess_q.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is self._sentinel:
                    saw_sentinel = True
                    break
                batch.append(item)

            active = []
            for page in batch:
                if page["job"].failed:
                    self.preprocess_slots.release()
                else:
                    active.append(page)
            if not active:
                continue
            try:
                images = self.preprocessor.preprocess_images(
                    [page["image"] for page in active],
                    batch_size=self.backend_config.preprocess_batch_size,
                )
                if len(images) != len(active):
                    raise RuntimeError("Preprocessor returned an unexpected number of pages.")
                for page in active:
                    self.preprocess_slots.release()
                    page["preprocess_slot_released"] = True
                for page, image in zip(active, images):
                    page["image"] = image

                completion = _BatchCompletion(len(active))
                for page in active:
                    page["batch_completion"] = completion
                    self._enqueue_parse(page)
                completion.wait(self.stop_event)
            except Exception as exc:
                for page in active:
                    if not page.get("preprocess_slot_released"):
                        self.preprocess_slots.release()
                    self._fail_job(page["job"], "preprocess", exc)

    def _parse_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    page = self.parse_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if page is self._sentinel:
                    break
                self.page_executor.submit(self._parse_page, page)
        except Exception as exc:
            self._report_error("parse-dispatcher", exc)
            self.stop_event.set()

    def _parse_page(self, page):
        job = page["job"]
        completion = page.get("batch_completion")
        try:
            if job.failed:
                return
            image = page["image"]
            page_idx = page["page_idx"]
            if job.single_task is not None:
                raw = self.model.batch_inference(
                    [image],
                    [TASK_PROMPTS[job.single_task]],
                    min_pixels=1003520,
                    max_tokens=4096 if job.single_task == "table" else None,
                )[0]
                output = _format_single_task_outputs(job.single_task, [raw])[0]
                self._complete_single_task_page(job, page_idx, output)
            elif job.config.end2end:
                if job.config.retry_repeat:
                    raw = batch_inference_with_repeat_retry(
                        self.model,
                        [image],
                        [ALL_PROMPT["END2END"]],
                        max_tokens=None,
                        max_retries=job.config.retry_repeat_max_retries,
                    )[0]
                else:
                    raw = self.model.batch_inference(
                        [image], [ALL_PROMPT["END2END"]], max_tokens=None
                    )[0]
                records, layouts = parse_end2end_output(raw, image.size)
                for record in records:
                    record["page_num"] = page_idx + 1
            else:
                layouts = get_layout(self.model, [image])[0]
                tasks = _build_page_tasks(page_idx, image, layouts)
                infer_tasks = [task for task in tasks if task["need_infer"]]
                if infer_tasks:
                    infer_images = [task["image"] for task in infer_tasks]
                    questions = [task["question"] for task in infer_tasks]
                    if job.config.retry_repeat:
                        outputs = batch_inference_with_repeat_retry(
                            self.model,
                            infer_images,
                            questions,
                            max_tokens=5000,
                            max_retries=job.config.retry_repeat_max_retries,
                        )
                    else:
                        outputs = self.model.batch_inference(
                            infer_images, questions, max_tokens=5000
                        )
                    raw_by_block = {
                        task["block_idx"]: raw for task, raw in zip(infer_tasks, outputs)
                    }
                else:
                    raw_by_block = {}
                with job.lock:
                    records = []
                    for task in tasks:
                        content = _format_block_content(
                            task,
                            raw_by_block.get(task["block_idx"], ""),
                            job.doc["name"],
                            job.picture_counts,
                            job.config.use_base64,
                            job.dirs.image_dir,
                        )
                        records.append({
                            "bbox": task["bbox"],
                            "label": task["label"],
                            "content": content,
                            "page_num": page_idx + 1,
                        })
            if job.single_task is None:
                self._complete_page(job, page_idx, image.size, records)
        except Exception as exc:
            self._fail_job(job, "parse-page", exc)
        finally:
            if completion is not None:
                completion.done()
            self.parse_slots.release()

    def _complete_page(self, job, page_idx, image_size, records):
        should_finalize = False
        with job.lock:
            if job.failed:
                return
            job.doc["image_size"][page_idx] = [int(image_size[0]), int(image_size[1])]
            job.page_results[page_idx] = records
            job.pending_pages -= 1
            should_finalize = job.pending_pages == 0
        if should_finalize:
            self.output_executor.submit(self._finalize_job, job)

    def _complete_single_task_page(self, job, page_idx, output):
        should_finalize = False
        with job.lock:
            if job.failed:
                return
            job.single_outputs[page_idx] = output
            job.pending_pages -= 1
            should_finalize = job.pending_pages == 0
        if should_finalize:
            self.output_executor.submit(self._finalize_single_task_job, job)

    def _finalize_single_task_job(self, job):
        if job.failed or job.future.done():
            return
        try:
            name = job.doc["name"]
            task = job.single_task
            outputs = list(job.single_outputs)
            md_path = job.dirs.md_dir / make_artifact_filename(name, f"_{task}_result.md")
            json_path = job.dirs.json_dir / make_artifact_filename(name, f"_{task}_result.json")
            md_path.write_text(_format_single_task_markdown(outputs), encoding="utf-8")
            json_path.write_text(
                json.dumps({
                    "image_name": job.doc["image_name"],
                    "image_path": job.doc["image_path"],
                    "task": task,
                    "outputs": outputs,
                }, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            results = [{
                "input_path": job.doc["image_path"],
                "task": task,
                "outputs": outputs,
                "markdown_path": str(md_path),
                "json_path": str(json_path),
            }]
            all_results_path = job.dirs.out_dir / f"single_task_{task}_results.json"
            all_results_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            job.future.set_result({
                "out_dir": job.dirs.out_dir,
                "json_dir": job.dirs.json_dir,
                "md_dir": job.dirs.md_dir,
                "elapsed": time.time() - job.started_at,
                "results": results,
                "all_results_path": all_results_path,
            })
        except Exception as exc:
            self._fail_job(job, "output-writer", exc)

    def _finalize_job(self, job):
        if job.failed or job.future.done():
            return
        try:
            with job.lock:
                results = [record for page in job.page_results for record in page]
                image_sizes = job.doc["image_size"]
                job.doc["image_size"] = image_sizes[0] if len(image_sizes) == 1 else image_sizes
                record = build_result_record(job.doc, results)
            name = job.doc["name"]
            (job.dirs.json_dir / make_artifact_filename(name, ".json")).write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            result2md(
                [name],
                [results],
                save_dir=str(job.dirs.md_dir),
                keep_header_footer=job.config.keep_header_footer,
            )
            all_results_path = job.dirs.out_dir / "all_results.json"
            all_results_path.write_text(
                json.dumps([record], ensure_ascii=False, indent=1), encoding="utf-8"
            )
            job.future.set_result({
                "out_dir": job.dirs.out_dir,
                "json_dir": job.dirs.json_dir,
                "md_dir": job.dirs.md_dir,
                "image_dir": job.dirs.image_dir,
                "elapsed": time.time() - job.started_at,
                "all_results_path": all_results_path,
            })
        except Exception as exc:
            self._fail_job(job, "output-writer", exc)

    def close(self):
        with self._jobs_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting_jobs = False
            active_jobs = list(self._active_jobs)

        for job in active_jobs:
            self._fail_job(
                job,
                "shutdown",
                RuntimeError("Service pipeline is shutting down."),
            )

        self.stop_event.set()
        for target_q in (self.request_q, self.preprocess_q, self.parse_q):
            try:
                target_q.put_nowait(self._sentinel)
            except queue.Full:
                pass
        for thread in self._threads:
            thread.join(timeout=5)
        self.pdf_renderer.close()
        self.page_executor.shutdown(wait=True)
        self.output_executor.shutdown(wait=True)


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
    already_compressed = {
        ".7z", ".avi", ".gif", ".gz", ".jpeg", ".jpg", ".mp3", ".mp4",
        ".pdf", ".png", ".rar", ".webp", ".zip",
    }
    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as zipf:
        for path in src_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in already_compressed
                    else zipfile.ZIP_DEFLATED
                )
                zipf.write(
                    path,
                    path.relative_to(src_dir),
                    compress_type=compression,
                    compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None,
                )


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


def _format_single_task_markdown(outputs: list[str]):
    if not outputs:
        return ""
    if len(outputs) == 1:
        content = (outputs[0] or "").strip()
    else:
        parts = []
        for idx, raw in enumerate(outputs, 1):
            parts.append(f"## Page {idx}\n\n{(raw or '').strip()}")
        content = "\n\n".join(parts).strip()
    return content + ("\n" if content else "")


def _format_single_task_outputs(task: str, outputs: list[str]) -> list[str]:
    label = {"text": "Text", "formula": "Formula", "table": "Table"}[task]
    formatted = []
    for page_idx, raw in enumerate(outputs):
        formatted.append(_format_block_content(
            {
                "label": label,
                "need_infer": True,
                "page_idx": page_idx,
                "image": None,
            },
            raw,
            "single_task",
            None,
            False,
            None,
        ))
    return formatted


def _run_single_task_with_model(input_path, output_path, task, model):
    task = task.lower()
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unsupported task: {task}. Choose from: {', '.join(TASK_PROMPTS)}")

    out_dir = Path(output_path).expanduser().resolve()
    md_dir = out_dir / "markdowns"
    json_dir = out_dir / "jsons"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    files = _list_single_task_inputs(input_path)
    if not files:
        raise ValueError(f"No supported input files found: {input_path}")

    started = time.time()
    results = []
    for file_path in files:
        images = _load_task_images(file_path)
        raw_outputs = model.batch_inference(
            images,
            [TASK_PROMPTS[task]] * len(images),
            min_pixels=1003520,
            max_tokens=4096 if task == "table" else None,
        )
        outputs = _format_single_task_outputs(task, raw_outputs)
        md_text = _format_single_task_markdown(outputs)
        md_path = md_dir / make_artifact_filename(file_path.stem, f"_{task}_result.md")
        json_path = json_dir / make_artifact_filename(file_path.stem, f"_{task}_result.json")
        md_path.write_text(md_text, encoding="utf-8")
        json_path.write_text(
            json.dumps({
                "image_name": file_path.name,
                "image_path": str(file_path),
                "task": task,
                "outputs": outputs,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        results.append({
            "input_path": str(file_path),
            "task": task,
            "outputs": outputs,
            "markdown_path": str(md_path),
            "json_path": str(json_path),
        })

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


def run_single_task_recognition(
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    backend_config: BackendConfig,
    *,
    backend_manager: BackendManager = DEFAULT_BACKEND_MANAGER,
):
    configure_runtime(backend_config)
    _, model = backend_manager.get(backend_config)
    return _run_single_task_with_model(input_path, output_path, task, model)

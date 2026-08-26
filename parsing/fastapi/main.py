#!/usr/bin/env python3
"""
FastAPI service for MonkeyOCRv2 parsing.

Run from parsing/fastapi:
    python main.py --server-url http://127.0.0.1:8888

Or:
    uvicorn main:app --host 0.0.0.0 --port 7861
"""

import argparse
import asyncio
import base64
import io
import json
import mimetypes
import os
import re
import sys
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PARSING_DIR = Path(__file__).resolve().parents[1]
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))

from core_runner import (  # noqa: E402
    BackendConfig,
    BackendManager,
    PipelineConfig,
    ServicePipelinePool,
    TASK_PROMPTS,
    make_artifact_filename,
    zip_dir,
)


DEFAULT_MODEL_PATH = str(PARSING_DIR.parent / "model_weight" / "MonkeyOCRv2-B-Parsing")
DEFAULT_OUTPUT_DIR = str(PARSING_DIR / "output" / "fastapi_outputs")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}
MAX_COMPONENT_BYTES = 255


class TaskResponse(BaseModel):
    success: bool
    task_type: str
    content: str
    message: Optional[str] = None


class ParseResponse(BaseModel):
    success: bool
    message: str
    output_dir: Optional[str] = None
    files: Optional[List[str]] = None
    download_url: Optional[str] = None


class Settings:
    def __init__(self):
        self.model_path = os.getenv("MOCR2_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.server_url = os.getenv("MOCR2_SERVER_URL", "")
        self.served_model_name = os.getenv("MOCR2_SERVED_MODEL_NAME", "MonkeyOCRv2")
        self.max_pixels = int(os.getenv("MOCR2_MAX_PIXELS", "1003520"))
        self.request_timeout = int(os.getenv("MOCR2_REQUEST_TIMEOUT", "300"))
        self.http_max_retries = int(os.getenv("MOCR2_HTTP_MAX_RETRIES", "5"))
        self.http_retry_backoff = float(os.getenv("MOCR2_HTTP_RETRY_BACKOFF", "1.0"))
        self.server_max_inflight = int(os.getenv("MOCR2_SERVER_MAX_INFLIGHT", "1024"))
        self.page_max_inflight = int(os.getenv("MOCR2_PAGE_MAX_INFLIGHT", "256"))
        self.preprocess_batch_size = int(os.getenv("MOCR2_PREPROCESS_BATCH_SIZE", "32"))
        self.api_workers = int(os.getenv("MOCR2_API_WORKERS", "128"))
        self.preprocess_wait_seconds = float(os.getenv("MOCR2_PREPROCESS_WAIT_SECONDS", "1.0"))
        self.skip_preprocess = os.getenv("MOCR2_SKIP_PREPROCESS", "0").lower() in {"1", "true", "yes"}
        self.end2end = os.getenv("MOCR2_END2END", "0").lower() in {"1", "true", "yes"}
        self.retry_repeat = os.getenv("MOCR2_RETRY_REPEAT", "0").lower() in {"1", "true", "yes"}
        self.keep_header_footer = os.getenv("MOCR2_KEEP_HEADER_FOOTER", "0").lower() in {"1", "true", "yes"}
        self.use_base64 = os.getenv("MOCR2_USE_BASE64", "0").lower() in {"1", "true", "yes"}
        self.debug = os.getenv("MOCR2_DEBUG", "0").lower() in {"1", "true", "yes"}
        self.output_dir = os.getenv("MOCR2_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


settings = Settings()
executor = None
backend_manager = BackendManager()
service_pool = None
backend = {
    "model": None,
    "loaded": False,
    "started_at": None,
}


def configure_from_args(argv=None):
    parser = argparse.ArgumentParser(description="Start the MonkeyOCRv2 FastAPI service.")
    parser.add_argument("--model-path", default=settings.model_path, help="Path to the model weights used by the preprocessor.")
    parser.add_argument("--server-url", "-s", dest="server_url", default=settings.server_url, help="Required vLLM OpenAI-compatible server URL, for example http://127.0.0.1:8888.")
    parser.add_argument("--served-model-name", default=settings.served_model_name, help="Model name exposed by vLLM serve.")
    parser.add_argument("--max-pixels", type=int, default=settings.max_pixels, help="Maximum input image pixels; larger images are resized proportionally.")
    parser.add_argument("--request-timeout", type=int, default=settings.request_timeout, help="HTTP request timeout in seconds when using vLLM serve.")
    parser.add_argument("--http-max-retries", type=int, default=settings.http_max_retries, help="Maximum retries for transient vLLM server HTTP failures.")
    parser.add_argument("--http-retry-backoff", type=float, default=settings.http_retry_backoff, help="Base exponential backoff seconds for transient vLLM server HTTP failures.")
    parser.add_argument("--server-max-inflight", type=int, default=settings.server_max_inflight, help="Maximum in-flight model requests submitted by this API process.")
    parser.add_argument("--page-max-inflight", type=int, default=settings.page_max_inflight, help="Maximum pages kept in the parsing pipeline at the same time.")
    parser.add_argument("--preprocess-batch-size", type=int, default=settings.preprocess_batch_size, help="Batch size used by the image preprocessor.")
    parser.add_argument("--api-workers", type=int, default=settings.api_workers, help="Maximum API request handlers running blocking pipeline work concurrently.")
    parser.add_argument("--preprocess-wait-seconds", type=float, default=settings.preprocess_wait_seconds, help="Maximum seconds to wait for a service preprocess batch to fill.")
    parser.add_argument("--skip-preprocess", action="store_true", default=settings.skip_preprocess, help="Skip image preprocessing before layout and recognition.")
    parser.add_argument("--end2end", action="store_true", default=settings.end2end, help="Use end-to-end parsing prompt instead of layout followed by block recognition.")
    parser.add_argument("--retry-repeat", action="store_true", default=settings.retry_repeat, help="Retry recognition when the generated output contains suspicious repetition. Disabled by default.")
    parser.add_argument("--keep-header-footer", action="store_true", default=settings.keep_header_footer, help="Keep Page-header and Page-footer blocks in markdown output.")
    parser.add_argument("--use-base64", action="store_true", default=settings.use_base64, help="Embed Picture blocks as base64 in markdown instead of saving image files.")
    parser.add_argument("--debug", action="store_true", default=settings.debug, help="Print full service-pipeline tracebacks and expose exception details in HTTP 500 responses.")
    parser.add_argument("--output-dir", default=settings.output_dir, help="Directory where API request outputs are saved.")
    parser.add_argument("--api-host", default=os.getenv("MOCR2_API_HOST", "0.0.0.0"), help="Host address for the FastAPI server.")
    parser.add_argument("--api-port", "-p", type=int, default=int(os.getenv("MOCR2_API_PORT", "8000")), help="Port for the FastAPI server.")
    args, _ = parser.parse_known_args(argv)

    settings.model_path = args.model_path
    settings.server_url = args.server_url
    settings.served_model_name = args.served_model_name
    settings.max_pixels = args.max_pixels
    settings.request_timeout = args.request_timeout
    settings.http_max_retries = args.http_max_retries
    settings.http_retry_backoff = args.http_retry_backoff
    settings.server_max_inflight = args.server_max_inflight
    settings.page_max_inflight = args.page_max_inflight
    settings.preprocess_batch_size = args.preprocess_batch_size
    settings.api_workers = max(1, args.api_workers)
    settings.preprocess_wait_seconds = max(0.0, args.preprocess_wait_seconds)
    settings.skip_preprocess = args.skip_preprocess
    settings.end2end = args.end2end
    settings.retry_repeat = args.retry_repeat
    settings.keep_header_footer = args.keep_header_footer
    settings.use_base64 = args.use_base64
    settings.debug = args.debug
    settings.output_dir = args.output_dir
    return args


def get_backend_config() -> BackendConfig:
    return BackendConfig(
        model_path=settings.model_path,
        server_url=settings.server_url,
        served_model_name=settings.served_model_name,
        max_pixels=settings.max_pixels,
        request_timeout=settings.request_timeout,
        http_max_retries=settings.http_max_retries,
        http_retry_backoff=settings.http_retry_backoff,
        server_max_inflight=settings.server_max_inflight,
        preprocess_batch_size=settings.preprocess_batch_size,
        skip_preprocess=settings.skip_preprocess,
    )


def initialize_backend():
    global service_pool
    if backend["loaded"]:
        return

    start = time.time()
    backend_config = get_backend_config()
    _, model = backend_manager.get(backend_config)
    service_pool = ServicePipelinePool(
        backend_config,
        settings.page_max_inflight,
        backend_manager=backend_manager,
        batch_wait_seconds=settings.preprocess_wait_seconds,
        debug=settings.debug,
    )
    backend["model"] = model
    backend["loaded"] = True
    backend["started_at"] = time.time()
    print(f"MonkeyOCRv2 FastAPI backend initialized in {time.time() - start:.2f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_backend()
    yield
    if service_pool is not None:
        service_pool.close()
    executor.shutdown(wait=True)
    backend_manager.close()


# Do not consume the embedding process' argv during import (pytest, gunicorn,
# and RAGFlow workers all import ``app``). CLI arguments are parsed in __main__.
cli_args = configure_from_args([])
executor = ThreadPoolExecutor(max_workers=settings.api_workers)
api_admission = asyncio.Semaphore(settings.api_workers)

app = FastAPI(
    title="MonkeyOCRv2 API",
    description="OCR and document parsing API using a vLLM OpenAI-compatible server.",
    version="2.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.output_dir), name="static")


@app.middleware("http")
async def limit_parse_concurrency(request, call_next):
    limited_paths = {"/parse", "/ocr/text", "/ocr/formula", "/ocr/table"}
    if request.url.path not in limited_paths:
        return await call_next(request)
    if api_admission.locked():
        return JSONResponse(
            status_code=503,
            content={"detail": "API parsing capacity is full. Retry later."},
            headers={"Retry-After": "1"},
        )
    await api_admission.acquire()
    try:
        return await call_next(request)
    finally:
        api_admission.release()


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if backend["loaded"] else "initializing",
        "backend": "server",
        "server_url": settings.server_url or None,
        "model_path": settings.model_path,
        "served_model_name": settings.served_model_name,
        "server_max_inflight": settings.server_max_inflight,
        "page_max_inflight": settings.page_max_inflight,
        "api_workers": settings.api_workers,
        "preprocess_wait_seconds": settings.preprocess_wait_seconds,
        "skip_preprocess": settings.skip_preprocess,
        "retry_repeat": settings.retry_repeat,
        "debug": settings.debug,
    }


@app.post("/ocr/text", response_model=TaskResponse)
async def extract_text(file: UploadFile = File(...)):
    return await perform_ocr_task(file, "text")


@app.post("/ocr/formula", response_model=TaskResponse)
async def extract_formula(file: UploadFile = File(...)):
    return await perform_ocr_task(file, "formula")


@app.post("/ocr/table", response_model=TaskResponse)
async def extract_table(file: UploadFile = File(...)):
    return await perform_ocr_task(file, "table")


@app.post("/parse")
async def parse_document(
    files: Optional[List[UploadFile]] = File(default=None), file: Optional[UploadFile] = File(default=None),
    start_page_id: int = Form(0), end_page_id: int = Form(99999),
):
    """Parse documents and return one ZIP archive.

    Multipart inputs: ``files`` (repeatable) or ``file`` (single file), plus
    optional ``start_page_id`` (inclusive, zero-based) and ``end_page_id``
    (exclusive). The response is the ZIP binary itself with Content-Type
    ``application/zip``; each document directory contains Markdown, native
    ``jsons``/``all_results.json`` artifacts, and generated images.
    """
    uploads = [*(files or [])]
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=422, detail="At least one file is required (field: files).")
    if start_page_id < 0 or end_page_id <= start_page_id:
        raise HTTPException(status_code=422, detail="Invalid page range: end_page_id must be greater than start_page_id.")

    result = {}
    for upload in uploads:
        parsed = await run_document_pipeline(upload, start_page_id=start_page_id, end_page_id=end_page_id)
        if isinstance(parsed, tuple):
            markdown, images, artifacts = parsed
        else:  # backwards-compatible hook for custom pipeline integrations
            markdown, images, artifacts = parsed, {}, {}
        # RAGFlow keys results by the original filename.  Ensure duplicate
        # names in a batch do not silently overwrite one another.
        name = upload.filename or "upload"
        if name in result:
            stem, suffix = Path(name).stem, Path(name).suffix
            index = 2
            while f"{stem}_{index}{suffix}" in result:
                index += 1
            name = f"{stem}_{index}{suffix}"
        result[name] = (markdown, images, artifacts)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, (markdown, images, artifacts) in result.items():
            stem = Path(name).stem
            root = f"{stem}/"
            zf.writestr(root + f"{stem}.md", markdown)
            for rel, content in artifacts.items():
                zf.writestr(root + rel, content)
            for image_name, data_uri in images.items():
                if ";base64," not in data_uri:
                    continue
                encoded = data_uri.split(";base64,", 1)[1]
                zf.writestr(root + image_name, base64.b64decode(encoded))
            if not any("content_list.json" in rel for rel in artifacts):
                zf.writestr(root + f"{stem}_content_list.json", json.dumps([
                    {"type": "text", "text": markdown, "page_idx": 0, "bbox": [0, 0, 1000, 1000]}
                ], ensure_ascii=False))
    return Response(content=archive.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=monkeyocrv2_results.zip"})


async def perform_ocr_task(file: UploadFile, task_type: str):
    if task_type not in TASK_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unsupported OCR task: {task_type}")

    run_id = make_run_id(suffix=f"_{task_type}")
    run_dir = Path(settings.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = await save_upload(file, run_dir)
    if input_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="OCR task endpoints currently accept image files only.")

    def run_task():
        return service_pool.run_single_task(
            str(input_path),
            str(run_dir),
            task_type,
        )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_task)
        outputs = result["results"][0]["outputs"] if result["results"] else []
        content = "\n\n".join(x.strip() for x in outputs if x is not None)
        return TaskResponse(success=True, task_type=task_type, content=content)
    except Exception as exc:
        raise_internal_error("single-task-ocr", exc)


async def parse_document_internal(file: UploadFile):
    run_id = make_run_id()
    run_dir = Path(settings.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = await save_upload(file, run_dir)
    if input_path.suffix.lower() not in INPUT_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image files.")

    def run_parse():
        service_pool.run(
            PipelineConfig(
                input_path=str(input_path),
                output_path=str(run_dir),
                backend=get_backend_config(),
                page_max_inflight=settings.page_max_inflight,
                draw_layout=False,
                end2end=settings.end2end,
                skip_processed=False,
                retry_repeat=settings.retry_repeat,
                retry_repeat_max_retries=3,
                keep_header_footer=settings.keep_header_footer,
                use_base64=settings.use_base64,
                verbose=False,
            )
        )
        zip_path = run_dir / make_artifact_filename(input_path.stem, "_results.zip")
        zip_dir(run_dir, zip_path)
        files = [
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        ]
        return zip_path, files

    try:
        loop = asyncio.get_running_loop()
        zip_path, files = await loop.run_in_executor(executor, run_parse)
        return ParseResponse(
            success=True,
            message="Document parsed successfully.",
            output_dir=run_id,
            files=files,
            download_url=f"/static/{run_id}/{zip_path.name}",
        )
    except Exception as exc:
        raise_internal_error("parse", exc)


async def run_document_pipeline(file: UploadFile, *, start_page_id: int = 0, end_page_id: int = 99999):
    """Run the parsing pipeline and return the generated Markdown text."""
    run_id = make_run_id()
    run_dir = Path(settings.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = await save_upload(file, run_dir)
    if input_path.suffix.lower() not in INPUT_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image files.")

    def run_parse():
        pipeline_input = input_path
        if input_path.suffix.lower() == ".pdf" and (start_page_id > 0 or end_page_id < 99999):
            from pypdf import PdfReader, PdfWriter
            reader, writer = PdfReader(str(input_path)), PdfWriter()
            stop = min(end_page_id, len(reader.pages))
            if start_page_id >= stop:
                raise ValueError("Requested page range is outside the document.")
            for page in reader.pages[start_page_id:stop]:
                writer.add_page(page)
            pipeline_input = run_dir / "paged_input.pdf"
            with pipeline_input.open("wb") as sliced:
                writer.write(sliced)
        service_pool.run(PipelineConfig(
            input_path=str(pipeline_input), output_path=str(run_dir),
            backend=get_backend_config(), page_max_inflight=settings.page_max_inflight,
            draw_layout=False, end2end=settings.end2end, skip_processed=False,
            retry_repeat=settings.retry_repeat, retry_repeat_max_retries=3,
            keep_header_footer=settings.keep_header_footer, use_base64=settings.use_base64,
            verbose=False,
        ))
        markdowns = sorted(p for p in run_dir.rglob("*.md") if p.is_file())
        if not markdowns:
            raise RuntimeError("Parser completed without producing a Markdown file.")
        markdown = "\n\n".join(p.read_text(encoding="utf-8") for p in markdowns)
        images = {}
        for image in run_dir.rglob("*"):
            if not image.is_file() or image in markdowns or image == input_path:
                continue
            mime, _ = mimetypes.guess_type(image.name)
            if not mime or not mime.startswith("image/"):
                continue
            relative = image.relative_to(run_dir).as_posix()
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            images[relative] = f"data:{mime};base64,{encoded}"
        artifacts = {}
        for artifact in run_dir.rglob("*"):
            if artifact.is_file() and artifact != input_path and artifact != pipeline_input:
                artifacts[artifact.relative_to(run_dir).as_posix()] = artifact.read_bytes()
        return markdown, images, artifacts

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, run_parse)
    except HTTPException:
        raise
    except Exception as exc:
        raise_internal_error("parse", exc)


def raise_internal_error(stage: str, exc: Exception):
    if settings.debug:
        print(f"[API:{stage}] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        detail = f"{type(exc).__name__}: {exc}"
    else:
        detail = "Internal server error."
    raise HTTPException(status_code=500, detail=detail) from exc


def make_run_id(suffix: str = "") -> str:
    prefix = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    suffix = _safe_filename_component(suffix, max_bytes=32, fallback="")
    return f"{prefix}{suffix}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _safe_filename_component(value: str, *, max_bytes: int, fallback: str) -> str:
    value = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", str(value or ""))
    value = value.strip().strip(".")
    value = _truncate_utf8(value, max_bytes).rstrip(" .")
    return value or fallback


async def save_upload(file: UploadFile, output_dir: Path) -> Path:
    suffix = Path(file.filename or "upload").suffix.lower()
    if not suffix or len(suffix.encode("utf-8")) > 16:
        suffix = ".bin"
    stem = _safe_filename_component(
        Path(file.filename or "upload").stem,
        max_bytes=MAX_COMPONENT_BYTES - len(suffix.encode("utf-8")),
        fallback="upload",
    )
    dst = output_dir / f"{stem}{suffix}"
    async with aiofiles.open(dst, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)
    return dst


if __name__ == "__main__":
    import uvicorn

    cli_args = configure_from_args()
    uvicorn.run(
        app,
        host=cli_args.api_host,
        port=cli_args.api_port,
        log_level="debug" if settings.debug else "info",
    )

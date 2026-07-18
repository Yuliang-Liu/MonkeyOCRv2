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
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PARSING_DIR = Path(__file__).resolve().parents[1]
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))

from core_runner import (  # noqa: E402
    BackendConfig,
    BackendManager,
    PipelineConfig,
    TASK_PROMPTS,
    run_pipeline,
    run_single_task_recognition,
    zip_dir,
)


DEFAULT_MODEL_PATH = str(PARSING_DIR.parent / "model_weight" / "MonkeyOCRv2-B-Parsing")
DEFAULT_OUTPUT_DIR = str(PARSING_DIR / "output" / "fastapi_outputs")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_EXTS = IMAGE_EXTS | {".pdf"}
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
        self.tp = int(os.getenv("MOCR2_TP", "1"))
        self.max_pixels = int(os.getenv("MOCR2_MAX_PIXELS", "1003520"))
        self.request_timeout = int(os.getenv("MOCR2_REQUEST_TIMEOUT", "300"))
        self.http_max_retries = int(os.getenv("MOCR2_HTTP_MAX_RETRIES", "5"))
        self.http_retry_backoff = float(os.getenv("MOCR2_HTTP_RETRY_BACKOFF", "1.0"))
        self.server_max_inflight = int(os.getenv("MOCR2_SERVER_MAX_INFLIGHT", "1024"))
        self.page_max_inflight = int(os.getenv("MOCR2_PAGE_MAX_INFLIGHT", "64"))
        self.preprocess_batch_size = int(os.getenv("MOCR2_PREPROCESS_BATCH_SIZE", "32"))
        self.skip_preprocess = os.getenv("MOCR2_SKIP_PREPROCESS", "0").lower() in {"1", "true", "yes"}
        self.end2end = os.getenv("MOCR2_END2END", "0").lower() in {"1", "true", "yes"}
        self.keep_header_footer = os.getenv("MOCR2_KEEP_HEADER_FOOTER", "0").lower() in {"1", "true", "yes"}
        self.use_base64 = os.getenv("MOCR2_USE_BASE64", "0").lower() in {"1", "true", "yes"}
        self.output_dir = os.getenv("MOCR2_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


settings = Settings()
executor = ThreadPoolExecutor(max_workers=int(os.getenv("MOCR2_API_WORKERS", "4")))
parse_semaphore = threading.Semaphore(int(os.getenv("MOCR2_API_PARSE_CONCURRENCY", "1")))
backend_manager = BackendManager()
backend = {
    "model": None,
    "loaded": False,
    "started_at": None,
}


def configure_from_args():
    parser = argparse.ArgumentParser(description="Start the MonkeyOCRv2 FastAPI service.")
    parser.add_argument("--model-path", default=settings.model_path, help="Path to the MonkeyOCRv2 model weights used by local Async engine and preprocessor.")
    parser.add_argument("--server-url", "-s", dest="server_url", default=settings.server_url, help="vLLM OpenAI-compatible server URL, for example http://127.0.0.1:8888. If omitted, local AsyncLLMEngine is used.")
    parser.add_argument("--served-model-name", default=settings.served_model_name, help="Model name exposed by vLLM serve.")
    parser.add_argument("--tp", type=int, default=settings.tp, help="Tensor parallel size for local AsyncLLMEngine fallback.")
    parser.add_argument("--max-pixels", type=int, default=settings.max_pixels, help="Maximum input image pixels; larger images are resized proportionally.")
    parser.add_argument("--request-timeout", type=int, default=settings.request_timeout, help="HTTP request timeout in seconds when using vLLM serve.")
    parser.add_argument("--http-max-retries", type=int, default=settings.http_max_retries, help="Maximum retries for transient vLLM server HTTP failures.")
    parser.add_argument("--http-retry-backoff", type=float, default=settings.http_retry_backoff, help="Base exponential backoff seconds for transient vLLM server HTTP failures.")
    parser.add_argument("--server-max-inflight", type=int, default=settings.server_max_inflight, help="Maximum in-flight model requests submitted by this API process.")
    parser.add_argument("--page-max-inflight", type=int, default=settings.page_max_inflight, help="Maximum pages kept in the parsing pipeline at the same time.")
    parser.add_argument("--preprocess-batch-size", type=int, default=settings.preprocess_batch_size, help="Batch size used by the image preprocessor.")
    parser.add_argument("--skip-preprocess", action="store_true", default=settings.skip_preprocess, help="Skip image preprocessing before layout and recognition.")
    parser.add_argument("--end2end", action="store_true", default=settings.end2end, help="Use end-to-end parsing prompt instead of layout followed by block recognition.")
    parser.add_argument("--keep-header-footer", action="store_true", default=settings.keep_header_footer, help="Keep Page-header and Page-footer blocks in markdown output.")
    parser.add_argument("--use-base64", action="store_true", default=settings.use_base64, help="Embed Picture blocks as base64 in markdown instead of saving image files.")
    parser.add_argument("--output-dir", default=settings.output_dir, help="Directory where API request outputs are saved.")
    parser.add_argument("--api-host", default=os.getenv("MOCR2_API_HOST", "0.0.0.0"), help="Host address for the FastAPI server.")
    parser.add_argument("--api-port", "-p", type=int, default=int(os.getenv("MOCR2_API_PORT", "8000")), help="Port for the FastAPI server.")
    args, _ = parser.parse_known_args()

    settings.model_path = args.model_path
    settings.server_url = args.server_url
    settings.served_model_name = args.served_model_name
    settings.tp = args.tp
    settings.max_pixels = args.max_pixels
    settings.request_timeout = args.request_timeout
    settings.http_max_retries = args.http_max_retries
    settings.http_retry_backoff = args.http_retry_backoff
    settings.server_max_inflight = args.server_max_inflight
    settings.page_max_inflight = args.page_max_inflight
    settings.preprocess_batch_size = args.preprocess_batch_size
    settings.skip_preprocess = args.skip_preprocess
    settings.end2end = args.end2end
    settings.keep_header_footer = args.keep_header_footer
    settings.use_base64 = args.use_base64
    settings.output_dir = args.output_dir
    return args


def get_backend_config() -> BackendConfig:
    return BackendConfig(
        model_path=settings.model_path,
        server_url=settings.server_url,
        served_model_name=settings.served_model_name,
        tp=settings.tp,
        max_pixels=settings.max_pixels,
        request_timeout=settings.request_timeout,
        http_max_retries=settings.http_max_retries,
        http_retry_backoff=settings.http_retry_backoff,
        server_max_inflight=settings.server_max_inflight,
        preprocess_batch_size=settings.preprocess_batch_size,
        skip_preprocess=settings.skip_preprocess,
    )


def initialize_backend():
    if backend["loaded"]:
        return

    start = time.time()
    _, model = backend_manager.get(get_backend_config())
    backend["model"] = model
    backend["loaded"] = True
    backend["started_at"] = time.time()
    print(f"MonkeyOCRv2 FastAPI backend initialized in {time.time() - start:.2f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_backend()
    yield
    executor.shutdown(wait=True)
    backend_manager.close()


cli_args = configure_from_args()

app = FastAPI(
    title="MonkeyOCRv2 API",
    description="OCR and document parsing API using MonkeyOCRv2 server/Async inference.",
    version="2.0.0",
    lifespan=lifespan,
)

Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.output_dir), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if backend["loaded"] else "initializing",
        "backend": "server" if settings.server_url else "async",
        "server_url": settings.server_url or None,
        "model_path": settings.model_path,
        "served_model_name": settings.served_model_name,
        "server_max_inflight": settings.server_max_inflight,
        "page_max_inflight": settings.page_max_inflight,
        "skip_preprocess": settings.skip_preprocess,
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


@app.post("/parse", response_model=ParseResponse)
async def parse_document(file: UploadFile = File(...)):
    return await parse_document_internal(file)


@app.post("/parse/split", response_model=ParseResponse)
async def parse_document_split(file: UploadFile = File(...)):
    return await parse_document_internal(file)


async def perform_ocr_task(file: UploadFile, task_type: str):
    if task_type not in TASK_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unsupported OCR task: {task_type}")

    run_id = make_run_id(file.filename or "upload", suffix=f"_{task_type}")
    run_dir = Path(settings.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = await save_upload(file, run_dir)
    if input_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="OCR task endpoints currently accept image files only.")

    def run_task():
        return run_single_task_recognition(
            str(input_path),
            str(run_dir),
            task_type,
            get_backend_config(),
            backend_manager=backend_manager,
            parse_semaphore=parse_semaphore,
        )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_task)
        outputs = result["results"][0]["outputs"] if result["results"] else []
        content = "\n\n".join(x.strip() for x in outputs if x is not None)
        return TaskResponse(success=True, task_type=task_type, content=content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def parse_document_internal(file: UploadFile):
    run_id = make_run_id(file.filename or "upload")
    run_dir = Path(settings.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = await save_upload(file, run_dir)
    if input_path.suffix.lower() not in INPUT_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image files.")

    def run_parse():
        run_pipeline(
            PipelineConfig(
                input_path=str(input_path),
                output_path=str(run_dir),
                backend=get_backend_config(),
                page_max_inflight=settings.page_max_inflight,
                draw_layout=False,
                end2end=settings.end2end,
                skip_processed=False,
                retry_repeat=True,
                retry_repeat_max_retries=3,
                keep_header_footer=settings.keep_header_footer,
                use_base64=settings.use_base64,
                verbose=False,
            ),
            backend_manager=backend_manager,
            parse_semaphore=parse_semaphore,
        )

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, run_parse)
        zip_path = run_dir / f"{input_path.stem}_results.zip"
        zip_dir(run_dir, zip_path)
        files = [str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()]
        return ParseResponse(
            success=True,
            message="Document parsed successfully.",
            output_dir=run_id,
            files=files,
            download_url=f"/static/{run_id}/{zip_path.name}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def make_run_id(filename: str, suffix: str = "") -> str:
    stem = Path(filename or "upload").stem or "upload"
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


async def save_upload(file: UploadFile, output_dir: Path) -> Path:
    suffix = Path(file.filename or "upload").suffix.lower()
    if not suffix:
        suffix = ".bin"
    stem = Path(file.filename or "upload").stem or "upload"
    dst = output_dir / f"{stem}{suffix}"
    with dst.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dst


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cli_args.api_host, port=cli_args.api_port)

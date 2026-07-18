import argparse
import atexit
import base64
import json
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image


PARSING_DIR = Path(__file__).resolve().parents[1]
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))
DEFAULT_MODEL_PATH = str(PARSING_DIR.parent / "model_weight" / "MonkeyOCRv2-B-Parsing")
DEFAULT_OUTPUT_DIR = str(PARSING_DIR / "output" / "demo_outputs")
EXAMPLES_DIR = Path(PARSING_DIR.parent / "images_test")
INITIAL_MARKDOWN = "Please upload a file and click Parse."
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXAMPLE_EXTS = IMAGE_EXTS | {".pdf"}


try:
    from core_runner import (
        BackendConfig,
        PipelineConfig,
        DEFAULT_BACKEND_MANAGER,
        load_all_results,
        load_markdowns,
        open_oriented_image,
        run_pipeline,
        run_single_task_recognition,
        zip_dir,
    )
except ModuleNotFoundError as exc:
    raise gr.Error(
        f"Missing dependency while loading MonkeyOCRv2-Parsing parsing pipeline: {exc.name}. "
        "Please run this demo in the environment used by parsing/parse.py."
    ) from exc

PIPELINE_LOCK = threading.Lock()
atexit.register(DEFAULT_BACKEND_MANAGER.close)


def _list_examples():
    if not EXAMPLES_DIR.exists():
        return {}
    return {
        path.name: str(path)
        for path in sorted(EXAMPLES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in EXAMPLE_EXTS
    }


EXAMPLE_FILES = _list_examples()


CSS = """
#page_info_html {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    margin: 0 12px;
}

#page_info_box {
    padding: 8px 20px;
    font-size: 16px;
    border: 1px solid #bbb;
    border-radius: 8px;
    background-color: #f8f8f8;
    text-align: center;
    min-width: 80px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

#markdown_output {
    min-height: 800px;
    max-height: 800px;
    overflow: auto;
}

#arabic_markdown_output {
    min-height: 800px;
    max-height: 800px;
    overflow: auto;
    direction: rtl;
    unicode-bidi: plaintext;
    text-align: right;
}

#arabic_markdown_output * {
    direction: rtl;
    unicode-bidi: plaintext;
    text-align: right;
}

#raw_markdown_output {
    min-height: 800px;
}

#raw_markdown_output textarea {
    min-height: 800px !important;
    height: 800px !important;
    overflow: auto !important;
}

footer {
    visibility: hidden;
}
"""


def _page_info(current: int = 0, total: int = 0) -> str:
    return f"<div id='page_info_box'>{current}/{total}</div>"


def create_session_state():
    return {
        "id": f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "file_path": None,
        "run_dir": None,
        "pdf_cache": {
            "images": [],
            "current_page": 0,
            "total_pages": 0,
        },
    }


def _reset_page_cache(session_state):
    session_state["pdf_cache"]["images"] = []
    session_state["pdf_cache"]["current_page"] = 0
    session_state["pdf_cache"]["total_pages"] = 0


def _set_page_cache(session_state, images):
    session_state["pdf_cache"]["images"] = images or []
    session_state["pdf_cache"]["current_page"] = 0
    session_state["pdf_cache"]["total_pages"] = len(session_state["pdf_cache"]["images"])


def _load_pdf_preview_pages(pdf_path: str):
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise gr.Error("PDF preview requires pypdfium2. Please install it or upload an image.") from exc

    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bmp = page.render(scale=200 / 72)
        pages.append(bmp.to_pil().convert("RGB"))
    return pages


def _load_image_preview(image_path: str):
    return open_oriented_image(image_path).convert("RGB")


def _relative_to_output_dir(path: str | Path, output_dir: str | Path) -> str:
    path = Path(path).resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _create_run_dir(file_path, session_state, output_dir=DEFAULT_OUTPUT_DIR):
    src = Path(file_path)
    session_state["id"] = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_name = f"{session_state['id']}_{src.stem}"
    run_dir = Path(output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    session_state["run_dir"] = str(run_dir)
    return run_dir


def _copy_input_to_run_dir(file_path, session_state, output_dir=DEFAULT_OUTPUT_DIR, new_run=False):
    if not file_path:
        return None

    src = Path(file_path).resolve()
    if not src.exists():
        return None

    run_dir = Path(session_state["run_dir"]).resolve() if session_state.get("run_dir") else None
    if new_run or run_dir is None:
        run_dir = _create_run_dir(src, session_state, output_dir)

    dst = run_dir / src.name
    if src != dst.resolve():
        if dst.exists():
            dst = run_dir / f"{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}"
        shutil.copy2(src, dst)

    session_state["file_path"] = str(dst)
    return str(dst)


def _relative_to_output_dir(path: str | Path, output_dir: str | Path) -> str:
    path = Path(path).resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _create_run_dir(file_path, session_state, output_dir=DEFAULT_OUTPUT_DIR):
    src = Path(file_path)
    session_state["id"] = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_name = f"{session_state['id']}_{src.stem}"
    run_dir = Path(output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    session_state["run_dir"] = str(run_dir)
    return run_dir


def _copy_input_to_run_dir(file_path, session_state, output_dir=DEFAULT_OUTPUT_DIR, new_run=False):
    if not file_path:
        return None

    src = Path(file_path).resolve()
    if not src.exists():
        return None

    run_dir = Path(session_state["run_dir"]).resolve() if session_state.get("run_dir") else None
    if new_run or run_dir is None:
        run_dir = _create_run_dir(src, session_state, output_dir)

    dst = run_dir / src.name
    if src != dst.resolve():
        if dst.exists():
            dst = run_dir / f"{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}"
        shutil.copy2(src, dst)

    session_state["file_path"] = str(dst)
    return str(dst)


def _load_example_preview(file_path: str, max_size=(260, 180)):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise gr.Error("PDF preview requires pypdfium2. Please install it or upload an image.") from exc

        pdf = pdfium.PdfDocument(file_path)
        if len(pdf) == 0:
            return None
        page = pdf[0]
        bmp = page.render(scale=100 / 72)
        image = bmp.to_pil().convert("RGB")
    else:
        image = open_oriented_image(file_path).convert("RGB")

    image.thumbnail(max_size, Image.LANCZOS)
    return image


def _preview_file(file_path, session_state):
    if not file_path:
        session_state["file_path"] = None
        _reset_page_cache(session_state)
        return None, _page_info(), session_state

    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf_preview_pages(file_path)
    elif suffix in IMAGE_EXTS:
        pages = [_load_image_preview(file_path)]
    else:
        session_state["file_path"] = None
        _reset_page_cache(session_state)
        return None, _page_info(), session_state

    session_state["file_path"] = file_path
    _set_page_cache(session_state, pages)
    return pages[0], _page_info(1, len(pages)), session_state


def preview_example(example_name):
    if not example_name:
        return None

    file_path = EXAMPLE_FILES.get(example_name)
    if not file_path:
        return None

    return _load_example_preview(file_path)


def choose_example(example_name, session_state, output_dir=DEFAULT_OUTPUT_DIR):
    if not example_name:
        return None, _page_info(), session_state

    file_path = EXAMPLE_FILES.get(example_name)
    if not file_path:
        return None, _page_info(), session_state

    saved_path = _copy_input_to_run_dir(file_path, session_state, output_dir, new_run=True)
    return _preview_file(saved_path, session_state)


def load_uploaded_file(file_path, session_state, output_dir=DEFAULT_OUTPUT_DIR):
    saved_path = _copy_input_to_run_dir(file_path, session_state, output_dir, new_run=True)
    preview_image, page_info, session_state = _preview_file(saved_path, session_state)
    return preview_image, page_info, session_state, gr.update(value=None), None


def turn_page(direction, session_state):
    cache = session_state["pdf_cache"]
    if not cache["images"]:
        return None, _page_info(), session_state

    if direction == "prev":
        cache["current_page"] = max(0, cache["current_page"] - 1)
    elif direction == "next":
        cache["current_page"] = min(cache["total_pages"] - 1, cache["current_page"] + 1)

    idx = cache["current_page"]
    return cache["images"][idx], _page_info(idx + 1, cache["total_pages"]), session_state


def _markdown_for_preview(md_text: str, md_dir: Path) -> str:
    def replace_image_with_base64(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        if img_path.startswith(("data:", "http://", "https://")):
            return match.group(0)

        full_img_path = Path(img_path)
        if not full_img_path.is_absolute():
            full_img_path = (md_dir / img_path).resolve()

        try:
            if not full_img_path.exists():
                return match.group(0)
            img_data = full_img_path.read_bytes()
            ext = full_img_path.suffix.lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext.lstrip('.')}"
            img_base64 = base64.b64encode(img_data).decode("ascii")
            return f"![{alt_text}](data:{mime_type};base64,{img_base64})"
        except Exception:
            return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image_with_base64, md_text)


def _contains_arabic(text: str) -> bool:
    text = re.sub(r"data:image/[^;\s)]+;base64,[A-Za-z0-9+/=]+", "", text or "")
    arabic_chars = re.findall(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]", text or "")
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not arabic_chars:
        return False
    if not letters:
        return True
    return len(arabic_chars) / max(len(letters), 1) >= 0.2

def _markdown_preview_updates(md_preview: str, md_text: str = None):
    if _contains_arabic(md_text):
        return (
            gr.update(value=md_preview, visible=False),
            gr.update(value=md_preview, visible=True),
        )
    return (
        gr.update(value=md_preview, visible=True),
        gr.update(value=md_preview, visible=False),
    )


def parse_file(
    file_path,
    session_state,
    keep_header_footer,
    model_path=DEFAULT_MODEL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    server_url="",
    served_model_name="MonkeyOCRv2",
    request_timeout=300,
    http_max_retries=5,
    http_retry_backoff=1.0,
    server_max_inflight=1024,
    page_max_inflight=16,
    preprocess_batch_size=8,
    skip_preprocess=False,
    end2end=False,
):
    file_path = session_state.get("file_path") or file_path
    file_path = _copy_input_to_run_dir(file_path, session_state, output_dir) or file_path

    if file_path is None:
        md_preview_ltr, md_preview_rtl = _markdown_preview_updates("Please upload a PDF or image first.")
        return (
            gr.update(),
            md_preview_ltr,
            md_preview_rtl,
            "Please upload a PDF or image first.",
            _page_info(),
            gr.update(value=None),
            gr.update(value=None),
            session_state,
        )

    print(f"Parsing file: {file_path}")

    start = time.time()
    tp = 1
    input_path = Path(file_path)
    if input_path.suffix.lower() not in EXAMPLE_EXTS:
        raise gr.Error("Unsupported file type. Please upload PDF or image files.")
    session_state["file_path"] = file_path
    run_dir = Path(session_state["run_dir"]).resolve() if session_state.get("run_dir") else _create_run_dir(input_path, session_state, output_dir)
    result_info = run_pipeline(
        PipelineConfig(
            input_path=str(input_path),
            output_path=str(run_dir),
            backend=BackendConfig(
                model_path=model_path,
                server_url=server_url,
                served_model_name=served_model_name,
                tp=tp,
                max_pixels=1003520,
                request_timeout=request_timeout,
                http_max_retries=http_max_retries,
                http_retry_backoff=http_retry_backoff,
                server_max_inflight=server_max_inflight,
                preprocess_batch_size=preprocess_batch_size,
                skip_preprocess=skip_preprocess,
            ),
            page_max_inflight=page_max_inflight,
            draw_layout=False,
            end2end=end2end,
            skip_processed=False,
            retry_repeat=True,
            retry_repeat_max_retries=3,
            keep_header_footer=keep_header_footer,
            use_base64=False,
        ),
        backend_manager=DEFAULT_BACKEND_MANAGER,
        parse_semaphore=PIPELINE_LOCK,
    )
    md_dir = result_info["md_dir"]

    result_records = load_all_results(run_dir)
    all_results_path = result_info["all_results_path"]
    if result_records:
        for record in result_records:
            record["image_path"] = _relative_to_output_dir(record.get("image_path", ""), output_dir)
        all_results_path.write_text(
            json.dumps(result_records, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    markdowns = load_markdowns(md_dir)

    zip_path = run_dir / f"{input_path.stem}_results.zip"
    zip_dir(run_dir, zip_path)

    preview_image, page_info, session_state = _preview_file(file_path, session_state)

    md_text = markdowns[0] if markdowns else ""
    md_preview = _markdown_for_preview(md_text, md_dir)
    elapsed = time.time() - start
    status = f"Parsed {max(1, len(result_records))} document(s) in {elapsed:.2f}s. Results saved to {run_dir}"
    md_preview_ltr, md_preview_rtl = _markdown_preview_updates(md_preview or status, md_text)

    return (
        preview_image,
        md_preview_ltr,
        md_preview_rtl,
        md_text or status,
        page_info,
        gr.update(value=str(zip_path), visible=True),
        gr.update(value=str(run_dir / "all_results.json"), visible=True),
        session_state,
    )


def recognize_single_task(
    file_path,
    session_state,
    task_label,
    model_path=DEFAULT_MODEL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    server_url="",
    served_model_name="MonkeyOCRv2",
    request_timeout=300,
    http_max_retries=5,
    http_retry_backoff=1.0,
    server_max_inflight=1024,
    preprocess_batch_size=8,
):
    file_path = session_state.get("file_path") or file_path
    file_path = _copy_input_to_run_dir(file_path, session_state, output_dir) or file_path

    if file_path is None:
        md_preview_ltr, md_preview_rtl = _markdown_preview_updates("Please upload a PDF or image first.")
        return (
            gr.update(),
            md_preview_ltr,
            md_preview_rtl,
            "Please upload a PDF or image first.",
            _page_info(),
            gr.update(value=None),
            gr.update(value=None),
            session_state,
        )

    task = (task_label or "Text").strip().lower()
    input_path = Path(file_path)
    if input_path.suffix.lower() not in EXAMPLE_EXTS:
        raise gr.Error("Unsupported file type. Please upload PDF or image files.")

    session_state["file_path"] = file_path
    run_dir = Path(session_state["run_dir"]).resolve() if session_state.get("run_dir") else _create_run_dir(input_path, session_state, output_dir)
    result_info = run_single_task_recognition(
        str(input_path),
        str(run_dir),
        task,
        BackendConfig(
            model_path=model_path,
            server_url=server_url,
            served_model_name=served_model_name,
            tp=1,
            max_pixels=1003520,
            request_timeout=request_timeout,
            http_max_retries=http_max_retries,
            http_retry_backoff=http_retry_backoff,
            server_max_inflight=server_max_inflight,
            preprocess_batch_size=preprocess_batch_size,
            skip_preprocess=True,
        ),
        backend_manager=DEFAULT_BACKEND_MANAGER,
        parse_semaphore=PIPELINE_LOCK,
    )

    markdowns = load_markdowns(result_info["md_dir"])
    zip_path = run_dir / f"{input_path.stem}_{task}_result.zip"
    zip_dir(run_dir, zip_path)
    preview_image, page_info, session_state = _preview_file(file_path, session_state)

    md_text = markdowns[0] if markdowns else ""
    md_preview = _markdown_for_preview(md_text, result_info["md_dir"])
    status = (
        f"Single task {task} recognition completed in {result_info['elapsed']:.2f}s. "
        f"Results saved to {run_dir}"
    )
    md_preview_ltr, md_preview_rtl = _markdown_preview_updates(md_preview or status, md_text)
    print(status)

    return (
        preview_image,
        md_preview_ltr,
        md_preview_rtl,
        md_text or status,
        page_info,
        gr.update(value=str(zip_path), visible=True),
        gr.update(value=str(result_info["all_results_path"]), visible=True),
        session_state,
    )


def clear_all(session_state):
    session_state["id"] = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_state["file_path"] = None
    session_state["run_dir"] = None
    _reset_page_cache(session_state)
    md_preview_ltr, md_preview_rtl = _markdown_preview_updates(INITIAL_MARKDOWN)
    return (
        None,
        None,
        md_preview_ltr,
        md_preview_rtl,
        "Waiting for parsing.",
        _page_info(),
        gr.update(value=None),
        gr.update(value=None),
        session_state,
        gr.update(value=None),
        None,
    )


def create_gradio_app(
    default_model_path=DEFAULT_MODEL_PATH,
    default_output_dir=DEFAULT_OUTPUT_DIR,
    server_url="",
    served_model_name="MonkeyOCRv2",
    request_timeout=300,
    http_max_retries=5,
    http_retry_backoff=1.0,
    server_max_inflight=1024,
    page_max_inflight=16,
    preprocess_batch_size=8,
    skip_preprocess=False,
    end2end=False,
):
    with gr.Blocks(title="MonkeyOCRv2-Parsing", theme="ocean", css=CSS) as demo:
        session_state = gr.State(create_session_state())
        initial_is_arabic = _contains_arabic(INITIAL_MARKDOWN)

        gr.HTML(
            """
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">MonkeyOCRv2-Parsing</h1>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, variant="compact"):
                gr.Markdown("### Upload")
                file_input = gr.File(
                    label="Select file",
                    type="filepath",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
                    show_label=True,
                )

                with gr.Accordion("Examples", open=True):
                    example_dropdown = gr.Dropdown(
                        label="Example file",
                        choices=list(EXAMPLE_FILES.keys()),
                        value=None,
                        show_label=True,
                    )
                    example_preview = gr.Image(
                        label="Preview",
                        height=180,
                        show_label=True,
                        interactive=False,
                    )
                    choose_example_button = gr.Button("Choose", variant="secondary")

                gr.Markdown("### Actions")
                keep_header_footer = gr.Checkbox(label="Keep header/footer", value=False)
                task_dropdown = gr.Dropdown(
                    label="Choose Task",
                    choices=["Text", "Formula", "Table"],
                    value="Text",
                    show_label=True,
                )
                parse_button = gr.Button("Parse", variant="primary")
                single_task_button = gr.Button("Single Task Recognition", variant="secondary")
                clear_button = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### File Preview")
                        file_preview = gr.Image(label="Preview", visible=True, height=800, show_label=False)
                        with gr.Row():
                            prev_btn = gr.Button("Previous")
                            page_info = gr.HTML(value=_page_info(), elem_id="page_info_html")
                            next_btn = gr.Button("Next")

                    with gr.Column(scale=3):
                        gr.Markdown("### Result Display")
                        with gr.Tabs(elem_id="markdown_tabs"):
                            with gr.TabItem("Markdown Preview"):
                                md_view_ltr = gr.Markdown(
                                    value=INITIAL_MARKDOWN,
                                    label="Markdown Preview",
                                    max_height=800,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                    ],
                                    visible=not initial_is_arabic,
                                    elem_id="markdown_output",
                                )
                                md_view_rtl = gr.Markdown(
                                    value=INITIAL_MARKDOWN,
                                    label="Markdown Preview",
                                    max_height=800,
                                    rtl=True,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                    ],
                                    visible=initial_is_arabic,
                                    elem_id="arabic_markdown_output",
                                )
                            with gr.TabItem("Markdown Raw Text"):
                                md_raw = gr.Textbox(
                                    value="Waiting for parsing.",
                                    label="Markdown Raw Text",
                                    max_lines=60,
                                    lines=45,
                                    show_copy_button=True,
                                    elem_id="raw_markdown_output",
                                    show_label=False,
                                )

                with gr.Row():
                    zip_download = gr.DownloadButton("Download Results ZIP", visible=True)
                    json_download = gr.DownloadButton("Download JSON", visible=True)

        file_input.upload(
            fn=lambda file_path, state: load_uploaded_file(
                file_path,
                state,
                output_dir=default_output_dir,
            ),
            inputs=[file_input, session_state],
            outputs=[file_preview, page_info, session_state, example_dropdown, example_preview],
        )

        example_dropdown.change(
            fn=preview_example,
            inputs=example_dropdown,
            outputs=example_preview,
        )

        choose_example_button.click(
            fn=lambda example_name, state: choose_example(
                example_name,
                state,
                output_dir=default_output_dir,
            ),
            inputs=[example_dropdown, session_state],
            outputs=[file_preview, page_info, session_state],
        )

        prev_btn.click(
            fn=lambda state: turn_page("prev", state),
            inputs=session_state,
            outputs=[file_preview, page_info, session_state],
            show_progress=False,
        )
        next_btn.click(
            fn=lambda state: turn_page("next", state),
            inputs=session_state,
            outputs=[file_preview, page_info, session_state],
            show_progress=False,
        )

        parse_button.click(
            fn=lambda file_path, state, keep_header_footer_value: parse_file(
                file_path,
                state,
                keep_header_footer_value,
                model_path=default_model_path,
                output_dir=default_output_dir,
                server_url=server_url,
                served_model_name=served_model_name,
                request_timeout=request_timeout,
                http_max_retries=http_max_retries,
                http_retry_backoff=http_retry_backoff,
                server_max_inflight=server_max_inflight,
                page_max_inflight=page_max_inflight,
                preprocess_batch_size=preprocess_batch_size,
                skip_preprocess=skip_preprocess,
                end2end=end2end,
            ),
            inputs=[file_input, session_state, keep_header_footer],
            outputs=[file_preview, md_view_ltr, md_view_rtl, md_raw, page_info, zip_download, json_download, session_state],
            show_progress=True,
            show_progress_on=[md_view_ltr, md_view_rtl, md_raw],
        )

        single_task_button.click(
            fn=lambda file_path, state, task_value: recognize_single_task(
                file_path,
                state,
                task_value,
                model_path=default_model_path,
                output_dir=default_output_dir,
                server_url=server_url,
                served_model_name=served_model_name,
                request_timeout=request_timeout,
                http_max_retries=http_max_retries,
                http_retry_backoff=http_retry_backoff,
                server_max_inflight=server_max_inflight,
                preprocess_batch_size=preprocess_batch_size,
            ),
            inputs=[file_input, session_state, task_dropdown],
            outputs=[file_preview, md_view_ltr, md_view_rtl, md_raw, page_info, zip_download, json_download, session_state],
            show_progress=True,
            show_progress_on=[md_view_ltr, md_view_rtl, md_raw],
        )

        clear_button.click(
            fn=clear_all,
            inputs=session_state,
            outputs=[
                file_input,
                file_preview,
                md_view_ltr,
                md_view_rtl,
                md_raw,
                page_info,
                zip_download,
                json_download,
                session_state,
                example_dropdown,
                example_preview,
            ],
            show_progress=False,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Start the MonkeyOCRv2 Gradio demo.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the MonkeyOCRv2 model weights used by local Async engine and preprocessor.")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, help="Directory where demo request outputs are saved.")
    parser.add_argument("--server-url", "-s", dest="server_url", default="", help="vLLM OpenAI-compatible server URL, for example http://127.0.0.1:8888. If omitted, local AsyncLLMEngine is used.")
    parser.add_argument("--served-model-name", default="MonkeyOCRv2", help="Model name exposed by vLLM serve.")
    parser.add_argument("--request-timeout", type=int, default=300, help="HTTP request timeout in seconds when using vLLM serve.")
    parser.add_argument("--http-max-retries", type=int, default=5, help="Maximum retries for transient vLLM server HTTP failures.")
    parser.add_argument("--http-retry-backoff", type=float, default=1.0, help="Base exponential backoff seconds for transient vLLM server HTTP failures.")
    parser.add_argument("--server-max-inflight", type=int, default=1024, help="Maximum in-flight model requests submitted by the demo process.")
    parser.add_argument("--page-max-inflight", type=int, default=256, help="Maximum pages kept in the parsing pipeline at the same time.")
    parser.add_argument("--preprocess-batch-size", type=int, default=32, help="Batch size used by the image preprocessor.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip image preprocessing before layout and recognition.")
    parser.add_argument("--end2end", action="store_true", help="Use end-to-end parsing prompt instead of layout followed by block recognition.")
    parser.add_argument("--demo-server-name", default="0.0.0.0", help="Host address for the Gradio demo server.")
    parser.add_argument("--demo-server-port", "-p", type=int, default=8891, help="Port for the Gradio demo server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    demo = create_gradio_app(
        args.model_path,
        args.output_dir,
        server_url=args.server_url,
        served_model_name=args.served_model_name,
        request_timeout=args.request_timeout,
        http_max_retries=args.http_max_retries,
        http_retry_backoff=args.http_retry_backoff,
        server_max_inflight=args.server_max_inflight,
        page_max_inflight=args.page_max_inflight,
        preprocess_batch_size=args.preprocess_batch_size,
        skip_preprocess=args.skip_preprocess,
        end2end=args.end2end,
    )
    demo.queue().launch(
        server_name=args.demo_server_name,
        server_port=args.demo_server_port,
        share=args.share,
        debug=True,
    )


if __name__ == "__main__":
    main()

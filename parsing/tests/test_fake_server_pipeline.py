"""Exercise the server client and layout/recognition handoff without a model."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

from core_runner import (
    ALL_PROMPT,
    MonkeyOCRv2_ServerParsing,
    _build_page_tasks,
    _format_block_fields,
    get_layout,
)


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    requests = []
    failures_remaining = 0
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802 - stdlib handler API
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        content = payload["messages"][0]["content"]
        question = next(item["text"] for item in content if item["type"] == "text")
        with self.lock:
            self.requests.append(payload)
            if type(self).failures_remaining:
                type(self).failures_remaining -= 1
                self.send_response(503)
                self.end_headers()
                return

        if question == ALL_PROMPT["LAYOUT"]:
            answer = '[{"bbox": [0, 0, 1000, 1000], "label": "Text"}]'
        else:
            answer = "recognized text"
        response = json.dumps({
            "choices": [{"message": {"content": answer}}],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        return


def test_fake_server_layout_and_recognition_flow():
    _FakeVLLMHandler.requests = []
    _FakeVLLMHandler.failures_remaining = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = MonkeyOCRv2_ServerParsing(
        f"http://127.0.0.1:{server.server_port}",
        model_name="fake-model",
        timeout=5,
        http_max_retries=0,
    )
    try:
        images = [Image.new("RGB", (40, 20), "white") for _ in range(2)]
        layouts = get_layout(model, images)
        assert layouts == [[{"bbox": [0, 0, 40, 20], "label": "Text"}]] * 2

        tasks = [
            task
            for page_idx, image in enumerate(images)
            for task in _build_page_tasks(page_idx, image, layouts[page_idx])
        ]
        outputs = model.batch_inference(
            [task["image"] for task in tasks],
            [task["question"] for task in tasks],
            concurrency=2,
        )
        assert outputs == ["recognized text", "recognized text"]
        assert _format_block_fields(
            tasks[0], outputs[0], "doc", [0], False, None
        )["content"] == "recognized text"

        assert len(_FakeVLLMHandler.requests) == 4
        layout_requests = [
            request for request in _FakeVLLMHandler.requests
            if request["messages"][0]["content"][1]["text"] == ALL_PROMPT["LAYOUT"]
        ]
        assert len(layout_requests) == 2
        assert all(request["max_tokens"] == 4096 for request in layout_requests)
        image_url = layout_requests[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        base64.b64decode(image_url.split(",", 1)[1])
    finally:
        model.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fake_server_retry_on_transient_http_error():
    _FakeVLLMHandler.requests = []
    _FakeVLLMHandler.failures_remaining = 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = MonkeyOCRv2_ServerParsing(
        f"http://127.0.0.1:{server.server_port}",
        timeout=5,
        http_max_retries=1,
        http_retry_backoff=0,
    )
    try:
        output = model.batch_inference(
            [Image.new("RGB", (8, 8), "white")],
            ["extract text"],
        )
        assert output == ["recognized text"]
        assert len(_FakeVLLMHandler.requests) == 2
    finally:
        model.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

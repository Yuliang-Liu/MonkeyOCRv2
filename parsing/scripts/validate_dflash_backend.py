#!/usr/bin/env python3
"""Validate one real-image baseline or DFlash vLLM request."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def http_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.loads(response.read().decode())


def build_command(args, serve_py: Path) -> list[str]:
    command = [
        sys.executable,
        str(serve_py),
        "--model-path",
        str(Path(args.model_path).expanduser()),
        "--served-model-name",
        "MonkeyOCRv2",
        "--port",
        str(args.port),
    ]
    command.extend(["--target-attention-backend", "FLASH_ATTN"])
    if args.mode == "dflash":
        command.extend(
            [
                "--draft-model",
                str(Path(args.draft_model).expanduser()),
                "--dflash-attention-backend",
                "FLASH_ATTN",
                "--validate-models",
            ]
        )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "dflash"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    if args.mode == "dflash" and not args.draft_model:
        parser.error("--draft-model is required in dflash mode")
    if args.mode == "baseline" and args.draft_model:
        parser.error("--draft-model is not allowed in baseline mode")

    model_path = Path(args.model_path).expanduser()
    draft_path = Path(args.draft_model).expanduser() if args.draft_model else None
    image_path = Path(args.image).expanduser()
    if not model_path.is_dir() or (draft_path is not None and not draft_path.is_dir()) or not image_path.is_file():
        print("SKIPPED: target, draft, or image path is missing", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"serve_{args.mode}_flashattn.log"
    serve_py = Path(__file__).resolve().parents[1] / "serve.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(serve_py.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    command = build_command(args, serve_py)
    started = time.time()
    process = None
    status = "FAIL"
    reason = ""
    try:
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.time() + args.timeout
            models = None
            while time.time() < deadline:
                if process.poll() is not None:
                    reason = f"server exited with code {process.returncode}"
                    break
                try:
                    models = http_json(f"http://127.0.0.1:{args.port}/v1/models", timeout=3)
                    break
                except (URLError, TimeoutError, OSError):
                    time.sleep(1)
            if models is None:
                reason = reason or "server did not become ready before timeout"
            else:
                mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
                encoded = base64.b64encode(image_path.read_bytes()).decode()
                payload = {
                    "model": "MonkeyOCRv2",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Recognize this document image."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ]}],
                    "max_tokens": 32,
                    "temperature": 0,
                }
                response = http_json(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    payload,
                    timeout=args.timeout,
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    status = "PASS"
                else:
                    reason = "chat completion returned no content"
            log.flush()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    has_forbidden_fallback = "fallback" in log_text.lower() and args.mode == "dflash"
    markers = {
        "target_backend": "FLASH_ATTN" in log_text,
        "draft_backend": args.mode == "baseline" or "FLASH_ATTN" in log_text,
        "method_dflash": args.mode == "baseline" or "method=dflash" in log_text or "method='dflash'" in log_text,
        "speculative_config": args.mode == "baseline" or "SpeculativeConfig" in log_text,
        "baseline_no_speculation": args.mode != "baseline" or "SpeculativeConfig" not in log_text,
        "no_fallback": not has_forbidden_fallback,
    }
    if status == "PASS" and not all(markers.values()):
        status = "FAIL"
        reason = "request succeeded but expected mode/backend markers were not all found"

    lines = [
        f"mode: {args.mode}",
        "target_backend: FLASH_ATTN",
        f"draft_backend: {'FLASH_ATTN' if args.mode == 'dflash' else 'none'}",
        f"status: {status}",
        f"elapsed_s: {time.time() - started:.2f}",
        f"method_dflash: {markers['method_dflash']}",
        f"speculative_config: {markers['speculative_config']}",
        f"baseline_no_speculation: {markers['baseline_no_speculation']}",
        f"no_backend_fallback: {markers['no_fallback']}",
    ]
    if reason:
        lines.append(f"reason: {reason}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

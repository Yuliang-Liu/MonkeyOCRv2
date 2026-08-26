"""Contract tests for the RAGFlow/MinerU-compatible endpoint.

These tests use a tiny fake pipeline, so they run without model weights or a
vLLM server and can be used in CI before opening a RAGFlow pull request.
"""
from pathlib import Path

import pytest
import asyncio
import io
import zipfile
from tempfile import SpooledTemporaryFile
from fastapi import UploadFile

import main


class FakePool:
    def run(self, config):
        source = Path(config.input_path).stem
        Path(config.output_path, f"{source}.md").write_text(
            f"# Parsed {source}\n\nMonkeyOCRv2", encoding="utf-8"
        )

    def close(self):
        pass


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "service_pool", FakePool())
    monkeypatch.setattr(main, "initialize_backend", lambda: main.backend.update(loaded=True))
    monkeypatch.setattr(main.settings, "output_dir", str(tmp_path))
    monkeypatch.setattr(main.settings, "server_url", "http://127.0.0.1:9")
    async def fake_save_upload(upload, output_dir):
        destination = Path(output_dir, upload.filename)
        destination.write_bytes(await upload.read())
        return destination
    monkeypatch.setattr(main, "save_upload", fake_save_upload)
    async def fake_parse_to_markdown(upload, **kwargs):
        return f"# Parsed {Path(upload.filename).stem}\n\nMonkeyOCRv2"
    monkeypatch.setattr(main, "run_document_pipeline", fake_parse_to_markdown)
    yield tmp_path


def upload(name):
    stream = SpooledTemporaryFile()
    stream.write(b"fake image")
    stream.seek(0)
    return UploadFile(filename=name, file=stream)


def test_file_parse_single_file(setup):
    response = asyncio.run(main.parse_document(file=upload("notice.png"), files=None, start_page_id=0, end_page_id=99999))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert "notice/notice.md" in archive.namelist()


def test_file_parse_multiple_images(setup):
    response = asyncio.run(main.parse_document(
        files=[upload("page1.png"), upload("page2.png")], file=None, start_page_id=0, end_page_id=99999
    ))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert "page1/page1.md" in archive.namelist()
        assert "page2/page2.md" in archive.namelist()

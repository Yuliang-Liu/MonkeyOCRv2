"""Contract tests for the batch ZIP parser endpoint.

These tests use a tiny fake pipeline, so they run without model weights or a
vLLM server and can be used in CI before opening a RAGFlow pull request.
"""
import asyncio
import io
import zipfile
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest
from fastapi import UploadFile

import main


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "output_dir", str(tmp_path))
    async def fake_run_document_pipeline(upload, **kwargs):
        return f"# Parsed {Path(upload.filename).stem}\n\nMonkeyOCRv2"
    monkeypatch.setattr(main, "run_document_pipeline", fake_run_document_pipeline)
    yield tmp_path


def upload(name):
    stream = SpooledTemporaryFile()
    stream.write(b"fake image")
    stream.seek(0)
    return UploadFile(filename=name, file=stream)


def test_parse_single_file_returns_zip(setup):
    response = asyncio.run(main.parse_document(file=upload("notice.png"), files=None, start_page_id=0, end_page_id=99999))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert "notice/notice.md" in archive.namelist()


def test_parse_multiple_files_returns_one_zip(setup):
    response = asyncio.run(main.parse_document(
        files=[upload("page1.png"), upload("page2.png")], file=None, start_page_id=0, end_page_id=99999
    ))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert "page1/page1.md" in archive.namelist()
        assert "page2/page2.md" in archive.namelist()

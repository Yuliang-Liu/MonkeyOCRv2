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
from fastapi import HTTPException, UploadFile

import main


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "output_dir", str(tmp_path))
    async def fake_run_document_pipeline(upload, **kwargs):
        return f"# Parsed {Path(upload.filename).stem}\n\nMonkeyOCRv2"
    monkeypatch.setattr(main, "run_document_pipeline", fake_run_document_pipeline)
    yield tmp_path


@pytest.fixture()
def json_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "output_dir", str(tmp_path))

    async def fake_run_document_pipeline(upload, **kwargs):
        markdown = f"# Parsed {Path(upload.filename).stem}\n\nMonkeyOCRv2"
        artifacts = {
            "jsons/notice.json": b'{"ok": true}',
            "all_results.json": b"{}",
        }
        return markdown, {}, artifacts

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


def test_parse_zip_exposes_stem_json_not_pipeline_json_paths(json_artifacts):
    response = asyncio.run(main.parse_document(file=upload("notice.png"), files=None, start_page_id=0, end_page_id=99999))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        names = archive.namelist()
        assert "notice/notice.md" in names
        assert "notice/notice.json" in names
        assert archive.read("notice/notice.json") == b'{"ok": true}'
        assert "all_results.json" not in names
        assert "notice/all_results.json" not in names
        assert "notice/notice_content_list.json" not in names
        assert "jsons/notice.json" not in names
        assert "notice/jsons/notice.json" not in names
        assert not any("/jsons/" in name or name.startswith("jsons/") for name in names)


def test_parse_requires_file_or_files(setup):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.parse_document(file=None, files=None, start_page_id=0, end_page_id=99999))
    assert exc_info.value.status_code == 422
    assert "file or files" in exc_info.value.detail


def test_parse_empty_files_list_requires_a_file(setup):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.parse_document(file=None, files=[], start_page_id=0, end_page_id=99999))
    assert exc_info.value.status_code == 422


def test_parse_rejects_equal_page_ids(setup):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.parse_document(file=upload("notice.png"), files=None, start_page_id=5, end_page_id=5))
    assert exc_info.value.status_code == 422
    assert "page range" in exc_info.value.detail


def test_parse_rejects_negative_start_page_id(setup):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.parse_document(file=upload("notice.png"), files=None, start_page_id=-1, end_page_id=99999))
    assert exc_info.value.status_code == 422


def test_parse_duplicate_filenames_keep_both_zip_entries(setup):
    response = asyncio.run(main.parse_document(
        files=[upload("notice.png"), upload("notice.png")], file=None, start_page_id=0, end_page_id=99999
    ))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        names = archive.namelist()
        assert "notice/notice.md" in names
        assert "notice_2/notice_2.md" in names


def test_health_check_reports_backend_and_status():
    payload = asyncio.run(main.health_check())
    assert payload["backend"] == "server"
    assert payload["status"] in {"healthy", "initializing"}

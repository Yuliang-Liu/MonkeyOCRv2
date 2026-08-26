"""Filename truncation, markdown/JSON write, and Picture relative paths."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core_runner import (
    MAX_FILENAME_BYTES,
    _format_block_fields,
    make_artifact_filename,
    result2md,
    save_picture_block,
)


def test_short_name_is_unchanged():
    assert make_artifact_filename("论文", ".md") == "论文.md"


def test_long_utf8_stem_fits_byte_limit_and_is_deterministic():
    stem = "测" * 200
    name = make_artifact_filename(stem, ".md")
    assert len(name.encode("utf-8")) <= MAX_FILENAME_BYTES
    assert name.endswith(".md")
    assert "_" in name
    assert make_artifact_filename(stem, ".md") == name
    assert name != stem + ".md"


def test_result2md_writes_long_and_special_names(tmp_path: Path):
    long_name = "测" * 200
    special = "报告 #1 (v2)"
    markdowns = result2md(
        [long_name, special],
        [
            [{"label": "Text", "content": "hello 你好"}],
            [{"label": "Text", "content": "world"}],
        ],
        save_dir=str(tmp_path),
    )
    assert markdowns[0].startswith("hello 你好")
    written = sorted(p.name for p in tmp_path.glob("*.md"))
    assert make_artifact_filename(long_name, ".md") in written
    assert "报告 #1 (v2).md" in written
    long_path = tmp_path / make_artifact_filename(long_name, ".md")
    assert len(long_path.name.encode("utf-8")) <= MAX_FILENAME_BYTES
    assert long_path.read_text(encoding="utf-8") == markdowns[0]


def test_result2md_keeps_image_relative_paths_and_drops_headers():
    results = [
        [
            {"label": "Page-header", "content": "secret header"},
            {"label": "Picture", "content": "![image](../images/doc_sub0.jpg)"},
            {"label": "Text", "content": "body"},
        ]
    ]
    markdowns = result2md(["doc"], results)
    assert markdowns[0] == "![image](../images/doc_sub0.jpg)\n\nbody\n"
    assert "secret header" not in markdowns[0]


def test_save_picture_block_returns_relative_images_path(tmp_path: Path):
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    ref = save_picture_block(image, tmp_path, "doc", 0)
    assert ref == "../images/doc_sub0.jpg"
    saved = tmp_path / "doc_sub0.jpg"
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_format_picture_block_uses_relative_markdown(tmp_path: Path):
    image = Image.new("RGB", (4, 4), color=(0, 255, 0))
    fields = _format_block_fields(
        {"label": "Picture", "need_infer": False, "image": image},
        "",
        "论文",
        [0],
        False,
        tmp_path,
    )
    assert fields["content"] == "![image](../images/论文_sub0.jpg)"
    assert (tmp_path / "论文_sub0.jpg").is_file()

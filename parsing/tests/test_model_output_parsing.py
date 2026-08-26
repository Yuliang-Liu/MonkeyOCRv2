"""Tolerant parsing of model layout / end-to-end text (no live model)."""

from __future__ import annotations

from core_runner import parse_end2end_output


def _labels(records):
    return [item["label"] for item in records]


def test_valid_json_list():
    text = (
        '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "hello"},'
        ' {"bbox": [200, 200, 400, 400], "label": "Title", "content": "Hi"}]'
    )
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert _labels(records) == ["Text", "Title"]
    assert records[0]["content"] == "hello"
    assert records[0]["bbox"] == [0, 0, 100, 100]
    assert layouts[1]["label"] == "Title"
    assert "content" not in layouts[1]


def test_valid_json_dict():
    text = '{"bbox": [0, 0, 100, 100], "label": "Title", "content": "Hi"}'
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["label"] == "Title"
    assert records[0]["content"] == "Hi"
    assert layouts[0]["bbox"] == [0, 0, 100, 100]


def test_python_literal_list():
    text = "[{'bbox': [0, 0, 50, 50], 'label': 'Text', 'content': 'py'}]"
    records, _layouts = parse_end2end_output(text, (100, 100))
    assert records[0]["content"] == "py"
    assert records[0]["bbox"] == [0, 0, 5, 5]


def test_truncated_json_keeps_complete_items():
    text = (
        '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "ok"},'
        ' {"bbox": [10, 10'
    )
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["content"] == "ok"
    assert layouts[0]["label"] == "Text"


def test_truncated_single_object_still_parses():
    text = '[{ "bbox": [1, 2, 3, 4], "label": "Text" '
    records, _layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["label"] == "Text"


def test_mixed_non_json_prose_around_list():
    text = (
        "Here is the layout:\n"
        '[{"bbox": [0, 0, 500, 80], "label": "Title", "content": "Hello"}]\n'
        "Thanks."
    )
    records, _layouts = parse_end2end_output(text, (200, 100))
    assert len(records) == 1
    assert records[0]["content"] == "Hello"
    assert records[0]["bbox"] == [0, 0, 100, 8]


def test_empty_and_non_structured_text_return_empty():
    assert parse_end2end_output("", (10, 10)) == ([], [])
    assert parse_end2end_output("   ", (10, 10)) == ([], [])
    assert parse_end2end_output("not json at all", (10, 10)) == ([], [])


def test_invalid_items_are_skipped():
    text = (
        '[{"bbox": [1, 2], "label": "Text"},'
        ' {"bbox": [1, 2, 3, 4]},'
        ' {"bbox": [0, 0, 10, 10], "label": "Title", "content": "keep"}]'
    )
    records, _layouts = parse_end2end_output(text, (100, 100))
    assert _labels(records) == ["Title"]
    assert records[0]["content"] == "keep"

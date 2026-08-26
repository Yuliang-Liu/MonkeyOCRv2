"""Regression tests for OTSL → HTML conversion.

Covers the #24 row-split-across-newlines fix, empty/malformed sequences, and
span tokens produced by ``parsing/train/html2otsl.py``.
"""

from __future__ import annotations

import pytest

from core_runner import _format_block_fields, otsl_to_html
from html_table import parse_table


def test_empty_otsl_returns_empty_table():
    assert otsl_to_html("") == "<table></table>"
    assert otsl_to_html("   ") == "<table></table>"
    assert otsl_to_html(None) == "<table></table>"


def test_basic_2x2_table():
    html = otsl_to_html("<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>")
    assert html == (
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>C</td><td>D</td></tr></table>"
    )
    parsed = parse_table(html)
    assert parsed.cell_texts == ["A", "B", "C", "D"]


def test_row_split_across_newlines_issue_24():
    """#24: cell text that contains a newline must stay in the same <td>.

    Before the DOTALL fix, ``re.findall(..., row_str)`` stopped at the first
    newline and the rest of the row was dropped.
    """
    html = otsl_to_html("<fcel>line1\nline2<fcel>B<nl>")
    assert html == "<table><tr><td>line1<br>line2</td><td>B</td></tr></table>"
    parse_table(html)


def test_crlf_in_cell_becomes_br():
    html = otsl_to_html("<fcel>line1\r\nline2<nl>")
    assert html == "<table><tr><td>line1<br>line2</td></tr></table>"


def test_colspan_from_lcel_tokens():
    html = otsl_to_html("<fcel>Head<lcel><lcel><nl><fcel>A<fcel>B<fcel>C<nl>")
    assert 'colspan="3"' in html
    assert html == (
        '<table><tr><td colspan="3">Head</td></tr>'
        "<tr><td>A</td><td>B</td><td>C</td></tr></table>"
    )


def test_rowspan_from_ucel_tokens():
    html = otsl_to_html("<fcel>Side<fcel>A<nl><ucel><fcel>B<nl>")
    assert 'rowspan="2"' in html
    assert html == (
        '<table><tr><td rowspan="2">Side</td><td>A</td></tr>'
        "<tr><td>B</td></tr></table>"
    )


def test_rowspan_and_colspan_with_xcel():
    html = otsl_to_html("<fcel>Big<lcel><fcel>A<nl><ucel><xcel><fcel>B<nl>")
    assert html == (
        '<table><tr><td rowspan="2" colspan="2">Big</td><td>A</td></tr>'
        "<tr><td>B</td></tr></table>"
    )


def test_empty_cell_token():
    html = otsl_to_html("<fcel>A<ecel><fcel>B<nl>")
    assert html == "<table><tr><td>A</td><td></td><td>B</td></tr></table>"


@pytest.mark.parametrize(
    "otsl",
    [
        "<nl>",
        "<fcel>A",
        "<zzzz>nope<fcel>ok<nl>",
        "notags just text",
        "<fcel>A<lcel><ucel><xcel><nl>",
        "<fcel>",
        "<<<>>>",
        "<fcel>A<nl><nl><fcel>B<nl>",
    ],
)
def test_malformed_otsl_does_not_raise_and_returns_table(otsl):
    html = otsl_to_html(otsl)
    parse_table(html)


def test_html_special_characters_are_escaped():
    html = otsl_to_html("<fcel>A&B<\"'><nl>")
    assert html == (
        "<table><tr><td>A&amp;B&lt;&quot;&#x27;&gt;</td></tr></table>"
    )


def test_format_block_fields_table_keeps_otsl_and_html(monkeypatch):
    monkeypatch.delenv("MOCR2_TABLE_HTML", raising=False)
    fields = _format_block_fields(
        {"label": "Table", "need_infer": True, "image": None},
        "<fcel>A<fcel>B<nl>",
        "doc",
        None,
        True,
        None,
    )
    assert fields["otsl"] == "<fcel>A<fcel>B<nl>"
    assert fields["content"] == "<table><tr><td>A</td><td>B</td></tr></table>"

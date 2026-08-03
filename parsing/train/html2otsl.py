#!/usr/bin/env python3
"""Convert assistant HTML responses in an ms-swift JSONL file to OTSL.

Each input line must contain a ``messages`` list. The HTML in every assistant
message's ``content`` field is replaced with its OTSL representation. All
other messages and fields, including ``images``, are preserved.

Usage::

    python html2otsl.py -i html.jsonl -o otsl.jsonl
    python html2otsl.py --test
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


_WS_BASE = 0xE000
_CELL_RE = re.compile(
    r"(<(?:td|th)\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>)(.*?)(</(?:td|th)\s*>)",
    re.I | re.S,
)


def _protect_cell_whitespace(source: str) -> str:
    """Protect repeated whitespace that BeautifulSoup's HTML parser collapses."""
    def repl(match: re.Match[str]) -> str:
        inner = match.group(2)
        # Only alter text between tags; whitespace in attributes is markup.
        pieces = re.split(r"(<[^>]*>)", inner)
        for i in range(0, len(pieces), 2):
            pieces[i] = "".join(chr(_WS_BASE + ord(ch)) if ch.isspace() else ch for ch in pieces[i])
        return match.group(1) + "".join(pieces) + match.group(3)
    return _CELL_RE.sub(repl, source)


@dataclass
class Span:
    """A cell occupying a future row."""

    rows_left: int
    primary: bool


def _positive_int(value: object, default: int = 1) -> int:
    try:
        number = int(str(value))
        return number if number > 0 else default
    except (TypeError, ValueError):
        return default


def _cell_text(cell) -> str:
    # Preserve cell text exactly. In particular, whitespace-only cells are
    # meaningful text cells and must not collapse into <ecel>.
    # Markup is intentionally discarded; only its text nodes remain.
    text = cell.get_text("", strip=False)
    return "".join(chr(ord(ch) - _WS_BASE) if _WS_BASE <= ord(ch) < _WS_BASE + 0x1000 else ch for ch in text)


def html_to_otsl(source: str) -> str:
    """Convert the first HTML ``table`` in *source* to an OTSL string.

    Empty input or input without a table returns ``""``.  ``th`` is treated
    the same as ``td``.  ``rowspan`` and ``colspan`` continuations use OTSL's
    ``ucel``, ``xcel`` and ``lcel`` markers.
    """
    if not source:
        return ""
    soup = BeautifulSoup(_protect_cell_whitespace(source), "html.parser")
    table = soup.find("table")
    if table is None:
        return ""

    # column -> pending vertical span.  A primary span emits <ucel>; a column
    # covered by the colspan part emits <xcel>.
    pending: Dict[int, Span] = {}
    output: List[str] = []

    # Ignore rows belonging to a nested table; their markup is treated as
    # content of the containing cell, rather than as extra outer rows.
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        row_tokens: List[str] = []
        col = 0
        cell_index = 0

        while cell_index < len(cells) or any(s.rows_left > 0 for s in pending.values()):
            span = pending.get(col)
            if span is not None and span.rows_left > 0:
                row_tokens.append("<ucel>" if span.primary else "<xcel>")
                span.rows_left -= 1
                if span.rows_left == 0:
                    del pending[col]
                col += 1
                continue

            if cell_index >= len(cells):
                # There can be a span in a later column while this column is
                # empty.  Advance without inventing an OTSL empty cell.
                later = [c for c, s in pending.items() if c > col and s.rows_left > 0]
                if later:
                    col += 1
                    continue
                break

            cell = cells[cell_index]
            text = _cell_text(cell)
            row_tokens.append(f"<fcel>{text}" if text else "<ecel>")
            colspan = _positive_int(cell.get("colspan"))
            rowspan = _positive_int(cell.get("rowspan"))
            if rowspan > 1:
                pending[col] = Span(rowspan - 1, True)
            col += 1
            for _ in range(1, colspan):
                row_tokens.append("<lcel>")
                if rowspan > 1:
                    pending[col] = Span(rowspan - 1, False)
                col += 1
            cell_index += 1

        row_tokens.append("<nl>")
        output.extend(row_tokens)

    return "".join(output)


# Spelling used by the older scripts in this directory.
html_to_ostl = html_to_otsl


# (name, HTML, expected OTSL).  These cover ordinary tables, whitespace and
# entities, headers, all span combinations, malformed attributes and input
# without a table.
TEST_CASES = [
    ("basic 2x2", "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>", "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>"),
    ("headers and nested markup", "<table><thead><tr><th><b>Name</b></th><th>Value</th></tr></thead><tbody><tr><td>x</td><td><em>1</em></td></tr></tbody></table>", "<fcel>Name<fcel>Value<nl><fcel>x<fcel>1<nl>"),
    ("empty and whitespace cells", "<table><tr><td></td><td>   </td><th>\n</th></tr></table>", "<ecel><fcel>   <fcel>\n<nl>"),
    ("exact whitespace", "<table><tr><td>  A  </td><td>\t\t</td><td>\n\n</td></tr></table>", "<fcel>  A  <fcel>\t\t<fcel>\n\n<nl>"),
    ("whitespace and entities", "<table><tr><td>  A\n B </td><td>&amp; &lt;x&gt;</td></tr></table>", "<fcel>  A\n B <fcel>& <x><nl>"),
    ("colspan", "<table><tr><td colspan='3'>Head</td></tr><tr><td>A</td><td>B</td><td>C</td></tr></table>", "<fcel>Head<lcel><lcel><nl><fcel>A<fcel>B<fcel>C<nl>"),
    ("rowspan", "<table><tr><td rowspan='2'>Side</td><td>A</td></tr><tr><td>B</td></tr></table>", "<fcel>Side<fcel>A<nl><ucel><fcel>B<nl>"),
    ("rowspan + colspan", "<table><tr><td rowspan='2' colspan='2'>Big</td><td>A</td></tr><tr><td>B</td></tr></table>", "<fcel>Big<lcel><fcel>A<nl><ucel><xcel><fcel>B<nl>"),
    ("two independent rowspans", "<table><tr><td rowspan='3'>A</td><td rowspan='2'>B</td><td>C</td></tr><tr><td>D</td></tr><tr><td>E</td></tr></table>", "<fcel>A<fcel>B<fcel>C<nl><ucel><ucel><fcel>D<nl><ucel><fcel>E<nl>"),
    ("invalid span values", "<table><tr><td colspan='0' rowspan='nope'>X</td><td>Y</td></tr></table>", "<fcel>X<fcel>Y<nl>"),
    ("no table", "<div><p>not a table</p></div>", ""),
    ("multiple tables uses first", "<table><tr><td>first</td></tr></table><table><tr><td>second</td></tr></table>", "<fcel>first<nl>"),
]


def run_tests() -> int:
    failures = 0
    print(f"Running {len(TEST_CASES)} HTML → OTSL test cases\n")
    for number, (name, source, expected) in enumerate(TEST_CASES, 1):
        actual = html_to_otsl(source)
        status = "PASS" if actual == expected else "FAIL"
        print(f"[{status}] Case {number}: {name}")
        print("HTML:")
        print(source)
        print("OTSL:")
        print(actual if actual else "(empty)")
        if actual != expected:
            failures += 1
            print("EXPECTED:")
            print(expected if expected else "(empty)")
        print("-" * 72)
    print(f"Summary: {len(TEST_CASES) - failures}/{len(TEST_CASES)} passed")
    return 1 if failures else 0


def convert_jsonl(input_path: Path, output_path: Path) -> int:
    """Convert assistant ``content`` from HTML to OTSL in every record.

    Blank lines are ignored. Invalid JSON, non-object records, and missing or
    malformed ``messages`` fields raise an error that includes the input line
    number. The return value is the number of converted assistant messages.
    """
    converted = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: each line must be a JSON object"
                )
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(
                    f'{input_path}:{line_number}: field "messages" must be a list'
                )
            assistant_count = 0
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{input_path}:{line_number}: messages[{message_index}] "
                        "must be a JSON object"
                    )
                if message.get("role") != "assistant":
                    continue
                assistant_count += 1
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError(
                        f"{input_path}:{line_number}: "
                        f'messages[{message_index}].content must be a string'
                    )
                message["content"] = html_to_otsl(content)
                converted += 1
            if assistant_count == 0:
                raise ValueError(
                    f"{input_path}:{line_number}: no assistant message found"
                )

            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    return converted


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert assistant message content from HTML to OTSL in JSONL"
    )
    parser.add_argument("-i", "--input", type=Path, help="input HTML JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="output OTSL JSONL file")
    parser.add_argument("-test", "--test", action="store_true", help="print and run all built-in test cases")
    args = parser.parse_args(argv)
    if args.test:
        return run_tests()

    if args.input is None or args.output is None:
        parser.error("-i/--input and -o/--output are required unless --test is used")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    try:
        converted = convert_jsonl(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Converted {converted} assistant messages: {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

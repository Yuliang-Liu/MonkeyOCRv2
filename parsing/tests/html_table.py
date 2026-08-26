"""Small HTML-table checks for OTSL conversion tests."""

from __future__ import annotations

from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.cell_texts: list[str] = []
        self._in_td = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)
        if tag == "td":
            self._in_td = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_td:
            self.cell_texts.append("".join(self._buf))
            self._in_td = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._buf.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._in_td:
            self._buf.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._in_td:
            self._buf.append(f"&#{name};")


def parse_table(html: str) -> _TableParser:
    assert html.startswith("<table>"), html
    assert html.endswith("</table>"), html
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    assert parser.tags[0] == "table"
    return parser

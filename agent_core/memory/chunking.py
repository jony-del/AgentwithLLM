from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

CHUNKER_VERSION = "markdown-v1"
_TOKEN_SPAN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    chunk_id: str
    ordinal: int
    heading: str
    content: str
    token_count: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class _Unit:
    heading: str
    content: str
    start: int
    end: int


def token_count(text: str) -> int:
    return sum(1 for _ in _TOKEN_SPAN_RE.finditer(text))


def _slice_tokens(text: str, start: int, end: int) -> tuple[str, int, int]:
    spans = list(_TOKEN_SPAN_RE.finditer(text))
    if not spans or start >= len(spans):
        return "", len(text), len(text)
    left = spans[max(0, start)].start()
    right = spans[min(len(spans), end) - 1].end()
    return text[left:right], left, right


def _markdown_units(text: str) -> list[_Unit]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    headings: list[tuple[int, str]] = []
    units: list[_Unit] = []
    paragraph_start: int | None = None
    paragraph_lines: list[str] = []

    def current_heading() -> str:
        return " / ".join(value for _, value in headings)

    def flush_paragraph(end: int) -> None:
        nonlocal paragraph_start, paragraph_lines
        content = "".join(paragraph_lines).strip()
        if content and paragraph_start is not None:
            units.append(_Unit(current_heading(), content, paragraph_start, end))
        paragraph_start = None
        paragraph_lines = []

    index = 0
    while index < len(lines):
        line = lines[index]
        start = offsets[index]
        heading = _HEADING_RE.match(line.rstrip("\r\n"))
        if heading:
            flush_paragraph(start)
            level = len(heading.group(1))
            value = heading.group(2).strip()
            headings[:] = [(old_level, old) for old_level, old in headings if old_level < level]
            headings.append((level, value))
            index += 1
            continue
        fence = _FENCE_RE.match(line)
        if fence:
            flush_paragraph(start)
            marker = fence.group(1)
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                if lines[index].lstrip().startswith(marker):
                    index += 1
                    break
                index += 1
            end = offsets[index] if index < len(offsets) else len(text)
            units.append(_Unit(current_heading(), "".join(code_lines).strip(), start, end))
            continue
        if not line.strip():
            flush_paragraph(start)
        else:
            if paragraph_start is None:
                paragraph_start = start
            paragraph_lines.append(line)
        index += 1
    flush_paragraph(len(text))
    return units


def _split_unit(unit: _Unit, maximum: int) -> list[_Unit]:
    count = token_count(unit.content)
    if count <= maximum:
        return [unit]
    pieces: list[_Unit] = []
    for start in range(0, count, maximum):
        content, left, right = _slice_tokens(unit.content, start, min(count, start + maximum))
        if content:
            pieces.append(
                _Unit(
                    unit.heading,
                    content,
                    unit.start + left,
                    min(unit.end, unit.start + right),
                )
            )
    return pieces


def chunk_markdown(
    document_id: str,
    text: str,
    *,
    chunk_tokens: int = 384,
    overlap_tokens: int = 64,
) -> list[MarkdownChunk]:
    """Split Markdown by semantic blocks, then pack bounded overlapping chunks."""
    maximum = max(32, int(chunk_tokens))
    overlap = max(0, min(int(overlap_tokens), maximum - 1))
    unit_maximum = maximum - overlap if overlap else maximum
    units = [
        piece
        for unit in _markdown_units(text)
        for piece in _split_unit(unit, unit_maximum)
    ]
    if not units and text.strip():
        units = [_Unit("", text.strip(), 0, len(text))]
    chunks: list[MarkdownChunk] = []
    buffer: list[_Unit] = []
    buffer_tokens = 0

    def emit() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        rendered = "\n\n".join(unit.content for unit in buffer if unit.content).strip()
        if not rendered:
            buffer = []
            buffer_tokens = 0
            return
        ordinal = len(chunks)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
        headings = list(dict.fromkeys(unit.heading for unit in buffer if unit.heading))
        chunks.append(
            MarkdownChunk(
                chunk_id=f"{document_id}:{ordinal:06d}:{digest}",
                ordinal=ordinal,
                heading=" | ".join(headings),
                content=rendered,
                token_count=token_count(rendered),
                start_char=min(unit.start for unit in buffer),
                end_char=max(unit.end for unit in buffer),
            )
        )
        if overlap <= 0:
            buffer = []
            buffer_tokens = 0
            return
        tail, left, _ = _slice_tokens(rendered, max(0, token_count(rendered) - overlap), token_count(rendered))
        last = buffer[-1]
        buffer = [
            _Unit(
                last.heading,
                tail,
                max(min(unit.start for unit in buffer), max(unit.end for unit in buffer) - len(tail)),
                max(unit.end for unit in buffer),
            )
        ] if tail else []
        buffer_tokens = token_count(tail)

    for unit in units:
        count = token_count(unit.content)
        if buffer and buffer_tokens + count > maximum:
            emit()
        if buffer and buffer_tokens + count > maximum:
            # A full overlap tail plus a maximum-sized unit cannot coexist. Keep
            # exactly the newest prefix that still honors the hard chunk bound.
            allowed = max(0, maximum - count)
            if allowed == 0:
                buffer = []
                buffer_tokens = 0
            else:
                tail = buffer[-1]
                sliced, left, _ = _slice_tokens(
                    tail.content,
                    max(0, token_count(tail.content) - allowed),
                    token_count(tail.content),
                )
                buffer = [_Unit(tail.heading, sliced, tail.start + left, tail.end)]
                buffer_tokens = token_count(sliced)
        buffer.append(unit)
        buffer_tokens += count
    emit()
    return chunks

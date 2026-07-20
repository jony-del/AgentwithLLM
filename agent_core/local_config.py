"""Small atomic updater for gitignored ``agent.local.toml`` settings."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class LocalConfigError(OSError):
    pass


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    raise LocalConfigError(f"unsupported local setting value: {type(value).__name__}")


def update_local_table(
    workspace: str | Path,
    table: str,
    values: dict[str, Any],
) -> Path:
    """Atomically update simple keys in one TOML table without touching other tables."""

    root = Path(workspace).resolve()
    return update_toml_table(root / "agent.local.toml", table, values)


def update_toml_table(path: str | Path, table: str, values: dict[str, Any]) -> Path:
    """Atomically update a table in an arbitrary TOML file."""

    path = Path(path).expanduser().resolve()
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        import tomllib

        if original.strip():
            tomllib.loads(original)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LocalConfigError(f"cannot update malformed {path.name}: {exc}") from exc

    header = re.compile(rf"(?m)^\[{re.escape(table)}\][ \t]*(?:#.*)?$")
    found = header.search(original)
    assignments = "".join(f"{key} = {_toml_value(value)}\n" for key, value in values.items())
    if found is None:
        prefix = original.rstrip()
        updated = (prefix + "\n\n" if prefix else "") + f"[{table}]\n" + assignments
    else:
        tail = original[found.end() :]
        next_header = re.search(r"(?m)^\[[^\]\n]+\][ \t]*(?:#.*)?$", tail)
        section_end = found.end() + (next_header.start() if next_header else len(tail))
        body = original[found.end() : section_end]
        for key, value in values.items():
            rendered = f"{key} = {_toml_value(value)}"
            assignment = re.compile(
                rf"(?m)^[ \t]*{re.escape(key)}[ \t]*="
            )
            match = assignment.search(body)
            if match is not None:
                value_end = _toml_value_end(body, match.end())
                body = body[: match.start()] + rendered + body[value_end:]
            else:
                body = body.rstrip() + "\n" + rendered + "\n"
        updated = original[: found.end()] + body + original[section_end:]

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _toml_value_end(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index] in " \t":
        index += 1
    if index >= len(text) or text[index] not in "[{":
        newline = text.find("\n", index)
        return len(text) if newline < 0 else newline
    opening = text[index]
    closing = "]" if opening == "[" else "}"
    depth = 0
    quote: str | None = None
    escaped = False
    for pos in range(index, len(text)):
        char = text[pos]
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos + 1
    raise LocalConfigError("unterminated TOML array/table value")

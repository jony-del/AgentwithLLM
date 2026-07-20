"""Safe, bounded Jupyter notebook formatting and atomic cell edits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from typing import Any


def notebook_fingerprint(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    stat = path.stat()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
        "size": len(raw),
    }


def load_notebook(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"notebook exceeds {max_bytes} byte limit")
    try:
        notebook = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid notebook JSON: {exc}") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("invalid notebook: top-level cells array is required")
    if any(not isinstance(cell, dict) for cell in notebook["cells"]):
        raise ValueError("invalid notebook: every cell must be an object")
    return notebook, raw


def stable_cell_id(cell: dict[str, Any], index: int) -> str:
    existing = str(cell.get("id", "")).strip()
    if existing:
        return existing
    source = cell.get("source", "")
    joined = "".join(str(item) for item in source) if isinstance(source, list) else str(source)
    digest = hashlib.sha256(f"{index}\0{cell.get('cell_type')}\0{joined}".encode()).hexdigest()[:12]
    return f"legacy-{digest}"


def format_notebook(path: Path, *, max_bytes: int = 16 * 1024 * 1024, max_output_chars: int = 8000) -> tuple[str, dict[str, object]]:
    notebook, raw = load_notebook(path, max_bytes=max_bytes)
    blocks: list[str] = [f"Notebook: {path.name}", f"nbformat: {notebook.get('nbformat', '?')}.{notebook.get('nbformat_minor', '?')}"]
    output_budget = max(0, max_output_chars)
    for index, raw_cell in enumerate(notebook["cells"]):
        if not isinstance(raw_cell, dict):
            continue
        cell_id = stable_cell_id(raw_cell, index)
        cell_type = str(raw_cell.get("cell_type", "unknown"))
        source = raw_cell.get("source", "")
        source_text = "".join(str(item) for item in source) if isinstance(source, list) else str(source)
        blocks.append(f"\n--- cell {cell_id} [{cell_type}] execution_count={raw_cell.get('execution_count')} ---")
        blocks.append(source_text)
        outputs = raw_cell.get("outputs", [])
        if not isinstance(outputs, list) or not outputs:
            continue
        rendered: list[str] = []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            kind = str(output.get("output_type", "output"))
            if kind == "stream":
                value = output.get("text", "")
                text = "".join(str(item) for item in value) if isinstance(value, list) else str(value)
                rendered.append(f"[{kind}] {text}")
            elif kind == "error":
                rendered.append(f"[error] {output.get('ename', '')}: {output.get('evalue', '')}")
            else:
                data = output.get("data", {})
                if not isinstance(data, dict):
                    continue
                textual = data.get("text/plain", data.get("text/markdown"))
                if textual is not None:
                    text = "".join(str(item) for item in textual) if isinstance(textual, list) else str(textual)
                    rendered.append(f"[{kind}] {text}")
                image_types = sorted(key for key in data if key.startswith("image/"))
                if image_types:
                    rendered.append(f"[{kind}] <binary image omitted: {', '.join(image_types)}>")
        output_text = "\n".join(rendered)
        if len(output_text) > output_budget:
            output_text = output_text[:output_budget] + "\n[notebook outputs truncated]"
            output_budget = 0
        else:
            output_budget -= len(output_text)
        if output_text:
            blocks.append(output_text)
    stat = path.stat()
    metadata: dict[str, object] = {
        "notebook": True,
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
        "size": len(raw),
        "cell_count": len(notebook["cells"]),
    }
    return "\n".join(blocks), metadata


_CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def edit_notebook(
    path: Path,
    *,
    expected: dict[str, object],
    cell_id: str | None,
    new_source: str,
    cell_type: str | None,
    edit_mode: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> dict[str, object]:
    current = notebook_fingerprint(path)
    if any(current.get(key) != expected.get(key) for key in ("sha256", "mtime_ns", "size")):
        raise RuntimeError("notebook changed since it was read; read it again before editing")
    notebook, _ = load_notebook(path, max_bytes=max_bytes)
    cells = notebook["cells"]
    ids = [stable_cell_id(cell, index) for index, cell in enumerate(cells) if isinstance(cell, dict)]
    if edit_mode not in {"replace", "insert", "delete"}:
        raise ValueError("edit_mode must be replace, insert, or delete")
    if edit_mode in {"replace", "delete"}:
        if not cell_id or cell_id not in ids:
            raise ValueError(f"unknown cell_id: {cell_id!r}")
        index = ids.index(cell_id)
        if edit_mode == "delete":
            del cells[index]
        else:
            cell = cells[index]
            assert isinstance(cell, dict)
            cell.setdefault("id", cell_id)
            cell["source"] = new_source.splitlines(keepends=True)
            if cell_type is not None:
                if cell_type not in {"code", "markdown", "raw"}:
                    raise ValueError("cell_type must be code, markdown, or raw")
                cell["cell_type"] = cell_type
                if cell_type == "code":
                    cell.setdefault("execution_count", None)
                    cell.setdefault("outputs", [])
                else:
                    cell.pop("execution_count", None)
                    cell.pop("outputs", None)
    else:
        if cell_id is not None and cell_id not in ids:
            raise ValueError(f"unknown insertion anchor cell_id: {cell_id!r}")
        kind = cell_type or "code"
        if kind not in {"code", "markdown", "raw"}:
            raise ValueError("cell_type must be code, markdown, or raw")
        generated = uuid.uuid4().hex[:12]
        if not _CELL_ID.fullmatch(generated):
            raise RuntimeError("failed to generate a valid notebook cell id")
        inserted_cell: dict[str, Any] = {
            "id": generated,
            "cell_type": kind,
            "metadata": {},
            "source": new_source.splitlines(keepends=True),
        }
        if kind == "code":
            inserted_cell.update({"execution_count": None, "outputs": []})
        insertion = len(cells) if cell_id is None else ids.index(cell_id) + 1
        cells.insert(insertion, inserted_cell)
        cell_id = generated
    encoded = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"edited notebook exceeds {max_bytes} byte limit")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    result = notebook_fingerprint(path)
    result.update({"cell_id": cell_id, "edit_mode": edit_mode})
    return result

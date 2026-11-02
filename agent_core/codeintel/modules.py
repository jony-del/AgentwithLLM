"""Static package boundaries; no build configuration is executed."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import sqlite3
import tomllib

from agent_core.codeintel.backends import module

MANIFESTS = {"pyproject.toml", "package.json", "Cargo.toml", "go.mod"}


def package_name(path: str, text: str) -> str | None:
    name = PurePosixPath(path).name
    try:
        value = None
        if name == "package.json":
            value = json.loads(text).get("name")
        elif name == "pyproject.toml":
            data = tomllib.loads(text)
            value = data.get("project", {}).get("name") or data.get("tool", {}).get("poetry", {}).get("name")
        elif name == "Cargo.toml":
            value = tomllib.loads(text).get("package", {}).get("name")
        elif name == "go.mod":
            for line in text.splitlines():
                parts = line.split("//", 1)[0].split()
                if len(parts) == 2 and parts[0] == "module":
                    value = parts[1].strip('"')
                    break
        return value if isinstance(value, str) and value.strip() else None
    except (ValueError, TypeError, AttributeError, RecursionError):
        return None


def resolve_module(db: sqlite3.Connection, path: str) -> str:
    for parent in PurePosixPath(path).parents:
        root = str(parent)
        if db.execute("SELECT 1 FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone():
            return root
    return module(path)


def update_boundary(db: sqlite3.Connection, path: str, text: str, digest: str) -> None:
    if PurePosixPath(path).name not in MANIFESTS:
        return
    root = str(PurePosixPath(path).parent)
    before = bool(db.execute("SELECT 1 FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone())
    name = package_name(path, text)
    if name:
        db.execute("INSERT OR REPLACE INTO module_roots VALUES(?,?,?,?)", (path, root, name, digest))
    else:
        db.execute("DELETE FROM module_roots WHERE manifest=?", (path,))
    after = bool(db.execute("SELECT 1 FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone())
    if before != after:
        # Publish invalidation with the boundary. Existing maintenance budgets
        # rebuild the affected subtree; readers cannot use obsolete module facts.
        where, args = ("1=1", ()) if root == "." else ("path>=? AND path<?", (root + "/", root + "0"))
        db.execute(f"INSERT OR IGNORE INTO pending SELECT path,'module_changed' FROM files WHERE {where}", args)
        db.execute(f"UPDATE files SET state='pending' WHERE {where}", args)

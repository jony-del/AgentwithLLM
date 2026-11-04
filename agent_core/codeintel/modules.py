"""Static package boundaries; no build configuration is executed."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import sqlite3
import tomllib

from agent_core.codeintel.backends import manifest_links, module

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


def _owner(db: sqlite3.Connection, manifest_path: str) -> tuple[str, str]:
    root = resolve_module(db, manifest_path)
    row = db.execute("SELECT name FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone()
    return root, (row[0] if row else root)


def _resolve_link(db: sqlite3.Connection, manifest_path: str, ref: dict) -> tuple[str | None, str, str]:
    """Resolve one structured ref against the bounded indexed inventory.
    Returns (to_root, resolution, reason); to_root None when not local-resolved."""
    if ref.get("member_glob"):
        return None, "workspace_member", ""
    if ref.get("path"):
        raw = str(ref["path"]).replace("\\", "/")
        base = PurePosixPath(manifest_path).parent
        candidate = PurePosixPath(raw) if PurePosixPath(raw).is_absolute() else base.joinpath(raw)
        parts: list[str] = []
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    return None, "unresolved", "path_escapes_workspace"
                parts.pop()
                continue
            parts.append(part)
        if PurePosixPath(raw).is_absolute() or "\\" in raw:
            return None, "unresolved", "path_escapes_workspace"
        target = "/".join(parts) or "."
        rows = db.execute("SELECT root FROM module_roots").fetchall()
        owners = [r[0] for r in rows if target == r[0] or target.startswith(r[0].rstrip("/") + "/") or r[0] == "." and target == "."]
        if not owners:
            return None, "unresolved", "no_indexed_package_at_path"
        return max(owners, key=len), "local_path", ""
    rows = db.execute("SELECT root FROM module_roots WHERE name=?", (ref["name"],)).fetchall()
    if len(rows) == 1:
        return rows[0][0], "local_name", ""
    if len(rows) > 1:
        return None, "unresolved", f"ambiguous_local_name:{len(rows)}"
    return None, "external", ""


def _match_members(db: sqlite3.Connection, manifest_path: str, glob: str) -> list[tuple[str, str]]:
    prefix = str(PurePosixPath(manifest_path).parent)
    pattern = (prefix + "/" + glob + "/" + PurePosixPath(manifest_path).name)
    pattern = pattern.removeprefix("./")
    return [(m, r) for m, r in db.execute("SELECT manifest, root FROM module_roots")
            if PurePosixPath(m).match(pattern)][:64]


def update_package_links(db: sqlite3.Connection, path: str, text: str) -> None:
    """Rebuild the local package graph edges of one manifest from indexed inventory.

    Runs inside the manifest's per-file publish transaction; resolution consults only
    the indexed module_roots, never the live filesystem. External registry names are
    not local edges and are not recorded here (they remain plain manifest edges).
    """
    if PurePosixPath(path).name not in MANIFESTS:
        return
    db.execute("DELETE FROM package_links WHERE manifest=?", (path,))
    try:
        refs = manifest_links(path, text)
    except (ValueError, TypeError, AttributeError, RecursionError):
        return  # A malformed manifest stays searchable text; links simply absent.
    from_root, from_name = _owner(db, path)
    for ref in refs:
        if ref.get("member_glob"):
            # A persistent scope marker keeps the declared glob re-matchable as new
            # packages appear; queries never return markers (filtered by resolution).
            db.execute("INSERT INTO package_links VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (path, from_root, from_name, None, ref["name"], "workspace_member",
                        None, None, ref["line"], "workspace_member_scope", ""))
            for manifest, member_root in _match_members(db, path, ref["member_glob"]):
                db.execute("INSERT INTO package_links VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (path, from_root, from_name, member_root, ref["name"], "workspace_member",
                            None, None, ref["line"], "workspace_member", ""))
            continue
        to_root, resolution, reason = _resolve_link(db, path, ref)
        if resolution == "external":
            continue  # Registry dependencies are edges, never local source links.
        db.execute("INSERT INTO package_links VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (path, from_root, from_name, to_root, ref["name"], ref["kind"],
                    ref.get("version"), ref.get("path"), ref["line"], resolution, reason))


def reresolve_links(db: sqlite3.Connection, new_root: str, new_name: str) -> None:
    """A newly indexed manifest may resolve previously unresolved links and matches
    workspace-member globs declared earlier."""
    markers = db.execute(
        "SELECT DISTINCT manifest, to_name, line FROM package_links WHERE resolution='workspace_member_scope'").fetchall()
    for manifest, glob, line in markers[:128]:
        from_root, from_name = _owner(db, manifest)
        for member_manifest, to_root in _match_members(db, manifest, glob):
            exists = db.execute(
                "SELECT 1 FROM package_links WHERE manifest=? AND to_root=? AND to_name=? "
                "AND resolution='workspace_member'", (manifest, to_root, glob)).fetchone()
            if not exists:
                db.execute("INSERT INTO package_links VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (manifest, from_root, from_name, to_root, glob, "workspace_member",
                            None, None, line, "workspace_member", ""))
    rows = db.execute("SELECT rowid, manifest, to_name, path FROM package_links WHERE resolution='unresolved'").fetchall()
    for rowid, manifest, to_name, path in rows[:512]:
        if path:  # Path refs keep path semantics; a same-named root never rewrites them.
            resolved_root, resolution, reason = _resolve_link(db, manifest, {"name": to_name, "path": path})
            if resolved_root is not None:
                db.execute("UPDATE package_links SET to_root=?, resolution=?, reason=? WHERE rowid=?",
                           (resolved_root, resolution, reason, rowid))
            continue
        if to_name == new_name:
            matches = db.execute("SELECT root FROM module_roots WHERE name=?", (to_name,)).fetchall()
            if len(matches) == 1:
                db.execute("UPDATE package_links SET to_root=?, resolution='local_name', reason='' WHERE rowid=?",
                           (matches[0][0], rowid))


def update_boundary(db: sqlite3.Connection, path: str, text: str, digest: str) -> None:
    if PurePosixPath(path).name not in MANIFESTS:
        return
    root = str(PurePosixPath(path).parent)
    before = bool(db.execute("SELECT 1 FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone())
    name = package_name(path, text)
    old = db.execute("SELECT name FROM module_roots WHERE manifest=?", (path,)).fetchone()
    if name:
        db.execute("INSERT OR REPLACE INTO module_roots VALUES(?,?,?,?)", (path, root, name, digest))
        reresolve_links(db, root, name)
        if old is not None and old[0] != name:
            db.execute("UPDATE package_links SET to_root=NULL, resolution='unresolved', reason='package_renamed' WHERE to_name=? AND resolution='local_name'", (old[0],))
        if old is None or old[0] != name:
            # Manifests whose plain dependency edges name the new root may now form local
            # links; re-queue them through the bounded pending path to re-emit links.
            # Only a genuinely new/renamed root triggers this, so dependency cycles
            # cannot re-enqueue each other forever.
            db.execute("""INSERT OR IGNORE INTO pending
                SELECT DISTINCT f.path,'package_added' FROM files f
                JOIN edges e ON e.file_id=f.id
                WHERE e.target=? AND e.kind IN ('package_dependency','build_dependency')""", (name,))
            db.execute("""UPDATE files SET state='pending' WHERE path IN (
                SELECT DISTINCT f.path FROM files f JOIN edges e ON e.file_id=f.id
                WHERE e.target=? AND e.kind IN ('package_dependency','build_dependency'))""", (name,))
    else:
        db.execute("DELETE FROM module_roots WHERE manifest=?", (path,))
        # Links into a removed root degrade to explainable unresolved rows.
        db.execute("UPDATE package_links SET to_root=NULL, resolution='unresolved', reason='package_removed' WHERE to_root=?", (root,))
    after = bool(db.execute("SELECT 1 FROM module_roots WHERE root=? LIMIT 1", (root,)).fetchone())
    if before != after:
        # Publish invalidation with the boundary. Existing maintenance budgets
        # rebuild the affected subtree; readers cannot use obsolete module facts.
        where, args = ("1=1", ()) if root == "." else ("path>=? AND path<?", (root + "/", root + "0"))
        db.execute(f"INSERT OR IGNORE INTO pending SELECT path,'module_changed' FROM files WHERE {where}", args)
        db.execute(f"UPDATE files SET state='pending' WHERE {where}", args)

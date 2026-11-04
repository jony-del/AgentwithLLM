"""Dependency-free Python structure extraction; other languages stay text-only.

References are syntactic candidates, never presented as type-resolved calls.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import PurePosixPath
from typing import Any

LANGUAGES = {".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
             ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
             ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp"}
PARSER_FINGERPRINT = "python-ast-v1"


def parser_fingerprint() -> str:
    # AST grammar changes with the interpreter; static manifest extraction is
    # versioned independently even though both currently publish in one batch.
    return f"{PARSER_FINGERPRINT}:py{sys.version_info.major}.{sys.version_info.minor}:manifests-v2:module-boundaries-v1"


def language(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def module(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "."


def python_structure(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tree = ast.parse(text)
    symbols: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: list[str] = []

        def definition(self, node: Any, kind: str) -> None:
            qualified = ".".join([*self.names, node.name])
            symbols.append({"name": node.name, "qualified": qualified, "kind": kind,
                            "line": node.lineno, "end_line": node.end_lineno})
            self.names.append(node.name)
            self.generic_visit(node)
            self.names.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.definition(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.definition(node, "function")

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for base in node.bases:
                edges.append({"source": node.name, "target": ast.unparse(base), "kind": "inherits",
                              "line": node.lineno, "precision": "syntactic"})
            self.definition(node, "class")

        def variable(self, target: ast.AST, node: Any) -> None:
            if isinstance(target, ast.Name):
                symbols.append({"name": target.id, "qualified": ".".join([*self.names, target.id]),
                                "kind": "variable", "line": node.lineno, "end_line": node.end_lineno})
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    self.variable(item, node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                self.variable(target, node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self.variable(node.target, node)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, (ast.Name, ast.Attribute)):
                edges.append({"source": ".".join(self.names), "target": ast.unparse(node.func),
                              "kind": "calls", "line": node.lineno, "precision": "syntactic"})
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                edges.append({"source": ".".join(self.names), "target": node.id,
                              "kind": "references", "line": node.lineno, "precision": "syntactic"})

        def visit_Import(self, node: ast.Import) -> None:
            for item in node.names:
                edges.append({"source": "", "target": item.name, "kind": "imports",
                              "line": node.lineno, "precision": "syntactic"})

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            edges.append({"source": "", "target": "." * node.level + (node.module or ""),
                          "kind": "imports", "line": node.lineno, "precision": "syntactic"})

    Visitor().visit(tree)
    return symbols, edges


def manifest_links(path: str, text: str) -> list[dict[str, Any]]:
    """Structured dependency references from a manifest: name, version/path/workspace
    markers and workspace-member globs. Static parse only; never executes scripts."""
    name = PurePosixPath(path).name
    refs: list[dict[str, Any]] = []

    def line_of(needle: str) -> int:
        return next((i for i, body in enumerate(text.splitlines(), 1) if needle in body), 1)

    def ref(target: str, kind: str, *, version: str | None = None, path_ref: str | None = None,
            member_glob: str | None = None) -> None:
        refs.append({"name": target, "kind": kind, "line": line_of(target),
                     "version": version, "path": path_ref, "member_glob": member_glob})

    if name == "package.json":
        value = json.loads(text)
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            entries = value.get(section, {})
            if not isinstance(entries, dict):
                continue
            for key, spec in entries.items():
                spec_text = str(spec)
                if spec_text.startswith(("file:", "link:")):
                    ref(str(key), "package_dependency", path_ref=spec_text.split(":", 1)[1])
                elif spec_text.startswith("workspace:"):
                    ref(str(key), "package_dependency", version="workspace")
                else:
                    ref(str(key), "package_dependency", version=spec_text[:128])
        workspaces = value.get("workspaces")
        globs = workspaces.get("packages", []) if isinstance(workspaces, dict) else workspaces
        if isinstance(globs, list):
            for pattern in globs:
                if isinstance(pattern, str) and pattern.strip():
                    ref(pattern.strip(), "workspace_member", member_glob=pattern.strip()[:256])
    elif name == "pyproject.toml":
        value = tomllib.loads(text)
        for item in value.get("project", {}).get("dependencies", []):
            target = re.split(r"[<>=!~;\[]", str(item), maxsplit=1)[0].strip()
            if target:
                ref(target, "package_dependency", version=str(item)[:128])
        for item in value.get("build-system", {}).get("requires", []):
            target = re.split(r"[<>=!~;\[]", str(item), maxsplit=1)[0].strip()
            if target:
                ref(target, "build_dependency", version=str(item)[:128])
        # Explicitly scoped local sources only: uv sources and Poetry path dependencies.
        uv_sources = value.get("tool", {}).get("uv", {}).get("sources", {})
        if isinstance(uv_sources, dict):
            for key, source in uv_sources.items():
                if isinstance(source, dict) and isinstance(source.get("path"), str):
                    ref(str(key), "package_dependency", path_ref=source["path"][:512])
        poetry_deps = value.get("tool", {}).get("poetry", {}).get("dependencies", {})
        if isinstance(poetry_deps, dict):
            for key, spec in poetry_deps.items():
                if isinstance(spec, dict) and isinstance(spec.get("path"), str):
                    ref(str(key), "package_dependency", path_ref=spec["path"][:512])
    elif name == "Cargo.toml":
        value = tomllib.loads(text)
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            entries = value.get(section, {})
            if not isinstance(entries, dict):
                continue
            kind = "build_dependency" if section == "build-dependencies" else "package_dependency"
            for key, spec in entries.items():
                if isinstance(spec, dict):
                    if isinstance(spec.get("path"), str):
                        ref(str(key), kind, path_ref=spec["path"][:512])
                    else:
                        ref(str(key), kind, version=str(spec.get("version", ""))[:128])
                else:
                    ref(str(key), kind, version=str(spec)[:128])
        workspace = value.get("workspace", {})
        if isinstance(workspace, dict) and isinstance(workspace.get("members"), list):
            for pattern in workspace["members"]:
                if isinstance(pattern, str) and pattern.strip():
                    ref(pattern.strip(), "workspace_member", member_glob=pattern.strip()[:256])
    elif name == "go.mod":
        block = None
        replaced: set[str] = set()
        deferred_requires: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.split("//", 1)[0].strip()
            if line in {"require (", "replace ("}:
                block = line.split()[0]
                continue
            if line == ")":
                block = None
                continue
            parts = line.split()
            if line.startswith("require ") or block == "require":
                parts = line.removeprefix("require ").split()
                if parts:
                    deferred_requires.append({"name": parts[0], "kind": "package_dependency", "line": line_number,
                                              "version": (parts[1][:128] if len(parts) > 1 else None),
                                              "path": None, "member_glob": None})
            elif line.startswith("replace ") or block == "replace":
                parts = line.removeprefix("replace ").split()
                if "=>" in parts:
                    arrow = parts.index("=>")
                    target = parts[arrow + 1] if len(parts) > arrow + 1 else ""
                    if target.startswith(("./", "../")):
                        replaced.add(parts[0])
                        refs.append({"name": parts[0], "kind": "package_dependency", "line": line_number,
                                     "version": None, "path": target[:512], "member_glob": None})
        # A local path replace owns the module link; the require entry stays an edge only.
        refs.extend(r for r in deferred_requires if r["name"] not in replaced)
    return refs


def manifest_dependencies(path: str, text: str) -> list[dict[str, Any]]:
    """Static manifest facts only; never execute package/build scripts."""
    name = PurePosixPath(path).name
    dependencies: list[tuple[str, str]] = []
    if name == "package.json":
        value = json.loads(text)
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            dependencies.extend((str(key), "package_dependency") for key in value.get(section, {}))
    elif name in {"pyproject.toml", "Cargo.toml"}:
        value = tomllib.loads(text)
        if name == "pyproject.toml":
            dependencies.extend((str(item), "package_dependency") for item in value.get("project", {}).get("dependencies", []))
            dependencies.extend((str(item), "build_dependency") for item in value.get("build-system", {}).get("requires", []))
        else:
            for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                dependencies.extend((str(key), "build_dependency" if section == "build-dependencies" else "package_dependency")
                                    for key in value.get(section, {}))
    elif name == "go.mod":
        block = False
        for line in text.splitlines():
            line = line.split("//", 1)[0].strip()
            if line == "require (":
                block = True
            elif line == ")":
                block = False
            elif line.startswith("require ") or block:
                parts = line.removeprefix("require ").split()
                if parts:
                    dependencies.append((parts[0], "package_dependency"))
    edges = []
    for target, kind in dependencies:
        target = re.split(r"[<>=!~;\[]", target, maxsplit=1)[0].strip()
        line_number = next((i for i, body in enumerate(text.splitlines(), 1) if target in body), 1)
        edges.append({"source": module(path), "target": target, "kind": kind, "line": line_number, "precision": "manifest"})
    return edges

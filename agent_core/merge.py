"""Conservative three-way source merge without reparsing/reprinting source text."""
from __future__ import annotations

import ast
import copy
import difflib


class MergeConflict(ValueError):
    pass


def _owners(text: str) -> dict[str, str]:
    result = {}

    def statements(body, prefix):
        other = []
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = prefix + node.name
                if key in result:
                    raise MergeConflict("duplicate Python symbol ownership")
                if isinstance(node, ast.ClassDef):
                    header = copy.copy(node)
                    header.body = []
                    result[key] = ast.dump(header, include_attributes=False)
                    statements(node.body, key + ".")
                else:
                    result[key] = ast.dump(node, include_attributes=False)
            else:
                other.append(node)
        result[prefix + "<statements>"] = ast.dump(ast.Module(body=other, type_ignores=[]), include_attributes=False)
    statements(ast.parse(text.removeprefix("\ufeff")).body, "")
    return result


def _ast_guard(base: str, current: str, incoming: str) -> None:
    maps = [_owners(text) for text in (base, current, incoming)]
    left = {key for key in maps[0].keys() | maps[1].keys() if maps[0].get(key) != maps[1].get(key)}
    right = {key for key in maps[0].keys() | maps[2].keys() if maps[0].get(key) != maps[2].get(key)}
    if any(a == b or a.startswith(b + ".") or b.startswith(a + ".") for a in left for b in right):
        raise MergeConflict("both branches changed the same Python symbol or class interface")


def three_way_merge(base: str, current: str, incoming: str, *, python: bool = False) -> str:
    if current == incoming or incoming == base:
        return current
    if current == base:
        return incoming
    if python:
        try:
            _ast_guard(base, current, incoming)
        except SyntaxError as exc:
            raise MergeConflict("Python input does not parse") from exc
    original = base.splitlines(keepends=True)

    def edits(text):
        lines = text.splitlines(keepends=True)
        return [(first, last, tuple(lines[start:end])) for tag, first, last, start, end in
                difflib.SequenceMatcher(a=original, b=lines, autojunk=False).get_opcodes() if tag != "equal"]
    left, right = edits(current), edits(incoming)
    for a, b, replacement in left:
        for c, d, addition in right:
            if (a, b, replacement) == (c, d, addition):
                continue
            overlaps = max(a, c) < min(b, d)
            insertion = (a == b and c <= a <= d) or (c == d and a <= c <= b)
            if overlaps or insertion:
                raise MergeConflict("overlapping edits or ambiguous insertion boundary")
    merged = list(original)
    for first, last, replacement in sorted(set(left + right), reverse=True):
        merged[first:last] = replacement
    result = "".join(merged)
    if python:
        try:
            ast.parse(result.removeprefix("\ufeff"))
        except SyntaxError as exc:
            raise MergeConflict("merged Python source does not parse") from exc
    return result

"""YAML-frontmatter parsing for skill files (PyYAML-backed).

A skill file is an optional ``---`` fenced YAML block followed by a Markdown body.
We own the fence detection (so a stray fence can't swallow the body) and the key
normalisation (``when-to-use`` -> ``when_to_use``), and delegate the actual value
parsing to ``yaml.safe_load`` — giving full YAML support (nested maps, typed scalars,
block/flow lists) instead of the previous hand-rolled subset.

Parsing is best-effort: malformed YAML degrades to "no metadata" (``({}, text)``)
rather than raising, matching the loader's "skip a bad file, never crash" contract.
"""

from __future__ import annotations

import yaml

_FENCE = "---"


def _normalise_keys(data: dict) -> dict[str, object]:
    """Lower-case keys and turn hyphens into underscores (``allowed-tools`` -> ``allowed_tools``)."""
    normalised: dict[str, object] = {}
    for key, value in data.items():
        normalised[str(key).strip().lower().replace("-", "_")] = value
    return normalised


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split ``text`` into a ``(metadata, body)`` pair.

    With no leading ``---`` fence the whole text is the body and metadata is empty.
    A second ``---`` closes the block; if it never appears the file is treated as pure
    body (no metadata) so a stray fence can't swallow the content. Any YAML error, or a
    block that isn't a mapping, also degrades to ``({}, text)``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return {}, text

    close = None
    for index in range(1, len(lines)):
        if lines[index].strip() == _FENCE:
            close = index
            break
    if close is None:
        return {}, text

    block = "\n".join(lines[1:close])
    try:
        loaded = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}, text
    if not isinstance(loaded, dict):
        # Empty frontmatter (``None``) or a non-mapping scalar/list — no usable metadata.
        return {}, "\n".join(lines[close + 1:])

    meta = _normalise_keys(loaded)
    body = "\n".join(lines[close + 1:])
    return meta, body

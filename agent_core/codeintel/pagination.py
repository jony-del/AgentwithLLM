"""Generation-bound keyset positions for files, source lines and indexed facts."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchPosition:
    path: str = ""
    line: int = 0
    fact: int = 0
    file_complete: bool = False

    def cursor(self, *, query: str, generation: int, kind: str, modules: tuple[str, ...]) -> str:
        value = {"version": 2, "query": query, "generation": generation, "kind": kind,
                 "modules": modules, "position": asdict(self)}
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def decode_cursor(raw: str, query: str) -> tuple[SearchPosition, dict[str, Any]]:
    try:
        if len(raw) > 32768:
            raise ValueError("oversized cursor")
        value = json.loads(base64.urlsafe_b64decode(raw))
        if value["version"] != 2 or value["query"] != query:
            raise ValueError("cursor version or query changed")
        position = SearchPosition(**value["position"])
        if (not isinstance(position.path, str) or type(position.line) is not int or position.line < 0
                or type(position.fact) is not int or position.fact < 0 or type(position.file_complete) is not bool
                or type(value["generation"]) is not int or not isinstance(value["modules"], list)
                or any(not isinstance(m, str) for m in value["modules"])):
            raise ValueError("invalid cursor fields")
        return position, value
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid query cursor; restart pagination without cursor") from exc

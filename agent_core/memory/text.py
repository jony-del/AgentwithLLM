from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in",
        "on", "at", "for", "with", "as", "by", "is", "are", "was", "were", "be", "been",
        "being", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
        "we", "they", "me", "my", "your", "our", "their", "do", "does", "did", "done",
        "have", "has", "had", "will", "would", "can", "could", "should", "shall", "may",
        "might", "must", "not", "no", "so", "than", "too", "very", "just", "about",
    }
)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_IDENTIFIER_RE = re.compile(
    r"(?<![\w])(?:--?[A-Za-z][\w-]*|[A-Za-z_][\w]*(?:[.:/\\][A-Za-z_][\w-]*)+|"
    r"[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][A-Za-z0-9]*)+)(?![\w])"
)
_PATH_RE = re.compile(
    r"(?<![\w])(?:(?:[A-Za-z]:[\\/]|\.{0,2}[\\/]|/)(?:[^\s\"'<>|:*?]+[\\/]?)+|"
    r"[A-Za-z0-9_.-]+[\\/](?:[A-Za-z0-9_.-]+[\\/]?)+)"
)
_VERSION_RE = re.compile(r"(?<![\w])v?\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?(?![\w])")
_COMMIT_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{7,40}(?![0-9a-fA-F])")
_ERROR_RE = re.compile(r"(?<![\w])(?:[A-Z][A-Z0-9_]{1,15}[-_:]?\d{2,}|E\d{3,})(?![\w])")
_QUOTED_RE = re.compile(r"[\"'“”‘’`]+([^\"'“”‘’`\r\n]{2,200})[\"'“”‘’`]+")
_ATOM_KINDS = (
    ("path", _PATH_RE),
    ("identifier", _IDENTIFIER_RE),
    ("version", _VERSION_RE),
    ("commit", _COMMIT_RE),
    ("error", _ERROR_RE),
)


def normalize_text(text: str) -> str:
    """Canonicalize text used by exact matching without changing source content."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = normalized.replace("\\", "/")
    return " ".join(normalized.split())


def _identifier_parts(value: str) -> list[str]:
    value = value.replace("\\", "/")
    pieces: list[str] = []
    for segment in re.split(r"[._:/-]+", value):
        if not segment:
            continue
        camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", segment)
        pieces.extend(part.casefold() for part in camel.split() if part)
    return pieces


def lexical_tokens(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Return frequency-preserving Unicode/CJK/code tokens suitable for FTS input.

    FTS receives only tokens generated here, never the raw query. Identifier hashes
    preserve a whole code/path form even when SQLite's tokenizer splits punctuation.
    """
    normalized = unicodedata.normalize("NFKC", str(text))
    lowered = normalized.casefold()
    tokens: list[str] = []
    for token in _WORD_RE.findall(lowered):
        if _CJK_RE.fullmatch(token):
            continue
        if len(token) <= 1 or (drop_stopwords and token in _STOPWORDS):
            continue
        tokens.append(token)
    for match in _CJK_RE.finditer(lowered):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
        for width in (2, 3):
            tokens.extend(run[index : index + width] for index in range(len(run) - width + 1))
    for match in _IDENTIFIER_RE.finditer(normalized):
        whole = normalize_text(match.group(0))
        # A deterministic alphanumeric sentinel survives unicode61 tokenization.
        import hashlib

        tokens.append("ident" + hashlib.sha256(whole.encode("utf-8")).hexdigest()[:20])
        tokens.extend(part for part in _identifier_parts(match.group(0)) if len(part) > 1)
    return tokens


def tokenize(text: str) -> set[str]:
    """Compatibility set view used by extraction/deduplication code."""
    return set(lexical_tokens(text, drop_stopwords=True))


def lexical_relevance(a: set[str], b: set[str]) -> float:
    """Return overlap coefficient ``|a∩b| / min(|a|,|b|)`` in [0, 1]."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / min(len(a), len(b))


def exact_atoms(text: str) -> list[tuple[str, str]]:
    """Extract high-precision atoms; ordinary prose words are intentionally absent."""
    found: list[tuple[str, str]] = []
    for match in _QUOTED_RE.finditer(str(text)):
        value = normalize_text(match.group(1))
        if value:
            found.append(("phrase", value))
    for kind, pattern in _ATOM_KINDS:
        for match in pattern.finditer(str(text)):
            value = normalize_text(match.group(0)).rstrip(".,;:)")
            if value:
                found.append((kind, value))
    return list(dict.fromkeys(found))


def field_atoms(kind: str, values: Iterable[str]) -> list[tuple[str, str]]:
    """Canonical atoms for structured document metadata."""
    result: list[tuple[str, str]] = []
    for raw in values:
        value = normalize_text(raw)
        if value:
            result.append((kind, value))
        result.extend(exact_atoms(raw))
    return list(dict.fromkeys(result))


def fts_query(text: str, *, max_tokens: int = 128) -> str:
    """Build a literal-only FTS5 query from generated tokens.

    Each term is quoted and quotes are doubled. No raw operator, parenthesis, NEAR,
    column selector, wildcard, or minus sign from user input reaches MATCH.
    """
    terms = lexical_tokens(text, drop_stopwords=True)[:max_tokens]
    quoted = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms if term]
    return " OR ".join(quoted)


def fts_document(text: str, *, max_tokens: int = 200_000) -> str:
    """Render generated terms for an FTS column while retaining term frequency."""
    return " ".join(lexical_tokens(text)[:max_tokens])

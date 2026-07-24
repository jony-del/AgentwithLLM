from __future__ import annotations

import re

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


def tokenize(text: str) -> set[str]:
    """Produce Unicode word tokens plus CJK character bi/tri-grams."""
    lowered = text.casefold()
    tokens = {
        token
        for token in _WORD_RE.findall(lowered)
        if len(token) > 1 and token not in _STOPWORDS and not _CJK_RE.fullmatch(token)
    }
    for match in _CJK_RE.finditer(lowered):
        run = match.group(0)
        if len(run) == 1:
            tokens.add(run)
        for width in (2, 3):
            tokens.update(run[index : index + width] for index in range(len(run) - width + 1))
    return tokens


def lexical_relevance(a: set[str], b: set[str]) -> float:
    """Return overlap coefficient ``|a∩b| / min(|a|,|b|)`` in [0, 1]."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / min(len(a), len(b))

from __future__ import annotations

import re

# Small, deliberately generic stopword set. The goal is only to stop the most common
# function words from dominating the lexical overlap; this is not meant to be a
# linguistically complete list (zero-dependency project — no NLTK).
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

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric runs, drop stopwords and 1-char tokens.

    Returns a *set* — recall here cares about which concepts overlap, not how many
    times each appears, so set overlap is the right primitive.
    """
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def lexical_relevance(a: set[str], b: set[str]) -> float:
    """Overlap coefficient ``|a∩b| / min(|a|,|b|)`` in [0,1].

    Overlap coefficient (rather than Jaccard) is chosen so that a short, specific
    memory still scores highly against a long query that contains it — we don't want
    to penalise a memory just because the query has many other unrelated words.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / min(len(a), len(b))

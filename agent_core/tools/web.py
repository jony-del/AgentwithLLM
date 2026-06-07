"""Web access tools: ``web_fetch`` (URL → markdown) and ``web_search``.

These are the first tools to take third-party dependencies (httpx / beautifulsoup4 /
markdownify / ddgs). To keep the core importable without the ``web`` extra, every
third-party import is **lazy** (inside the network seam functions), so merely importing
this module — which catalog discovery does at startup — never requires the extra.

Network egress is guarded against SSRF: ``_check_url_safe`` rejects non-http(s) schemes
and any host that resolves to a loopback/private/link-local/reserved address, and the
fetch re-checks the target on every redirect hop so a redirect can't smuggle the request
to an internal service.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

from agent_core.models import ToolRisk, ToolResult
from agent_core.tools.base import Tool
from agent_core.tools.catalog import builtin_tool

_USER_AGENT = "AgentwithLLM/0.1 (+https://github.com/) web_fetch"
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_TIMEOUT = 20.0
_MAX_REDIRECTS = 5
_MISSING_DEPS = (
    'web tools need the "web" extra — install with: pip install -e ".[web]" '
    "(httpx, beautifulsoup4, markdownify, ddgs)."
)


class WebError(Exception):
    """A web request was rejected (unsafe URL) or failed."""

    def __init__(self, message: str, error_type: str = "WebError") -> None:
        super().__init__(message)
        self.error_type = error_type


# --- SSRF guard (pure stdlib, always available) ----------------------------------


def _check_url_safe(url: str) -> None:
    """Raise ``WebError`` unless ``url`` is an http(s) URL whose host resolves to a
    public address. Blocks loopback/private/link-local/reserved/multicast targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebError(f"only http/https URLs are allowed, got '{parsed.scheme or '(none)'}'", "BadURL")
    host = parsed.hostname
    if not host:
        raise WebError("URL has no host", "BadURL")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise WebError(f"cannot resolve host '{host}': {exc}", "BadURL") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise WebError(f"refusing to fetch internal/private address {ip} (host '{host}')", "SSRF")


# --- network seams (lazy imports; monkeypatched in tests) ------------------------


def _fetch_url(url: str, *, timeout: float = _DEFAULT_TIMEOUT, max_hops: int = _MAX_REDIRECTS) -> tuple[str, str, str]:
    """Fetch ``url``, following redirects manually and re-checking each hop for SSRF.

    Returns ``(final_url, content_type, text)``. Raises ``WebError`` on an unsafe or
    failing request, or ``ImportError`` if httpx isn't installed.
    """
    import httpx  # lazy: only needed when the tool actually runs

    headers = {"User-Agent": _USER_AGENT}
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for _ in range(max_hops + 1):
            _check_url_safe(current)
            response = client.get(current)
            location = response.headers.get("location")
            if response.status_code in (301, 302, 303, 307, 308) and location:
                current = urljoin(current, location)
                continue
            if response.status_code >= 400:
                raise WebError(f"HTTP {response.status_code} for {current}", "HTTPError")
            return str(response.url), response.headers.get("content-type", ""), response.text
    raise WebError(f"too many redirects (>{max_hops})", "TooManyRedirects")


def _html_to_markdown(html: str) -> str:
    """Strip scripts/styles and convert HTML to markdown. Lazy bs4 + markdownify."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return markdownify(str(soup)).strip()


def _search_ddgs(query: str, max_results: int) -> list[dict]:
    """Keyless DuckDuckGo search. Handles both the new ``ddgs`` and old package name."""
    try:
        from ddgs import DDGS
    except ImportError:  # older releases shipped as duckduckgo_search
        from duckduckgo_search import DDGS

    rows = DDGS().text(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in rows
    ]


def _search_brave(query: str, max_results: int, api_key: str) -> list[dict]:
    import httpx

    response = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=_DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    rows = response.json().get("web", {}).get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in rows[:max_results]
    ]


def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    import httpx

    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=_DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    rows = response.json().get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in rows[:max_results]
    ]


def _select_search(query: str, max_results: int) -> tuple[str, list[dict]]:
    """Pick a backend by which API key (if any) is set; default to keyless ddgs."""
    brave = os.environ.get("BRAVE_API_KEY")
    tavily = os.environ.get("TAVILY_API_KEY")
    if brave:
        return "brave", _search_brave(query, max_results, brave)
    if tavily:
        return "tavily", _search_tavily(query, max_results, tavily)
    return "ddgs", _search_ddgs(query, max_results)


# --- tools -----------------------------------------------------------------------


@builtin_tool
class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a public http(s) URL and return its main content as markdown (HTML is "
        "cleaned and converted; other text is returned as-is). Internal/private addresses "
        "are refused. Use this to read a specific page you already have a URL for."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http(s) URL to fetch."},
            "max_chars": {"type": "integer", "description": f"Truncate output to this many chars (default {_DEFAULT_MAX_CHARS})."},
        },
        "required": ["url"],
    }
    risk = ToolRisk.READ

    def run(self, arguments: dict[str, object]) -> ToolResult:
        url = str(arguments.get("url", "")).strip()
        max_chars = int(arguments.get("max_chars", _DEFAULT_MAX_CHARS))
        if not url:
            return ToolResult(self.name, "url must not be empty", ok=False, metadata={"error_type": "BadArgs"})
        # Pre-check before any network/dep work so an unsafe URL is rejected cheaply.
        # _fetch_url re-checks every redirect hop (including this one) too.
        try:
            _check_url_safe(url)
        except WebError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type})
        try:
            final_url, content_type, text = _fetch_url(url)
        except ImportError:
            return ToolResult(self.name, _MISSING_DEPS, ok=False, metadata={"error_type": "MissingDeps"})
        except WebError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type})
        except Exception as exc:  # noqa: BLE001 - network/parse errors → readable result
            return ToolResult(self.name, f"Fetch failed: {type(exc).__name__}: {exc}", ok=False, metadata={"error_type": "FetchError"})

        try:
            body = _html_to_markdown(text) if "html" in content_type.lower() else text.strip()
        except ImportError:
            return ToolResult(self.name, _MISSING_DEPS, ok=False, metadata={"error_type": "MissingDeps"})
        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars] + "\n\n[... truncated ...]"
        header = f"# Fetched: {final_url}\n\n"
        return ToolResult(self.name, header + (body or "(empty document)"), metadata={"final_url": final_url, "truncated": truncated})


@builtin_tool
class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return a ranked list of results (title, URL, snippet). Uses a "
        "keyless DuckDuckGo backend by default; set BRAVE_API_KEY or TAVILY_API_KEY to use "
        "those instead. Follow up with web_fetch to read a specific result."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "How many results to return (default 5)."},
        },
        "required": ["query"],
    }
    risk = ToolRisk.READ

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results", 5))
        if not query:
            return ToolResult(self.name, "query must not be empty", ok=False, metadata={"error_type": "BadArgs"})
        try:
            backend, results = _select_search(query, max_results)
        except ImportError:
            return ToolResult(self.name, _MISSING_DEPS, ok=False, metadata={"error_type": "MissingDeps"})
        except Exception as exc:  # noqa: BLE001 - backend/network errors → readable result
            return ToolResult(self.name, f"Search failed: {type(exc).__name__}: {exc}", ok=False, metadata={"error_type": "SearchError"})

        if not results:
            return ToolResult(self.name, f"No results for '{query}'.", metadata={"backend": backend, "count": 0})
        lines = []
        for index, row in enumerate(results, start=1):
            lines.append(f"{index}. {row.get('title', '(no title)')}\n   {row.get('url', '')}\n   {row.get('snippet', '')}")
        body = f"Results for '{query}' (via {backend}):\n\n" + "\n\n".join(lines)
        return ToolResult(self.name, body, metadata={"backend": backend, "count": len(results)})

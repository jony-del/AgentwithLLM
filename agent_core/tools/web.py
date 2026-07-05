"""Web access tools: ``web_fetch`` (URL → markdown) and ``web_search``.

Their third-party dependencies (beautifulsoup4 / markdownify / ddgs) live in the
optional ``[web]`` extra (httpx is core). When the extra is not installed this module
still imports cleanly — the tools simply skip registration with a logged warning, so
a core-only install degrades to fewer tools, never an import-time crash.

Network egress is guarded against SSRF: ``_check_url_safe`` rejects non-http(s) schemes
and any host that resolves to a loopback/private/link-local/reserved address, and the
fetch re-checks the target on every redirect hop so a redirect can't smuggle the request
to an internal service.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_rules import _match_domain
from agent_core.tools.base import ConcurrencySpec, Tool
from agent_core.tools.catalog import builtin_tool

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    from ddgs import DDGS
    from markdownify import markdownify

    _MISSING_WEB_DEP: str | None = None
except ModuleNotFoundError as _exc:  # the [web] extra is not installed
    BeautifulSoup = DDGS = markdownify = None  # type: ignore[assignment]
    _MISSING_WEB_DEP = _exc.name or str(_exc)
    logger.warning(
        "web tools disabled: missing dependency %r — pip install 'agent-with-llm[web]' (or [all])",
        _MISSING_WEB_DEP,
    )


def _web_tool(cls: type[Tool]) -> type[Tool]:
    """Register a web tool as a built-in only when the [web] extra is installed."""
    if _MISSING_WEB_DEP is None:
        return builtin_tool(cls)
    return cls

_USER_AGENT = "AgentwithLLM/0.1 (+https://github.com/) web_fetch"
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_TIMEOUT = 20.0
_MAX_REDIRECTS = 5


class WebError(Exception):
    """A web request was rejected (unsafe URL) or failed."""

    def __init__(self, message: str, error_type: str = "WebError") -> None:
        super().__init__(message)
        self.error_type = error_type


# --- outbound domain policy (decision D10, S4) ------------------------------------


@dataclass(slots=True)
class WebPolicyConfig:
    """The ``[web]`` outbound domain policy (decision D10 — the S4 exfiltration guard).

    ``blocked_domains`` always refuse (tightening — applies in every mode).
    ``allowed_domains`` matter only to *unattended* permission modes (auto/dontask/
    bypass): there, a domain not on the list is refused (fail-closed), because nobody
    is present to notice an injection-driven exfiltration fetch. Attended modes stay
    open by default. Both lists use the same host-suffix matching as the
    ``domain:example.com`` permission-rule form. In an in-repo ``agent.toml``,
    ``allowed_domains`` is a privilege-widening key gated by TOFU (``trust.py``).
    """

    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WebPolicyConfig":
        from agent_core.config import overlay_dataclass

        return overlay_dataclass(cls(), data)


_DEFAULT_WEB_POLICY = WebPolicyConfig()


def _check_domain_policy(url_or_host: str, policy: WebPolicyConfig, unattended: bool) -> None:
    """Raise :class:`WebError` when ``url_or_host`` violates the bound egress policy."""
    host = urlparse(url_or_host).hostname or url_or_host
    for domain in policy.blocked_domains:
        if _match_domain(host, domain):
            raise WebError(
                f"domain {host!r} is blocked by [web].blocked_domains", "BlockedDomain"
            )
    if not unattended:
        return
    if any(_match_domain(host, domain) for domain in policy.allowed_domains):
        return
    raise WebError(
        f"unattended permission mode: domain {host!r} is not in [web].allowed_domains "
        "(fail-closed for auto/dontask/bypass runs). Add the domain to "
        "[web].allowed_domains — an in-repo grant needs one-time TOFU approval — "
        "or run in an attended mode.",
        "DomainNotAllowed",
    )


class WebPolicyAwareMixin:
    """Mixin giving a web tool the bound egress policy + the run's attendance.

    Like ``SandboxAwareMixin``: a permissive class-level default stands in until
    ``ReActAgent`` rebinds the live policy at construction.
    """

    web_policy: WebPolicyConfig = _DEFAULT_WEB_POLICY
    unattended: bool = False

    def bind_web_policy(self, policy: WebPolicyConfig, *, unattended: bool) -> None:
        self.web_policy = policy
        self.unattended = unattended

    def check_domain_policy(self, url_or_host: str) -> None:
        _check_domain_policy(url_or_host, self.web_policy, self.unattended)


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


# --- network seams (monkeypatched in tests) --------------------------------------


def _fetch_url(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_hops: int = _MAX_REDIRECTS,
    hop_check: Callable[[str], None] | None = None,
) -> tuple[str, str, str]:
    """Fetch ``url``, following redirects manually and re-checking each hop for SSRF.

    ``hop_check`` (when given) runs on every hop alongside the SSRF check — the domain
    policy uses it so a redirect can't smuggle the request to a blocked/unlisted host.
    Returns ``(final_url, content_type, text)``. Raises ``WebError`` on an unsafe or
    failing request.
    """
    headers = {"User-Agent": _USER_AGENT}
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for _ in range(max_hops + 1):
            _check_url_safe(current)
            if hop_check is not None:
                hop_check(current)
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
    """Strip scripts/styles and convert HTML to markdown."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return markdownify(str(soup)).strip()


def _search_ddgs(query: str, max_results: int) -> list[dict]:
    """Keyless DuckDuckGo search."""
    rows = DDGS().text(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in rows
    ]


def _search_brave(query: str, max_results: int, api_key: str) -> list[dict]:
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


# The outbound host each search backend talks to — what the domain policy is checked
# against for web_search (the QUERY itself is the exfiltration channel there).
_SEARCH_BACKEND_DOMAINS = {
    "brave": "api.search.brave.com",
    "tavily": "api.tavily.com",
    "ddgs": "duckduckgo.com",
}


def _active_search_backend() -> str:
    """Which backend a search would use right now (keyed off the env API keys)."""
    if os.environ.get("BRAVE_API_KEY"):
        return "brave"
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    return "ddgs"


def _select_search(query: str, max_results: int) -> tuple[str, list[dict]]:
    """Run the search on the active backend; default to keyless ddgs."""
    backend = _active_search_backend()
    if backend == "brave":
        return "brave", _search_brave(query, max_results, os.environ["BRAVE_API_KEY"])
    if backend == "tavily":
        return "tavily", _search_tavily(query, max_results, os.environ["TAVILY_API_KEY"])
    return "ddgs", _search_ddgs(query, max_results)


# --- tools -----------------------------------------------------------------------


@_web_tool
class WebFetchTool(WebPolicyAwareMixin, Tool):
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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec()

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        url = str(arguments.get("url", "")).strip()
        max_chars = int(arguments.get("max_chars", _DEFAULT_MAX_CHARS))
        if not url:
            return ToolResult(self.name, "url must not be empty", ok=False, metadata={"error_type": "BadArgs"})
        # Pre-check before any network/dep work so an unsafe URL is rejected cheaply.
        # _fetch_url re-checks every redirect hop (including this one) too.
        try:
            _check_url_safe(url)
            self.check_domain_policy(url)
        except WebError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type})
        try:
            final_url, content_type, text = _fetch_url(url, hop_check=self.check_domain_policy)
        except WebError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type})
        except Exception as exc:  # noqa: BLE001 - network/parse errors → readable result
            return ToolResult(self.name, f"Fetch failed: {type(exc).__name__}: {exc}", ok=False, metadata={"error_type": "FetchError"})

        body = _html_to_markdown(text) if "html" in content_type.lower() else text.strip()
        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars] + "\n\n[... truncated ...]"
        header = f"# Fetched: {final_url}\n\n"
        return ToolResult(self.name, header + (body or "(empty document)"), metadata={"final_url": final_url, "truncated": truncated})


@_web_tool
class WebSearchTool(WebPolicyAwareMixin, Tool):
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

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec()

    def _invoke(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results", 5))
        if not query:
            return ToolResult(self.name, "query must not be empty", ok=False, metadata={"error_type": "BadArgs"})
        # The query itself leaves the machine, so the policy is checked against the
        # search backend's own host before any request is made.
        try:
            self.check_domain_policy(_SEARCH_BACKEND_DOMAINS[_active_search_backend()])
        except WebError as exc:
            return ToolResult(self.name, str(exc), ok=False, metadata={"error_type": exc.error_type})
        try:
            backend, results = _select_search(query, max_results)
        except Exception as exc:  # noqa: BLE001 - backend/network errors → readable result
            return ToolResult(self.name, f"Search failed: {type(exc).__name__}: {exc}", ok=False, metadata={"error_type": "SearchError"})

        if not results:
            return ToolResult(self.name, f"No results for '{query}'.", metadata={"backend": backend, "count": 0})
        lines = []
        for index, row in enumerate(results, start=1):
            lines.append(f"{index}. {row.get('title', '(no title)')}\n   {row.get('url', '')}\n   {row.get('snippet', '')}")
        body = f"Results for '{query}' (via {backend}):\n\n" + "\n\n".join(lines)
        return ToolResult(self.name, body, metadata={"backend": backend, "count": len(results)})

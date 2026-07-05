import pytest

from agent_core.models import ToolRisk
from agent_core.tools import web
from agent_core.tools.web import (
    WebError,
    WebFetchTool,
    WebPolicyConfig,
    WebSearchTool,
    _check_domain_policy,
    _check_url_safe,
)


# --- SSRF guard (pure stdlib, no network/deps) -------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "gopher://x"])
def test_check_url_safe_rejects_non_http(url: str) -> None:
    with pytest.raises(WebError) as exc:
        _check_url_safe(url)
    assert exc.value.error_type == "BadURL"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1:8000",
        "http://10.0.0.5",
        "http://192.168.1.1",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
    ],
)
def test_check_url_safe_rejects_internal(url: str) -> None:
    with pytest.raises(WebError) as exc:
        _check_url_safe(url)
    assert exc.value.error_type == "SSRF"


def test_check_url_safe_allows_public_ip_literal() -> None:
    # A public IP literal resolves to itself offline — no network call needed.
    _check_url_safe("http://8.8.8.8/")  # should not raise


# --- web_fetch (network seam monkeypatched) ----------------------------------


async def test_web_fetch_converts_html(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: ("https://ex.com/final", "text/html; charset=utf-8", "<h1>Hi</h1>"))
    monkeypatch.setattr(web, "_html_to_markdown", lambda html: "# Hi")
    result = await WebFetchTool().run({"url": "https://ex.com"})
    assert result.ok
    assert "# Hi" in result.content
    assert result.metadata["final_url"] == "https://ex.com/final"


async def test_web_fetch_plain_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: ("https://ex.com/x.txt", "text/plain", "raw body"))
    result = await WebFetchTool().run({"url": "https://ex.com/x.txt"})
    assert result.ok
    assert "raw body" in result.content


async def test_web_fetch_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: ("https://ex.com", "text/plain", "x" * 100))
    result = await WebFetchTool().run({"url": "https://ex.com", "max_chars": 10})
    assert result.metadata["truncated"] is True
    assert "[... truncated ...]" in result.content


async def test_web_fetch_rejects_internal_without_network() -> None:
    # No monkeypatch: the SSRF pre-check fires before any httpx import.
    result = await WebFetchTool().run({"url": "http://localhost:9000"})
    assert not result.ok
    assert result.metadata["error_type"] == "SSRF"


def test_web_fetch_is_read_risk() -> None:
    assert WebFetchTool().risk is ToolRisk.READ


# --- web_search (backend selection monkeypatched) ----------------------------


async def test_web_search_defaults_to_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        web, "_search_ddgs",
        lambda q, n: [{"title": "T", "url": "https://e.com", "snippet": "S"}],
    )
    result = await WebSearchTool().run({"query": "python"})
    assert result.ok
    assert result.metadata["backend"] == "ddgs"
    assert "https://e.com" in result.content


async def test_web_search_prefers_brave_when_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    called: list = []
    monkeypatch.setattr(web, "_search_brave", lambda q, n, key: called.append(key) or [])
    monkeypatch.setattr(web, "_search_ddgs", lambda q, n: pytest.fail("ddgs should not be used"))
    result = await WebSearchTool().run({"query": "x"})
    assert result.metadata["backend"] == "brave"
    assert called == ["k"]


async def test_web_search_rejects_empty_query() -> None:
    result = await WebSearchTool().run({"query": "  "})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


def test_web_search_is_read_risk() -> None:
    assert WebSearchTool().risk is ToolRisk.READ


# --- outbound domain policy (D10 / S4) ----------------------------------------


def _fetch_tool(policy: WebPolicyConfig, unattended: bool) -> WebFetchTool:
    tool = WebFetchTool()
    tool.bind_web_policy(policy, unattended=unattended)
    return tool


async def test_blocked_domain_refused_in_any_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: pytest.fail("must not fetch"))
    policy = WebPolicyConfig(blocked_domains=["evil.example"])
    for unattended in (False, True):
        result = await _fetch_tool(policy, unattended).run({"url": "https://sub.evil.example/x"})
        assert not result.ok
        assert result.metadata["error_type"] == "BlockedDomain"


async def test_unattended_requires_allowlisted_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: pytest.fail("must not fetch"))
    result = await _fetch_tool(WebPolicyConfig(), unattended=True).run({"url": "https://anything.example"})
    assert not result.ok
    assert result.metadata["error_type"] == "DomainNotAllowed"
    assert "allowed_domains" in result.content  # actionable


async def test_unattended_allows_allowlisted_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: ("https://ok.example", "text/plain", "body"))
    policy = WebPolicyConfig(allowed_domains=["ok.example"])
    result = await _fetch_tool(policy, unattended=True).run({"url": "https://ok.example"})
    assert result.ok


async def test_attended_mode_stays_open_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Degradation check: no [web] config + attended mode = the pre-policy behavior.
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url, **kw: ("https://any.example", "text/plain", "body"))
    result = await WebFetchTool().run({"url": "https://any.example"})
    assert result.ok


def test_redirect_hop_check_covers_domain_policy() -> None:
    # _fetch_url runs hop_check on every hop; the policy raising there must propagate.
    policy = WebPolicyConfig(blocked_domains=["evil.example"])
    with pytest.raises(WebError) as exc:
        _check_domain_policy("https://evil.example/leak", policy, unattended=False)
    assert exc.value.error_type == "BlockedDomain"


async def test_web_search_checks_backend_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web, "_search_ddgs", lambda q, n: pytest.fail("must not search"))
    tool = WebSearchTool()
    tool.bind_web_policy(WebPolicyConfig(), unattended=True)
    result = await tool.run({"query": "secret data"})
    assert not result.ok
    assert result.metadata["error_type"] == "DomainNotAllowed"

    allowed = WebSearchTool()
    allowed.bind_web_policy(WebPolicyConfig(allowed_domains=["duckduckgo.com"]), unattended=True)
    monkeypatch.setattr(web, "_search_ddgs", lambda q, n: [{"title": "T", "url": "u", "snippet": "s"}])
    result = await allowed.run({"query": "secret data"})
    assert result.ok

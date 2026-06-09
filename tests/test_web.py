import pytest

from agent_core.models import ToolRisk
from agent_core.tools import web
from agent_core.tools.web import WebError, WebFetchTool, WebSearchTool, _check_url_safe


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


def test_web_fetch_converts_html(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url: ("https://ex.com/final", "text/html; charset=utf-8", "<h1>Hi</h1>"))
    monkeypatch.setattr(web, "_html_to_markdown", lambda html: "# Hi")
    result = WebFetchTool().run({"url": "https://ex.com"})
    assert result.ok
    assert "# Hi" in result.content
    assert result.metadata["final_url"] == "https://ex.com/final"


def test_web_fetch_plain_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url: ("https://ex.com/x.txt", "text/plain", "raw body"))
    result = WebFetchTool().run({"url": "https://ex.com/x.txt"})
    assert result.ok
    assert "raw body" in result.content


def test_web_fetch_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "_check_url_safe", lambda url: None)
    monkeypatch.setattr(web, "_fetch_url", lambda url: ("https://ex.com", "text/plain", "x" * 100))
    result = WebFetchTool().run({"url": "https://ex.com", "max_chars": 10})
    assert result.metadata["truncated"] is True
    assert "[... truncated ...]" in result.content


def test_web_fetch_rejects_internal_without_network() -> None:
    # No monkeypatch: the SSRF pre-check fires before any httpx import.
    result = WebFetchTool().run({"url": "http://localhost:9000"})
    assert not result.ok
    assert result.metadata["error_type"] == "SSRF"


def test_web_fetch_is_read_risk() -> None:
    assert WebFetchTool().risk is ToolRisk.READ


# --- web_search (backend selection monkeypatched) ----------------------------


def test_web_search_defaults_to_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        web, "_search_ddgs",
        lambda q, n: [{"title": "T", "url": "https://e.com", "snippet": "S"}],
    )
    result = WebSearchTool().run({"query": "python"})
    assert result.ok
    assert result.metadata["backend"] == "ddgs"
    assert "https://e.com" in result.content


def test_web_search_prefers_brave_when_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    called: list = []
    monkeypatch.setattr(web, "_search_brave", lambda q, n, key: called.append(key) or [])
    monkeypatch.setattr(web, "_search_ddgs", lambda q, n: pytest.fail("ddgs should not be used"))
    result = WebSearchTool().run({"query": "x"})
    assert result.metadata["backend"] == "brave"
    assert called == ["k"]


def test_web_search_rejects_empty_query() -> None:
    result = WebSearchTool().run({"query": "  "})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


def test_web_search_is_read_risk() -> None:
    assert WebSearchTool().risk is ToolRisk.READ

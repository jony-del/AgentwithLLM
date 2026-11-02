from types import SimpleNamespace

import pytest

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.lsp_evidence import line_range, query_evidence
from agent_core.codeintel.models import ChangeSet, SearchRequest, StaleEvidence
from agent_core.codeintel.modules import package_name
from agent_core.permission_rules import RuleSet
from tests.test_codeintel import built


def session(root):
    return SimpleNamespace(workspace=root, codeintel_config=CodeIntelConfig(), code_permission_rules=None,
                           code_working_set=None, code_read_versions={}, task_run=None)


def loc(path, line=0, column=0):
    return {"uri": path.as_uri(), "range": {"start": {"line": line, "character": column},
                                             "end": {"line": line, "character": column + 1}}}


async def test_lsp_evidence_filters_denied_and_foreign_and_never_certifies_edit(tmp_path):
    (tmp_path / "a.py").write_text("target()\n")
    (tmp_path / "secret.py").write_text("secret\n")
    s = session(tmp_path)
    s.code_permission_rules = lambda: RuleSet.from_lists(deny=["read_text_file(secret.py)"])
    class Manager:
        async def request(self, *args, **kwargs):
            return [loc(tmp_path / "a.py"), loc(tmp_path / "secret.py"), loc(tmp_path.parent / "outside.py")]
    page = await query_evidence(s, Manager(), "definition", path="a.py")
    assert [h.path for h in page.hits] == ["a.py"]
    assert page.hits[0].precision == "lsp_reported_location"
    assert not page.coverage.complete and not s.code_read_versions
    assert "unavailable_or_invalid_lsp_location" in page.coverage.reasons


async def test_lsp_rejects_source_moving_during_request(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("one\n")
    class Manager:
        async def request(self, *args, **kwargs):
            source.write_text("two\n")
            return [loc(source)]
    with pytest.raises(StaleEvidence):
        await query_evidence(session(tmp_path), Manager(), "definition", path="a.py")


def test_lsp_utf16_and_exclusive_end():
    assert line_range("abc\ndef", [(0, 1), (1, 0)]) == (1, 1)
    assert line_range("😀x", [(0, 2), (0, 3)]) == (1, 1)
    with pytest.raises(UnicodeError):
        line_range("😀x", [(0, 1), (0, 2)])


@pytest.mark.parametrize("path,text,name", [
    ("pyproject.toml", '[project]\nname="root"', "root"),
    ("package.json", '{"name":"js"}', "js"),
    ("Cargo.toml", '[package]\nname="rust"', "rust"),
    ("go.mod", 'module example.org/lib\n', "example.org/lib"),
    ("package.json", '[]', None), ("pyproject.toml", 'broken[', None),
])
def test_manifests(path, text, name):
    assert package_name(path, text) == name


async def test_nested_modules_update_remove_and_fallback(tmp_path):
    root = tmp_path / "repo"
    nested = root / "packages" / "worker"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="root"')
    manifest = nested / "package.json"
    manifest.write_text('{"name":"worker"}')
    (nested / "a.py").write_text("def target(): return 1\n")
    service = await built(tmp_path)
    async def owner():
        return (await service.search(SearchRequest("target", kind="symbol"))).hits[0].module
    try:
        assert await owner() == "packages/worker"
        assert (await service.read_region("packages/worker/a.py")).module == "packages/worker"
        manifest.unlink()
        await service.publish_changes(ChangeSet("deleted-boundary", ("packages/worker/package.json",)))
        await service.ensure_index()
        assert await owner() == "."
        (root / "pyproject.toml").write_text("invalid[")
        await service.publish_changes(ChangeSet("invalid-boundary", ("pyproject.toml",)))
        await service.ensure_index()
        assert await owner() == "packages"
    finally:
        await service.close()


async def test_candidate_identity_ambiguity_and_staleness(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "a.py"
    source.write_text("def target():\n    return 1\ndef other():\n    return 1\n")
    service = await built(tmp_path)
    try:
        hit = (await service.search(SearchRequest("target", kind="symbol"))).hits[0]
        result = await service.validate_candidate("a.py", hit.version.to_dict(), "def target():", hit.symbol)
        assert result["status"] == "validated_anchor"
        with pytest.raises(ValueError, match="exactly once"):
            await service.validate_candidate("a.py", hit.version.to_dict(), "return 1")
        with pytest.raises(ValueError, match="supplied current symbol"):
            await service.validate_candidate("a.py", hit.version.to_dict(), "def other():", hit.symbol)
        source.write_text("def target(): return 2\n")
        with pytest.raises(StaleEvidence):
            await service.validate_candidate("a.py", hit.version.to_dict(), "def target():")
    finally:
        await service.close()

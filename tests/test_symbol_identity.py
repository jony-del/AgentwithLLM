import pytest

from agent_core.codeintel.models import FileVersion, SearchRequest, symbol_reference
from agent_core.codeintel.store import CodeStore
from agent_core.codeintel.service import CodeIntelligenceService
from agent_core.codeintel.config import CodeIntelConfig


def test_definition_identity_separates_scope_and_version():
    def ref(path="a.py", digest="a" * 64, name="Client.send", parser="v1"):
        return symbol_reference(FileVersion("wt", path, digest), name, "function", 2, 5, parser)
    first = ref()
    assert first == ref() and first["container"] == "Client"
    assert len({first["id"], ref(path="b.py")["id"], ref(digest="b" * 64)["id"],
                ref(name="Other.send")["id"], ref(parser="v2")["id"]}) == 5


async def test_parser_upgrade_requeues_legacy_and_rejects_old_reader(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("class Client:\n def send(self): return 1\n")
    database = tmp_path / "index.db"
    service = CodeIntelligenceService(root, CodeIntelConfig(maintain=False, watch=False), database=database)
    try:
        await service.ensure_index()
        page = await service.search(SearchRequest("send", kind="symbol"))
        assert page.hits[0].symbol["qualified_name"] == "Client.send"
        old_id = page.hits[0].symbol["id"]
        with service.store.connect() as db:
            db.execute("DELETE FROM meta WHERE key='parser_fingerprint'")
        replacement = CodeStore(root, database)
        replacement.initialize()
        try:
            assert replacement.status()["pending"] == 1
            await service.ensure_index()
            assert (await service.search(SearchRequest("send", kind="symbol"))).hits[0].symbol["id"] == old_id
            monkeypatch.setattr("agent_core.codeintel.backends.PARSER_FINGERPRINT", "test-next-parser")
            upgraded = CodeStore(root, database)
            upgraded.initialize()
            try:
                page = await service.search(SearchRequest("send", kind="symbol"))
                assert not page.hits and any("parser_changed" in r for r in page.coverage.reasons)
                with pytest.raises(RuntimeError, match="parser changed"):
                    await service.ensure_index()
            finally:
                upgraded.close()
        finally:
            replacement.close()
    finally:
        await service.close()

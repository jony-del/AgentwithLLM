"""Static workspace/local-package dependency graph from manifests (S6)."""
from __future__ import annotations


from agent_core.codeintel.models import ChangeSet, SearchRequest
from tests.test_codeintel import built


async def graph(tmp_path, files: dict[str, str], **config):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return await built(tmp_path, **config)


async def test_npm_workspaces_and_file_and_name_refs(tmp_path):
    service = await graph(tmp_path, {
        "package.json": '{"name":"root","workspaces":["packages/*"]}',
        "packages/a/package.json": '{"name":"pkg-a","dependencies":{"pkg-b":"1.0.0","react":"18.0.0"}}',
        "packages/b/package.json": '{"name":"pkg-b"}',
        "tools/c/package.json": '{"name":"pkg-c","dependencies":{"pkg-b":"file:../../packages/b"}}',
    })
    try:
        page = await service.search(SearchRequest("pkg-b", kind="local_dependency"))
        by_source = {(h.source_symbol, h.path): h.precision for h in page.hits}
        assert ("pkg-a", "packages/a/package.json") in by_source
        assert by_source[("pkg-a", "packages/a/package.json")] == "manifest_local:local_name"
        assert by_source[("pkg-c", "tools/c/package.json")] == "manifest_local:local_path"
        # The workspace member glob matched indexed manifests without a filesystem walk.
        members = await service.search(SearchRequest("root", kind="local_dependency", direction="outgoing"))
        member_targets = {h.target_symbol for h in members.hits}
        assert "packages/a" in member_targets and "packages/b" in member_targets
        # react is an external registry dependency: it must not appear as a local edge.
        assert not (await service.search(SearchRequest("react", kind="local_dependency"))).hits
        assert (await service.search(SearchRequest("react", kind="package_dependency"))).hits
    finally:
        await service.close()


async def test_cargo_path_dependencies_and_workspace_members(tmp_path):
    service = await graph(tmp_path, {
        "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
        "crates/core/Cargo.toml": '[package]\nname = "core"\n',
        "crates/cli/Cargo.toml": '[package]\nname = "cli"\n[dependencies]\ncore = { path = "../core" }\nserde = "1"\n',
    })
    try:
        page = await service.search(SearchRequest("core", kind="local_dependency"))
        assert [(h.source_symbol, h.precision) for h in page.hits] == [("cli", "manifest_local:local_path")]
        assert not (await service.search(SearchRequest("serde", kind="local_dependency"))).hits
    finally:
        await service.close()


async def test_go_replace_and_module_name_links(tmp_path):
    service = await graph(tmp_path, {
        "go.mod": "module example.com/main\n\ngo 1.22\n\nrequire example.com/lib v0.0.0\n\nreplace example.com/lib => ./vendor/lib\n",
        "vendor/lib/go.mod": "module example.com/lib\n",
    })
    try:
        page = await service.search(SearchRequest("example.com/lib", kind="local_dependency"))
        assert [(h.precision, h.target_symbol) for h in page.hits] == [("manifest_local:local_path", "vendor/lib")]
    finally:
        await service.close()


async def test_python_scoped_local_sources_only(tmp_path):
    service = await graph(tmp_path, {
        "pyproject.toml": '[project]\nname = "root"\ndependencies = ["requests>=2"]\n[tool.uv.sources]\nhelper = { path = "packages/helper" }\n',
        "packages/helper/pyproject.toml": '[project]\nname = "helper"\n',
    })
    try:
        page = await service.search(SearchRequest("helper", kind="local_dependency"))
        assert [(h.source_symbol, h.precision) for h in page.hits] == [("root", "manifest_local:local_path")]
        assert not (await service.search(SearchRequest("requests", kind="local_dependency"))).hits
    finally:
        await service.close()


async def test_ambiguous_names_escaping_paths_and_missing_targets_are_explained(tmp_path):
    service = await graph(tmp_path, {
        "packages/a/package.json": '{"name":"dup"}',
        "packages/b/package.json": '{"name":"dup"}',
        "packages/c/package.json": '{"name":"c","dependencies":{"dup":"1.0.0","gone":"file:../nowhere"}}',
        "evil/package.json": '{"name":"evil","dependencies":{"x":"file:../../outside"}}',
    })
    try:
        page = await service.search(SearchRequest("dup", kind="local_dependency"))
        assert [h.precision for h in page.hits] == ["manifest_unresolved:ambiguous_local_name:2"]
        missing = await service.search(SearchRequest("gone", kind="local_dependency"))
        assert [h.precision for h in missing.hits] == ["manifest_unresolved:no_indexed_package_at_path"]
        escaped = await service.search(SearchRequest("x", kind="local_dependency"))
        assert [h.precision for h in escaped.hits] == ["manifest_unresolved:path_escapes_workspace"]
    finally:
        await service.close()


async def test_cycle_records_both_directions_and_bounded_expansion_terminates(tmp_path):
    service = await graph(tmp_path, {
        "a/package.json": '{"name":"a","dependencies":{"b":"file:../b"}}',
        "b/package.json": '{"name":"b","dependencies":{"a":"file:../a"}}',
    }, max_depth=2)
    try:
        page = await service.expand_relations(["a"], relation="local_dependency", depth=4)
        assert "max_depth" in page.coverage.reasons
        names = {h.source_symbol for h in page.hits}
        assert names == {"a", "b"}  # visited set terminates the cycle
        outgoing = await service.search(SearchRequest("a", kind="local_dependency", direction="outgoing"))
        assert [h.target_symbol for h in outgoing.hits] == ["b"] * len(outgoing.hits)
    finally:
        await service.close()


async def test_links_follow_manifest_edits_deletions_and_late_roots(tmp_path):
    service = await graph(tmp_path, {
        "packages/a/package.json": '{"name":"pkg-a","dependencies":{"pkg-b":"1.0.0"}}',
    })
    try:
        # Target manifest does not exist yet: external-looking name, no local link.
        assert not (await service.search(SearchRequest("pkg-b", kind="local_dependency"))).hits
        root = tmp_path / "repo"
        (root / "packages" / "b").mkdir(parents=True)
        (root / "packages" / "b" / "package.json").write_text('{"name":"pkg-b"}', encoding="utf-8")
        await service.publish_changes(ChangeSet("add-b", ("packages/b/package.json",), "test"))
        for _ in range(50):
            status = await service.ensure_index()
            if status["catalog_complete"] and not status["pending"]:
                break
        page = await service.search(SearchRequest("pkg-b", kind="local_dependency"))
        assert [h.precision for h in page.hits] == ["manifest_local:local_name"]
        # Deleting the target degrades the link to an explained unresolved row.
        (root / "packages" / "b" / "package.json").unlink()
        await service.publish_changes(ChangeSet("del-b", ("packages/b/package.json",), "test"))
        for _ in range(50):
            status = await service.ensure_index()
            if status["catalog_complete"] and not status["pending"]:
                break
        page = await service.search(SearchRequest("pkg-b", kind="local_dependency"))
        assert [h.precision for h in page.hits] == ["manifest_unresolved:package_removed"]
    finally:
        await service.close()

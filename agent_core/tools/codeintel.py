from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from agent_core.codeintel.models import SearchRequest
from agent_core.codeintel.runtime import get_service, remember_version
from agent_core.models import ToolResult, ToolRisk
from agent_core.permission_safety import ordinary_read_permission
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool, WorkspacePathMixin, current_execution_scope
from agent_core.tools.catalog import builtin_tool


def path_allowed(session: Any, path: str, tool: str = "code_search") -> bool:
    rules = session.code_permission_rules() if callable(session.code_permission_rules) else None
    if rules is None:
        return True
    arguments = ({"path": path}, {"path": str((session.workspace / path).resolve())})
    return not any(rules.deny_match(name, argument) or rules.ask_match(name, argument)
                   for argument in arguments for name in {tool, "read_text_file", "search_text"})


class _CodeTool(SessionAwareMixin, WorkspacePathMixin, Tool):
    risk = ToolRisk.READ
    deferred = True
    safely_cancellable = True

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        path = (self.session.workspace / str(arguments.get("path", "."))).resolve()
        return ConcurrencySpec((ResourceLock("fs", str(path), "read", subtree=True, requires_success=True),))

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        return ordinary_read_permission(self.name, arguments, context)


@builtin_tool
class CodeSearchTool(_CodeTool):
    name = "code_search"
    description = ("Search versioned code facts: paths, text, Python symbols and syntactic references. "
                   "Starts in task modules unless path/modules or expand=true is provided. "
                   "Reports incomplete coverage; use polaris code build to construct the index.")
    input_schema = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1},
        "kind": {"enum": ["auto", "path", "text", "symbol", "references", "calls", "imports", "inherits", "package_dependency", "build_dependency"]},
        "path": {"type": "string"}, "modules": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"}, "expand": {"type": "boolean"},
        "regex": {"type": "boolean"}, "ignore_case": {"type": "boolean"},
        "direction": {"enum": ["incoming", "outgoing"]},
        "limit": {"type": "integer", "minimum": 1}, "cursor": {"type": "string"}}, "required": ["query"]}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        service = await get_service(self.session)
        assert self.session.code_working_set is not None
        modules = tuple(str(m) for m in arguments.get("modules", []))
        implicit_scope = "modules" not in arguments and not arguments.get("expand") and not arguments.get("path")
        if implicit_scope:
            modules = self.session.code_working_set.modules()
        page = await service.search(SearchRequest(
            query=str(arguments["query"]), kind=str(arguments.get("kind", "auto")),
            path=str(arguments.get("path", ".")), modules=modules,
            language=arguments.get("language"), regex=bool(arguments.get("regex")),
            ignore_case=bool(arguments.get("ignore_case")), limit=int(arguments.get("limit", 100)),
            cursor=arguments.get("cursor"), direction=str(arguments.get("direction", "incoming")),
            expand_scope=implicit_scope), scope=current_execution_scope(),
            allowed=lambda path: path_allowed(self.session, path, self.name))
        for hit in page.hits:
            self.session.code_working_set.add(hit)
            remember_version(self.session, hit.path, hit.version.to_dict())
        body = page.to_dict()
        if modules:
            body["scope_hint"] = "local first; coverage.checked identifies any expanded ranges" if implicit_scope else "explicit module scope"
        return ToolResult(self.name, json.dumps(body, ensure_ascii=False), metadata={"codeintel": body})


@builtin_tool
class CodeRelationsTool(CodeSearchTool):
    name = "code_relations"
    description = "Find syntactic references/calls/imports/inheritance with versioned source evidence and explicit coverage."
    input_schema = {**CodeSearchTool.input_schema, "properties": {
        **CodeSearchTool.input_schema["properties"],
        "kind": {"enum": ["references", "calls", "imports", "inherits", "package_dependency", "build_dependency"]},
        "depth": {"type": "integer", "minimum": 1}}}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if int(arguments.get("depth", 1)) > 1:
            service = await get_service(self.session)
            page = await service.expand_relations([str(arguments["query"])], relation=str(arguments.get("kind", "references")),
                depth=int(arguments["depth"]), direction=str(arguments.get("direction", "incoming")),
                path=str(arguments.get("path", ".")), modules=tuple(arguments.get("modules", [])),
                scope=current_execution_scope(), allowed=lambda path: path_allowed(self.session, path, self.name))
            assert self.session.code_working_set is not None
            for hit in page.hits:
                self.session.code_working_set.add(hit, reason="relation_expansion")
                remember_version(self.session, hit.path, hit.version.to_dict())
            return ToolResult(self.name, json.dumps(page.to_dict(), ensure_ascii=False), metadata={"codeintel": page.to_dict()})
        return await super().run({"kind": "references", **arguments})


@builtin_tool
class CodeContextTool(_CodeTool):
    name = "code_context"
    description = "Read a bounded current code region, optionally requiring the version returned by code_search."
    input_schema = {"type": "object", "properties": {
        "path": {"type": "string"}, "start": {"type": "integer", "minimum": 1},
        "limit": {"type": "integer", "minimum": 1}, "expected_version": {"type": "object"}}, "required": ["path"]}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments["path"])
        if not path_allowed(self.session, path, self.name):
            return ToolResult(self.name, "Code path is restricted by read policy.", ok=False)
        service = await get_service(self.session)
        assert self.session.code_working_set is not None
        hit = await service.read_region(path, int(arguments.get("start", 1)), int(arguments.get("limit", 200)),
            expected_version=arguments.get("expected_version"), scope=current_execution_scope())
        self.session.code_working_set.add(hit, reason="explicit_read")
        remember_version(self.session, hit.path, hit.version.to_dict())
        return ToolResult(self.name, json.dumps(asdict(hit), ensure_ascii=False),
                          metadata={"file_version": hit.version.to_dict()})

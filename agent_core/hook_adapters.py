"""Config-driven external hook adapters — the ``[[hooks.external]]`` loader.

Each :class:`~agent_core.hooks.ExternalHookSpec` is wrapped in an adapter that implements
the lifecycle hook Protocols, so an externally-declared hook folds into the same
``HookPipeline`` as the built-in programmatic ones (``react.py`` is untouched). Four
transports, mirroring the reference's settings.json hooks:

* ``command`` — spawn a subprocess, feed the projected ``HookContext`` JSON on stdin,
  read stdout JSON / exit code 2 for the block decision.
* ``http`` — POST the same JSON to a URL and parse the response.
* ``prompt`` — re-prompt the LLM (via the shared gated provider) for advisory context.
* ``agent`` — run a verifier sub-agent (via the session's depth-limited factory).

Invariants (aligned with the project's timeout / degrade discipline):

* A **single, non-stacked** timeout bounds every external call; a command that overruns is
  killed and awaited (no zombies).
* When the hook ITSELF fails (timeout, crash, network error), ``spec.fail_mode`` decides:
  ``"open"`` (default) **degrades to allow** — an empty ``HookOutcome`` plus a log, never
  raising into the run; ``"closed"`` converts the failure into a **block** decision, so a
  crashed security gate does not silently swing open. ``fail_mode`` is honored only by the
  transports that carry the block contract (``command`` / ``http``).
* The context handed across the boundary is a **bounded JSON projection** (recent messages
  only, content truncated), never the full live history.
* Only the compaction events honor ``matcher`` (matched against the ``trigger``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_core.hooks import ExternalHookSpec, HookContext, HookLimitsConfig, HookOutcome
from agent_core.models import Message
from agent_core.permission_rules import ParsedRule, RuleSet, parse_rule
from agent_core.process_tree import terminate_process_tree
from agent_core.providers.base import ProviderConfig
from agent_core.prompt_ingress import defang_reserved_tags

if TYPE_CHECKING:
    from agent_core.execution import ExecutionScope
    from agent_core.providers.base import LLMProvider
    from agent_core.storage import JSONLRunLogger

# Bounds on the JSON projection so an external hook never receives the whole transcript.
_MAX_PROJECTED_MESSAGES = 20
_MAX_PROJECTED_CONTENT = 2000
_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|password|authorization|api[-_]?key|cookie|credential)", re.IGNORECASE
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRUNCATION_MARKER = "...[truncated]"


def _truncate_utf8(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    marker = _TRUNCATION_MARKER.encode("utf-8")
    usable = max(0, limit - len(marker))
    return encoded[:usable].decode("utf-8", "ignore") + _TRUNCATION_MARKER, True


def _sanitize_projection_key(
    value: object, limits: HookLimitsConfig, counters: dict[str, int]
) -> str:
    """Apply the UTF-8, control-character, and framing rules to object keys."""

    cleaned = defang_reserved_tags(_CONTROL_CHARS.sub("�", str(value)))
    cleaned, truncated = _truncate_utf8(cleaned, limits.string_bytes)
    counters["keys_truncated"] += int(truncated)
    return cleaned


def _sanitize_projection(
    value: Any,
    limits: HookLimitsConfig,
    counters: dict[str, int],
    *,
    depth: int = 0,
    key: str = "",
) -> Any:
    if _SENSITIVE_KEY.search(key):
        counters["redacted"] += 1
        return "<redacted>"
    if depth >= limits.max_depth:
        counters["depth_truncated"] += 1
        return "<max-depth>"
    if isinstance(value, str):
        cleaned = defang_reserved_tags(_CONTROL_CHARS.sub("�", value))
        cleaned, truncated = _truncate_utf8(cleaned, limits.string_bytes)
        counters["strings_truncated"] += int(truncated)
        return cleaned
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        if len(items) > limits.max_items:
            counters["items_truncated"] += len(items) - limits.max_items
        for raw_key, item in items[: limits.max_items]:
            item_key = _sanitize_projection_key(raw_key, limits, counters)
            if item_key in result:
                # Truncation can make distinct hostile keys collide. Preserve the
                # first value rather than silently replacing sanitized data.
                counters["items_truncated"] += 1
                continue
            result[item_key] = _sanitize_projection(
                item, limits, counters, depth=depth + 1, key=str(raw_key)
            )
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > limits.max_items:
            counters["items_truncated"] += len(value) - limits.max_items
        return [
            _sanitize_projection(item, limits, counters, depth=depth + 1)
            for item in value[: limits.max_items]
        ]
    return _sanitize_projection(str(value), limits, counters, depth=depth, key=key)


def _longest_string_path(value: Any, path: tuple[Any, ...] = ()) -> tuple[tuple[Any, ...], int]:
    best = (path, len(value.encode("utf-8"))) if isinstance(value, str) else ((), -1)
    if isinstance(value, dict):
        for key, item in value.items():
            candidate = _longest_string_path(item, (*path, key))
            if candidate[1] > best[1]:
                best = candidate
    elif isinstance(value, list):
        for index, item in enumerate(value):
            candidate = _longest_string_path(item, (*path, index))
            if candidate[1] > best[1]:
                best = candidate
    return best


def _set_path(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def bounded_hook_payload(
    value: Any, limits: HookLimitsConfig | None = None
) -> tuple[Any, dict[str, int]]:
    limits = limits or HookLimitsConfig()
    counters = {
        "redacted": 0,
        "keys_truncated": 0,
        "strings_truncated": 0,
        "items_truncated": 0,
        "depth_truncated": 0,
        "total_truncated": 0,
    }
    projected = _sanitize_projection(value, limits, counters)
    while len(json.dumps(projected, ensure_ascii=False, default=str).encode("utf-8")) > limits.total_bytes:
        path, length = _longest_string_path(projected)
        if not path or length <= len(_TRUNCATION_MARKER.encode("utf-8")):
            if isinstance(projected, dict):
                removable = [key for key in reversed(projected) if key != "hook_event_name"]
                if not removable:
                    projected = {"error": "projection_too_large"}
                    break
                projected.pop(removable[0], None)
                counters["items_truncated"] += 1
                continue
            projected = "<projection-too-large>"
            break
        target = projected
        for part in path:
            target = target[part]
        replacement, _ = _truncate_utf8(target, max(16, length // 2))
        _set_path(projected, path, replacement)
        counters["total_truncated"] += 1
    return projected, counters

# Official hook events mapped to the HookPipeline collection they belong in.  Events for
# which Polaris does not yet have a host firing seam are retained in ``unhandled_hooks``
# rather than being silently discarded; this preserves the manifest faithfully while the
# two security-critical tool events are wired into ToolExecutor below.
LIFECYCLE_EVENT_ATTRS: dict[str, str] = {
    "UserPromptSubmit": "user_prompt_hooks",
    "PostSampling": "post_sampling_hooks",
    "PreToolUse": "external_pre_tool_hooks",
    "PostToolUse": "external_post_tool_hooks",
    "PreCompact": "pre_compact_hooks",
    "PostCompact": "post_compact_hooks",
    "Stop": "stop_hooks",
    # Observational events (C5) — an external hook may watch them; decisions are discarded.
    "SessionStart": "session_start_hooks",
    "SessionEnd": "session_end_hooks",
    "SubagentStart": "subagent_start_hooks",
    "SubagentStop": "subagent_stop_hooks",
    "PostToolUseFailure": "tool_failure_hooks",
    # Control-path event (R1): programmatic approval of an "ask" permission decision.
    "PermissionRequest": "permission_request_hooks",
    "Setup": "unhandled_hooks",
    "UserPromptExpansion": "unhandled_hooks",
    "PermissionDenied": "unhandled_hooks",
    "PostToolBatch": "unhandled_hooks",
    "Notification": "unhandled_hooks",
    "MessageDisplay": "unhandled_hooks",
    "TaskCreated": "unhandled_hooks",
    "TaskCompleted": "unhandled_hooks",
    "StopFailure": "unhandled_hooks",
    "TeammateIdle": "unhandled_hooks",
    "InstructionsLoaded": "unhandled_hooks",
    "ConfigChange": "unhandled_hooks",
    "CwdChanged": "unhandled_hooks",
    "DirectoryAdded": "unhandled_hooks",
    "FileChanged": "unhandled_hooks",
    "WorktreeCreate": "unhandled_hooks",
    "WorktreeRemove": "unhandled_hooks",
    "Elicitation": "unhandled_hooks",
    "ElicitationResult": "unhandled_hooks",
}


def project_hook_input(
    ctx: HookContext,
    *,
    max_messages: int = _MAX_PROJECTED_MESSAGES,
    max_content_chars: int = _MAX_PROJECTED_CONTENT,
    limits: HookLimitsConfig | None = None,
) -> dict[str, Any]:
    """Project a ``HookContext`` to the stable JSON an external hook receives.

    Reference-shaped (``hook_event_name`` + event-specific fields) plus a bounded tail of
    recent messages (role + truncated content). This is the one place the in-process "live
    object" world is reduced to a serializable snapshot — keep it small and stable.
    """
    data: dict[str, Any] = {
        "hook_event_name": ctx.event.value,
        "session_id": ctx.session_id,
        "stop_hook_active": ctx.stop_hook_active,
    }
    if ctx.prompt is not None:
        data["prompt"] = ctx.prompt
    if ctx.trigger is not None:
        data["trigger"] = ctx.trigger
    if ctx.summary is not None:
        data["summary"] = ctx.summary
    if ctx.last_assistant_message is not None:
        data["last_assistant_message"] = ctx.last_assistant_message
    if ctx.detail is not None:
        # Already a small, JSON-safe payload built at the firing seam (see HookContext).
        data["detail"] = ctx.detail
    limits = limits or HookLimitsConfig(
        max_messages=max_messages, message_chars=max_content_chars
    )
    max_messages = min(max_messages, limits.max_messages)
    max_content_chars = min(max_content_chars, limits.message_chars)
    tail = ctx.messages[-max_messages:] if max_messages > 0 else []
    projected = []
    for message in tail:
        content = message.content or ""
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "…"
        projected.append({"role": message.role, "content": content})
    data["messages"] = projected
    bounded, counters = bounded_hook_payload(data, limits)
    if isinstance(bounded, dict):
        bounded["_projection"] = counters
        # The counters themselves may cross a very small custom test budget.
        if len(json.dumps(bounded, ensure_ascii=False).encode("utf-8")) > limits.total_bytes:
            bounded.pop("_projection", None)
        return bounded
    return {"hook_event_name": ctx.event.value, "error": "projection_too_large"}


class HookFailedError(RuntimeError):
    """The external hook itself failed (timeout / transport error / crash).

    Raised by transport ``_invoke`` implementations so :meth:`_ExternalHookAdapter._run`
    can apply the spec's ``fail_mode`` uniformly (open → allow, closed → block).
    """


def outcome_from_output(
    stdout: str, returncode: int, limits: HookLimitsConfig | None = None
) -> HookOutcome:
    """Map a command/http hook's textual output + exit code to a :class:`HookOutcome`.

    Block when the exit code is 2 (reference convention) or the JSON says so
    (``continue: false`` / ``decision: "block"``). ``hookSpecificOutput.additionalContext``
    (or top-level ``additionalContext``) is injected; ``stopReason`` / ``reason`` is surfaced.
    On the PermissionRequest event, ``decision: "allow"`` approves the asked-about call
    and ``decision: "deny"`` / ``"block"`` (or exit code 2) refuses it. Non-JSON stdout
    is ignored except for the exit-code signal.
    """
    block = returncode == 2
    additional: str | None = None
    reason: str | None = None
    decision: str | None = "deny" if block else None
    text = stdout.strip()
    if text:
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            raw_decision = data.get("decision")
            if data.get("continue") is False or raw_decision in {"block", "deny"}:
                block = True
                decision = "deny"
            elif raw_decision == "allow":
                decision = "allow"
            reason = data.get("stopReason") or data.get("reason")
            spec_out = data.get("hookSpecificOutput")
            if isinstance(spec_out, dict):
                additional = spec_out.get("additionalContext")
                permission = spec_out.get("permissionDecision")
                if permission in {"deny", "block"}:
                    block = True
                    decision = "deny"
                elif permission in {"allow", "ask"}:
                    decision = str(permission)
                if spec_out.get("permissionDecisionReason") and not reason:
                    reason = str(spec_out["permissionDecisionReason"])
                updated_input = spec_out.get("updatedInput")
                updated_output = spec_out.get("updatedMCPToolOutput")
                metadata = {
                    key: value
                    for key, value in {
                        "updated_input": updated_input,
                        "updated_output": updated_output,
                    }.items()
                    if value is not None
                }
            else:
                metadata = {}
            if additional is None:
                additional = data.get("additionalContext")
        else:
            metadata = {}
    else:
        metadata = {}
    output_limits = replace(
        limits or HookLimitsConfig(),
        total_bytes=(limits.output_bytes if limits is not None else HookLimitsConfig().output_bytes),
        string_bytes=min(
            (limits.string_bytes if limits is not None else HookLimitsConfig().string_bytes),
            (limits.output_bytes if limits is not None else HookLimitsConfig().output_bytes),
        ),
    )
    bounded, counters = bounded_hook_payload(
        {
            "additional_context": additional,
            "reason": reason,
            "metadata": metadata,
        },
        output_limits,
    )
    bounded = bounded if isinstance(bounded, dict) else {}
    metadata_out = bounded.get("metadata")
    if not isinstance(metadata_out, dict):
        metadata_out = {}
    metadata_out["projection"] = counters
    if len(
        json.dumps(bounded, ensure_ascii=False, default=str).encode("utf-8")
    ) > output_limits.total_bytes:
        metadata_out.pop("projection", None)
    return HookOutcome(
        block=block,
        additional_context=bounded.get("additional_context") if isinstance(bounded.get("additional_context"), str) else None,
        reason=bounded.get("reason") if isinstance(bounded.get("reason"), str) else None,
        decision=decision,
        metadata=metadata_out,
    )


class _ExternalHookAdapter:
    """Base adapter: implements every lifecycle Protocol, routes to ``_invoke``.

    An instance is appended to exactly one pipeline list (its ``spec.event``), so only the
    matching Protocol method is ever called; implementing them all keeps the adapter a
    structural match for whichever list it lands in. ``_run`` applies the matcher gate and
    the degrade-to-allow guard shared by all transports.
    """

    def __init__(
        self,
        spec: ExternalHookSpec,
        logger: "JSONLRunLogger",
        limits: HookLimitsConfig | None = None,
    ) -> None:
        self.spec = spec
        self.logger = logger
        self.event = spec.event
        self.limits = limits or HookLimitsConfig()

    def _matches(self, ctx: HookContext) -> bool:
        if self.spec.matcher is not None and ctx.trigger is not None:
            patterns = [part.strip() for part in self.spec.matcher.split("|")]
            if not any(_tool_names_match(part, ctx.trigger) for part in patterns):
                return False
        if self.spec.condition is None:
            return True
        # Claude's ``if`` is valid only on tool events and uses permission-rule
        # syntax (for example ``Bash(git *)``).  Reuse Polaris' command-aware
        # matcher after rebasing the Claude tool alias to the actual registered
        # tool name.  A malformed condition fails closed by not spawning the hook.
        if ctx.trigger is None or not isinstance(ctx.detail, dict):
            return False
        arguments = ctx.detail.get("tool_input")
        if not isinstance(arguments, dict):
            return False
        parsed = parse_rule(self.spec.condition)
        if parsed is None or not _tool_names_match(parsed.tool_name, ctx.trigger):
            return False
        # RuleSet's shell-aware matcher uses Claude's canonical lowercase tool
        # identifiers (``bash``/``powershell``), so keep that canonical name after
        # checking it aliases the actual Polaris tool.
        rule_name = parsed.tool_name.casefold()
        rule = ParsedRule(rule_name, parsed.content, parsed.source)
        return RuleSet(allow=[rule]).allow_matches(rule_name, arguments)

    async def _invoke(self, ctx: HookContext) -> HookOutcome:  # pragma: no cover - overridden
        raise NotImplementedError

    async def _run(self, ctx: HookContext) -> HookOutcome:
        if not self._matches(ctx):
            return HookOutcome()
        try:
            return await self._invoke(ctx)
        except HookFailedError as exc:
            await self._log("hook_failed", str(exc))
            return self._failure_outcome(str(exc))
        except Exception as exc:  # noqa: BLE001 - never raise into the run.
            await self._log("exception", f"{type(exc).__name__}: {exc}")
            return self._failure_outcome(f"{type(exc).__name__}: {exc}")

    def _failure_outcome(self, detail: str) -> HookOutcome:
        """Apply ``fail_mode`` to a hook-side failure.

        Only the block-contract transports (command/http) may fail closed; the advisory
        transports (prompt/agent) always degrade to allow, matching their "never block"
        contract.
        """
        if self.spec.fail_mode == "closed" and self.spec.type in {"command", "http"}:
            return HookOutcome(
                block=True,
                reason=f"external {self.spec.type} hook failed and fail_mode=closed: {detail}",
            )
        return HookOutcome()

    async def _log(self, status: str, detail: str = "") -> None:
        try:
            await self.logger.write(
                "hook_external",
                {"hook": self.event, "type": self.spec.type, "status": status, "detail": detail[:300]},
            )
        except Exception:  # noqa: BLE001
            pass

    # --- lifecycle Protocol methods --------------------------------------------
    async def on_user_prompt(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)

    async def before_compact(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)

    async def after_compact(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)

    async def on_stop(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)

    async def after_sampling(self, ctx: HookContext) -> None:
        # PostSampling is observational; run the side effect, discard any decision.
        await self._run(ctx)

    # The C5 observational events: side effect only, any decision is discarded.
    async def on_session_start(self, ctx: HookContext) -> None:
        await self._run(ctx)

    async def on_session_end(self, ctx: HookContext) -> None:
        await self._run(ctx)

    async def on_subagent_start(self, ctx: HookContext) -> None:
        await self._run(ctx)

    async def on_subagent_stop(self, ctx: HookContext) -> None:
        await self._run(ctx)

    async def on_tool_failure(self, ctx: HookContext) -> None:
        await self._run(ctx)

    async def on_permission_request(self, ctx: HookContext) -> HookOutcome:
        # Control path: a block (from output, or from fail_mode=closed on a hook
        # failure) IS a deny; only an explicit {"decision": "allow"} approves. The
        # advisory transports (prompt/agent) can never block, so at most they deny
        # nothing and allow nothing — their replies are informational only.
        outcome = await self._run(ctx)
        if outcome.block and outcome.decision is None:
            outcome.decision = "deny"
        return outcome

    async def on_pre_tool(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)

    async def on_post_tool(self, ctx: HookContext) -> HookOutcome:
        return await self._run(ctx)


_TOOL_ALIASES: dict[str, frozenset[str]] = {
    "bash": frozenset({"bash", "shell", "shell_command"}),
    "powershell": frozenset({"powershell", "powershell_command"}),
    "read": frozenset({"read", "read_text_file"}),
    "write": frozenset({"write", "write_text_file"}),
    "edit": frozenset({"edit", "edit_file", "apply_patch"}),
    "webfetch": frozenset({"webfetch", "web_fetch"}),
    "websearch": frozenset({"websearch", "web_search"}),
    "agent": frozenset({"agent", "dispatch", "task"}),
}


def _tool_names_match(expected: str, actual: str) -> bool:
    left = expected.strip().casefold()
    right = actual.strip().casefold()
    if not left:
        return True
    if left == right:
        return True
    return any(left in aliases and right in aliases for aliases in _TOOL_ALIASES.values())


async def _bounded_communicate(
    proc: asyncio.subprocess.Process, payload: bytes, output_limit: int
) -> tuple[bytes, bytes]:
    if proc.stdin is not None:
        proc.stdin.write(payload)
        await proc.stdin.drain()
        proc.stdin.close()
    reads: list[asyncio.Task[bytes]] = []
    for stream in (proc.stdout, proc.stderr):
        if stream is None:
            reads.append(asyncio.create_task(asyncio.sleep(0, result=b"")))
        else:
            reads.append(asyncio.create_task(stream.read(output_limit + 1)))
    try:
        stdout, stderr = await asyncio.gather(*reads)
    except BaseException:
        # ``wait_for`` waits for cancellation cleanup. Kill the complete process
        # tree first so inherited pipe handles close and cancelled Proactor reads
        # cannot hold the timeout path open until a grandchild exits naturally.
        await terminate_process_tree(proc)
        raise
    finally:
        for task in reads:
            if not task.done():
                task.cancel()
        await asyncio.gather(*reads, return_exceptions=True)
    if len(stdout) > output_limit or len(stderr) > output_limit:
        raise ValueError("command hook output exceeded configured byte limit")
    await proc.wait()
    return stdout, stderr


class CommandHookAdapter(_ExternalHookAdapter):
    """Spawn a subprocess, feed projected JSON on stdin, read stdout/exit-code decision."""

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.command and not self.spec.command_argv:
            return HookOutcome()
        payload = json.dumps(project_hook_input(ctx, limits=self.limits)).encode("utf-8")
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        if self.spec.command_argv:
            proc = await asyncio.create_subprocess_exec(
                *self.spec.command_argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.spec.env or {})},
                **process_options,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                self.spec.command or "",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.spec.env or {})},
                **process_options,
            )
        try:
            communicate = _bounded_communicate(proc, payload, self.limits.output_bytes)
            if ctx.execution_scope is None:
                stdout, _stderr = await asyncio.wait_for(
                    communicate, timeout=self.spec.timeout
                )
            else:
                stdout, _stderr = await ctx.execution_scope.run_awaitable(
                    communicate, timeout=self.spec.timeout
                )
        except asyncio.TimeoutError:
            await self._log("timeout", f"{self.spec.timeout}s")
            raise HookFailedError(f"timed out after {self.spec.timeout}s") from None
        except ValueError as exc:
            raise HookFailedError(str(exc)) from exc
        finally:
            await terminate_process_tree(proc)
        return outcome_from_output(
            stdout.decode("utf-8", "replace"), proc.returncode or 0, self.limits
        )


async def _post_within_scope(
    scope: ExecutionScope, post: Callable[[], tuple[int, str]], timeout: float
) -> tuple[int, str]:
    """Run a blocking urlopen POST inside the scope's cancellation/budget authority.

    ``scope.run_awaitable`` bounds the wait by the run's remaining budget; the poll
    loop folds token cancellation into a prompt ``CancelledError``. The urlopen
    worker thread cannot be killed mid-request — it stays bounded by its own socket
    timeout, and on abandon the wrapper task is cancelled so the late result is
    discarded quietly.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(post))

    async def _polled() -> tuple[int, str]:
        while not worker.done():
            scope.raise_if_cancelled()
            await asyncio.sleep(0.05)
        return worker.result()

    try:
        return await scope.run_awaitable(_polled(), timeout=timeout)
    finally:
        if not worker.done():
            worker.cancel()


class HttpHookAdapter(_ExternalHookAdapter):
    """POST the projected JSON to a URL; parse the response body for the decision."""

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.url:
            return HookOutcome()
        payload = json.dumps(project_hook_input(ctx, limits=self.limits)).encode("utf-8")
        headers = {"Content-Type": "application/json", **(self.spec.headers or {})}
        url = self.spec.url
        timeout = self.spec.timeout

        def _post() -> tuple[int, str]:
            request = urllib.request.Request(url, data=payload, method="POST", headers=headers)
            # urlopen's own timeout is the single bound (no stacked outer timeout).
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - project-trusted URL
                status = getattr(response, "status", 200) or 200
                body = response.read(self.limits.output_bytes + 1)
                if len(body) > self.limits.output_bytes:
                    raise HookFailedError("HTTP hook response exceeded output limit")
                return status, body.decode("utf-8", "replace")

        try:
            if ctx.execution_scope is None:
                status, body = await asyncio.to_thread(_post)
            else:
                status, body = await _post_within_scope(ctx.execution_scope, _post, timeout)
        except Exception as exc:  # noqa: BLE001 - network/timeout → fail_mode decides.
            await self._log("http_error", f"{type(exc).__name__}: {exc}")
            raise HookFailedError(f"{type(exc).__name__}: {exc}") from exc
        if status >= 400:
            await self._log("http_status", str(status))
            raise HookFailedError(f"HTTP {status}")
        # HTTP carries no exit code; only the JSON body drives the block decision.
        return outcome_from_output(body, 0, self.limits)


class PromptHookAdapter(_ExternalHookAdapter):
    """Re-prompt the LLM (shared gated provider) for advisory context — never blocks.

    The hook's ``prompt`` plus the projected context is sent as a bounded, tool-less model
    call; the reply is injected as ``additional_context``. Kept advisory (no block) so a
    model can't abort a run; use ``command``/``http`` when a hard gate is needed.
    """

    def __init__(
        self,
        spec: ExternalHookSpec,
        logger: "JSONLRunLogger",
        provider: "LLMProvider",
        base_config: "ProviderConfig",
        limits: HookLimitsConfig | None = None,
    ) -> None:
        super().__init__(spec, logger, limits)
        self.provider = provider
        self.base_config = base_config

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.prompt:
            return HookOutcome()
        config = replace(
            self.base_config,
            model=self.spec.model or self.base_config.model,
            max_tokens=min(self.base_config.max_tokens or 1024, 1024),
        )
        snapshot = json.dumps(project_hook_input(ctx, limits=self.limits), ensure_ascii=False)
        messages = [
            Message(
                "user",
                f"{self.spec.prompt}\n\n<hook_input>\n{snapshot}\n</hook_input>",
            )
        ]
        if ctx.execution_scope is None:
            result = await asyncio.wait_for(
                self.provider.complete(messages, [], config), timeout=self.spec.timeout
            )
        else:
            result = await ctx.execution_scope.run_awaitable(
                self.provider.complete(messages, [], config, scope=ctx.execution_scope),
                timeout=self.spec.timeout,
            )
        text, _ = _truncate_utf8((result.content or "").strip(), self.limits.output_bytes)
        text = defang_reserved_tags(_CONTROL_CHARS.sub("�", text))
        return HookOutcome(additional_context=text or None)


class AgentHookAdapter(_ExternalHookAdapter):
    """Run a depth-limited verifier sub-agent for advisory context — never blocks."""

    def __init__(
        self,
        spec: ExternalHookSpec,
        logger: "JSONLRunLogger",
        subagent_factory: Callable[..., Awaitable[str]],
        limits: HookLimitsConfig | None = None,
    ) -> None:
        super().__init__(spec, logger, limits)
        self.subagent_factory = subagent_factory

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.prompt:
            return HookOutcome()
        snapshot = json.dumps(project_hook_input(ctx, limits=self.limits), ensure_ascii=False)
        task = f"{self.spec.prompt}\n\n<hook_input>\n{snapshot}\n</hook_input>"
        if ctx.execution_scope is None:
            result = await asyncio.wait_for(
                self.subagent_factory(task, "hook", self.spec.model), timeout=self.spec.timeout
            )
        else:
            result = await ctx.execution_scope.run_awaitable(
                self.subagent_factory(task, "hook", self.spec.model),
                timeout=self.spec.timeout,
            )
        text, _ = _truncate_utf8((result or "").strip(), self.limits.output_bytes)
        text = defang_reserved_tags(_CONTROL_CHARS.sub("�", text))
        return HookOutcome(additional_context=text or None)


def build_external_adapter(
    spec: ExternalHookSpec,
    *,
    logger: "JSONLRunLogger",
    provider: "LLMProvider | None" = None,
    base_config: "ProviderConfig | None" = None,
    subagent_factory: Callable[..., Awaitable[str]] | None = None,
    limits: HookLimitsConfig | None = None,
) -> _ExternalHookAdapter | None:
    """Build the adapter for one spec, or ``None`` when its dependencies are unavailable.

    ``command`` / ``http`` need only the logger; ``prompt`` needs a provider; ``agent``
    needs the session's sub-agent factory. A spec whose transport lacks its dependency is
    skipped (returns ``None``) rather than raising, so an offline run silently drops the
    model-backed hooks instead of failing.
    """
    if spec.type == "command":
        return CommandHookAdapter(spec, logger, limits)
    if spec.type == "http":
        return HttpHookAdapter(spec, logger, limits)
    if spec.type == "prompt":
        if provider is None:
            return None
        return PromptHookAdapter(
            spec, logger, provider, base_config or ProviderConfig(), limits
        )
    if spec.type == "agent":
        if subagent_factory is None:
            return None
        return AgentHookAdapter(spec, logger, subagent_factory, limits)
    return None

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
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_core.hooks import ExternalHookSpec, HookContext, HookOutcome
from agent_core.models import Message
from agent_core.providers.base import ProviderConfig

if TYPE_CHECKING:
    from agent_core.providers.base import LLMProvider
    from agent_core.storage import JSONLRunLogger

# Bounds on the JSON projection so an external hook never receives the whole transcript.
_MAX_PROJECTED_MESSAGES = 20
_MAX_PROJECTED_CONTENT = 2000

# The lifecycle events an external adapter can attach to, mapped to the HookPipeline list
# they belong in. Tool events (Pre/PostToolUse) use the separate sync tool-hook surface and
# are intentionally excluded here.
LIFECYCLE_EVENT_ATTRS: dict[str, str] = {
    "UserPromptSubmit": "user_prompt_hooks",
    "PostSampling": "post_sampling_hooks",
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
}


def project_hook_input(
    ctx: HookContext,
    *,
    max_messages: int = _MAX_PROJECTED_MESSAGES,
    max_content_chars: int = _MAX_PROJECTED_CONTENT,
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
    tail = ctx.messages[-max_messages:] if max_messages > 0 else []
    projected = []
    for message in tail:
        content = message.content or ""
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "…"
        projected.append({"role": message.role, "content": content})
    data["messages"] = projected
    return data


class HookFailedError(RuntimeError):
    """The external hook itself failed (timeout / transport error / crash).

    Raised by transport ``_invoke`` implementations so :meth:`_ExternalHookAdapter._run`
    can apply the spec's ``fail_mode`` uniformly (open → allow, closed → block).
    """


def outcome_from_output(stdout: str, returncode: int) -> HookOutcome:
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
            if additional is None:
                additional = data.get("additionalContext")
    return HookOutcome(block=block, additional_context=additional, reason=reason, decision=decision)


class _ExternalHookAdapter:
    """Base adapter: implements every lifecycle Protocol, routes to ``_invoke``.

    An instance is appended to exactly one pipeline list (its ``spec.event``), so only the
    matching Protocol method is ever called; implementing them all keeps the adapter a
    structural match for whichever list it lands in. ``_run`` applies the matcher gate and
    the degrade-to-allow guard shared by all transports.
    """

    def __init__(self, spec: ExternalHookSpec, logger: "JSONLRunLogger") -> None:
        self.spec = spec
        self.logger = logger
        self.event = spec.event

    def _matches(self, ctx: HookContext) -> bool:
        # Matcher only applies where there's a trigger to match (the compaction events);
        # elsewhere it's ignored. Pipe-separated exact match, mirroring the reference.
        if self.spec.matcher is None or ctx.trigger is None:
            return True
        patterns = [part.strip() for part in self.spec.matcher.split("|")]
        return ctx.trigger in patterns

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


class CommandHookAdapter(_ExternalHookAdapter):
    """Spawn a subprocess, feed projected JSON on stdin, read stdout/exit-code decision."""

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.command:
            return HookOutcome()
        payload = json.dumps(project_hook_input(ctx)).encode("utf-8")
        proc = await asyncio.create_subprocess_shell(
            self.spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(payload), timeout=self.spec.timeout
            )
        except asyncio.TimeoutError:
            # Single non-stacked timeout: kill the process and reap it (no zombies).
            proc.kill()
            await proc.wait()
            await self._log("timeout", f"{self.spec.timeout}s")
            raise HookFailedError(f"timed out after {self.spec.timeout}s") from None
        return outcome_from_output(stdout.decode("utf-8", "replace"), proc.returncode or 0)


class HttpHookAdapter(_ExternalHookAdapter):
    """POST the projected JSON to a URL; parse the response body for the decision."""

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.url:
            return HookOutcome()
        payload = json.dumps(project_hook_input(ctx)).encode("utf-8")
        headers = {"Content-Type": "application/json", **(self.spec.headers or {})}
        url = self.spec.url
        timeout = self.spec.timeout

        def _post() -> tuple[int, str]:
            request = urllib.request.Request(url, data=payload, method="POST", headers=headers)
            # urlopen's own timeout is the single bound (no stacked outer timeout).
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - project-trusted URL
                status = getattr(response, "status", 200) or 200
                return status, response.read().decode("utf-8", "replace")

        try:
            status, body = await asyncio.to_thread(_post)
        except Exception as exc:  # noqa: BLE001 - network/timeout → fail_mode decides.
            await self._log("http_error", f"{type(exc).__name__}: {exc}")
            raise HookFailedError(f"{type(exc).__name__}: {exc}") from exc
        if status >= 400:
            await self._log("http_status", str(status))
            raise HookFailedError(f"HTTP {status}")
        # HTTP carries no exit code; only the JSON body drives the block decision.
        return outcome_from_output(body, 0)


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
    ) -> None:
        super().__init__(spec, logger)
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
        snapshot = json.dumps(project_hook_input(ctx), ensure_ascii=False)
        messages = [
            Message(
                "user",
                f"{self.spec.prompt}\n\n<hook_input>\n{snapshot}\n</hook_input>",
            )
        ]
        result = await asyncio.wait_for(
            self.provider.complete(messages, [], config), timeout=self.spec.timeout
        )
        text = (result.content or "").strip()
        return HookOutcome(additional_context=text or None)


class AgentHookAdapter(_ExternalHookAdapter):
    """Run a depth-limited verifier sub-agent for advisory context — never blocks."""

    def __init__(
        self,
        spec: ExternalHookSpec,
        logger: "JSONLRunLogger",
        subagent_factory: Callable[[str, str, str | None], Awaitable[str]],
    ) -> None:
        super().__init__(spec, logger)
        self.subagent_factory = subagent_factory

    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        if not self.spec.prompt:
            return HookOutcome()
        snapshot = json.dumps(project_hook_input(ctx), ensure_ascii=False)
        task = f"{self.spec.prompt}\n\n<hook_input>\n{snapshot}\n</hook_input>"
        result = await asyncio.wait_for(
            self.subagent_factory(task, "hook", self.spec.model), timeout=self.spec.timeout
        )
        text = (result or "").strip()
        return HookOutcome(additional_context=text or None)


def build_external_adapter(
    spec: ExternalHookSpec,
    *,
    logger: "JSONLRunLogger",
    provider: "LLMProvider | None" = None,
    base_config: "ProviderConfig | None" = None,
    subagent_factory: Callable[[str, str, str | None], Awaitable[str]] | None = None,
) -> _ExternalHookAdapter | None:
    """Build the adapter for one spec, or ``None`` when its dependencies are unavailable.

    ``command`` / ``http`` need only the logger; ``prompt`` needs a provider; ``agent``
    needs the session's sub-agent factory. A spec whose transport lacks its dependency is
    skipped (returns ``None``) rather than raising, so an offline run silently drops the
    model-backed hooks instead of failing.
    """
    if spec.type == "command":
        return CommandHookAdapter(spec, logger)
    if spec.type == "http":
        return HttpHookAdapter(spec, logger)
    if spec.type == "prompt":
        if provider is None:
            return None
        return PromptHookAdapter(spec, logger, provider, base_config or ProviderConfig())
    if spec.type == "agent":
        if subagent_factory is None:
            return None
        return AgentHookAdapter(spec, logger, subagent_factory)
    return None

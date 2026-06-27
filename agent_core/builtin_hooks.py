"""Built-in programmatic lifecycle hooks — the always-available, in-process handlers.

These give the lifecycle hook seams real default behavior without any external config.
Unlike the config-driven external hooks (``hook_adapters.py``), which only ever see a
JSON snapshot, these hold *live* objects (the run's ``SessionContext`` and event logger)
and so can do things the JSON boundary can't — e.g. read the model's to-do list to decide
whether the agent is really done.

Each is toggled by ``BuiltinHooksConfig`` and assembled into the shared ``HookPipeline``
by ``ReActAgent._build_hook_pipeline``. They mirror the reference project's built-in hooks
(its PostSampling memory/doc hooks; its Stop continuation): observation and gentle steering,
never a hard failure. Every handler degrades to a no-op on error so a buggy hook can never
sink a run — the pipeline awaits them directly in the loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent_core.hooks import HookContext, HookOutcome

if TYPE_CHECKING:
    from agent_core.session import SessionContext
    from agent_core.storage import JSONLRunLogger


# Statuses that mean a to-do item is not yet done (see session.VALID_TODO_STATUS).
_OPEN_TODO_STATUSES = frozenset({"pending", "in_progress"})


class StopCompletionHook:
    """Block a *stop* while the model still has open to-do items.

    This is the clearest payoff of the "可阻断/可续跑" Stop contract: if the model laid
    out a plan with ``update_todos`` and tries to stop with items still ``pending`` /
    ``in_progress``, we block the stop once and inject a continuation directive naming the
    unfinished work. ``ctx.stop_hook_active`` (True after the first block this run) makes
    this fire at most once, and ``react.py``'s ``max_stop_blocks`` caps it regardless — so
    a model that deliberately stops with open items is never pinned in a loop.
    """

    def __init__(self, session: "SessionContext") -> None:
        self.session = session

    async def on_stop(self, ctx: HookContext) -> HookOutcome:
        try:
            if ctx.stop_hook_active:
                return HookOutcome()
            open_items = [
                todo
                for todo in self.session.todos.items()
                if todo.status in _OPEN_TODO_STATUSES
            ]
            if not open_items:
                return HookOutcome()
            listing = "\n".join(f"- [{todo.status}] {todo.content}" for todo in open_items)
            return HookOutcome(
                block=True,
                reason=f"{len(open_items)} unfinished to-do item(s) remain.",
                additional_context=(
                    "You are about to stop, but your to-do list still has unfinished "
                    f"items:\n{listing}\n"
                    "Continue working to complete them, or call update_todos to mark them "
                    "done / removed and then explain briefly why you are stopping."
                ),
            )
        except Exception:  # noqa: BLE001 - a steering hook must never sink the run.
            return HookOutcome()


class PostSamplingObserverHook:
    """Fire-and-forget observation written to the run's JSONL log after each turn.

    Mirrors the reference's internal PostSampling hooks (which run memory extraction / doc
    updates off the critical path) but stays purely observational: it records lightweight
    turn telemetry and never touches the conversation. ``run_post_sampling`` is itself
    fire-and-forget and already reaps exceptions, but we still guard here so a logging
    failure is swallowed rather than surfaced.
    """

    def __init__(self, logger: "JSONLRunLogger") -> None:
        self.logger = logger

    async def after_sampling(self, ctx: HookContext) -> None:
        try:
            last = ctx.last_assistant_message or ""
            await self.logger.write(
                "hook_observe",
                {
                    "hook": "PostSampling",
                    "messages": len(ctx.messages),
                    "assistant_chars": len(last),
                },
            )
        except Exception:  # noqa: BLE001 - observational; must never sink a run.
            pass


class CompactionLoggerHook:
    """Record compaction telemetry around a fold (one class, both Pre/PostCompact methods).

    Pure observation: logs the trigger (``auto`` / ``reactive``), the message count going
    into a fold, and the new summary length coming out. Never blocks compaction and never
    injects context, so it is safe on both the proactive and the forced reactive path.
    """

    def __init__(self, logger: "JSONLRunLogger") -> None:
        self.logger = logger

    async def before_compact(self, ctx: HookContext) -> HookOutcome:
        try:
            await self.logger.write(
                "hook_observe",
                {"hook": "PreCompact", "trigger": ctx.trigger, "messages": len(ctx.messages)},
            )
        except Exception:  # noqa: BLE001 - observational.
            pass
        return HookOutcome()

    async def after_compact(self, ctx: HookContext) -> HookOutcome:
        try:
            await self.logger.write(
                "hook_observe",
                {
                    "hook": "PostCompact",
                    "trigger": ctx.trigger,
                    "summary_chars": len(ctx.summary or ""),
                },
            )
        except Exception:  # noqa: BLE001 - observational.
            pass
        return HookOutcome()


class UserPromptContextHook:
    """Minimal prompt validation + grounding injection (off by default).

    A tiny demonstration of the UserPromptSubmit contract's two powers: it blocks an empty
    prompt (aborting the run before the first model call), and otherwise injects a one-time
    grounding line (a precise UTC timestamp). Kept off by default because ``context.py``
    already injects a userContext block at run start; enable it to layer per-prompt context
    on top, the way the reference's ``additionalContext`` does.
    """

    async def on_user_prompt(self, ctx: HookContext) -> HookOutcome:
        try:
            if not (ctx.prompt or "").strip():
                return HookOutcome(block=True, reason="Empty prompt rejected by UserPromptContext hook.")
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            return HookOutcome(additional_context=f"Prompt submitted at {stamp}.")
        except Exception:  # noqa: BLE001 - a steering hook must never sink the run.
            return HookOutcome()

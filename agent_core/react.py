from __future__ import annotations

import asyncio
import logging
import os
import time
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_core.agents.team import TeamStore
from agent_core.compression import (
    CompressionConfig,
    CompressionEvent,
    CompressionPipeline,
    is_summary_message,
    parse_prompt_too_long_gap,
    shrink_oversize_messages,
    truncate_head_for_ptl_retry,
)
from agent_core.compression_summary import build_summarizer
from agent_core.context import (
    append_system_context,
    build_git_status,
    build_project_instructions,
    current_date_line,
    prepend_user_context,
)
from agent_core.builtin_hooks import (
    CompactionLoggerHook,
    PostSamplingObserverHook,
    PromptValidationHook,
    StopCompletionHook,
)
from agent_core.hook_adapters import LIFECYCLE_EVENT_ATTRS, build_external_adapter
from agent_core.hooks import (
    HookContext,
    HookEvent,
    HookOutcome,
    HookPipeline,
    HooksConfig,
    MaxOutputPostHook,
    OutputLimitConfig,
)
from agent_core.memory import MemoryConfig, MemoryExtractor, MemoryRetriever, MemoryStore
from agent_core.managed_policy import FileManagedPolicyProvider, ManagedPolicyProvider
from agent_core.models import LLMContextTooLongError, Message, ToolCall, ToolResult
from agent_core.model_validation import is_model_allowed, unsupported_model_message
from agent_core.permission_classifier import (
    AutoPermissionClassifier,
    ProviderAutoPermissionClassifier,
)
from agent_core.permission_rules import RuleSet
from agent_core.permission_types import PermissionRuleSource, ToolCallSource
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import LLMProvider, ProviderConfig, gated_provider
from agent_core.sandbox import (
    SandboxAwareMixin,
    SandboxConfig,
    SandboxManager,
    SandboxRequiredError,
    get_shared_manager,
)
from agent_core.scheduler import SchedulerStore
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.skills import (
    SkillRegistry,
    SkillsConfig,
    builtin_programmatic_skills,
    discover_skill_dirs,
    load_skills,
)
from agent_core.storage import JSONLRunLogger
from agent_core.process_supervisor import (
    ProcessSupervisor,
    ShellUnavailableError,
    resolve_bash_executable,
    resolve_powershell_executable,
)
from agent_core.tool_config import ToolSuiteConfig
from agent_core.worktree import WorktreeManager
from agent_core import tokens
from agent_core.tool_use_summary import (
    ToolUseSummaryConfig,
    ToolUseSummarizer,
    build_tool_use_summarizer,
)
from agent_core.transcript import TranscriptStore, new_session_id
from agent_core.tools.catalog import default_tools, populate_registry
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.team import TeamInboxReadTool, TeamMessageSendTool
from agent_core.tools.web import WebPolicyAwareMixin, WebPolicyConfig
from agent_core.ui import AgentUI, NullUI

logger = logging.getLogger(__name__)

# Injected once as a system message when the run crosses its soft deadline, so the
# model can land a useful final answer before the hard wall-clock stop discards work.
WRAPUP_TEXT = (
    "You are almost out of time for this task. Stop calling tools now and reply with "
    "your best final answer based on what you have so far, noting anything left undone."
)
PLAN_MODE_TEXT = (
    "Plan mode is active. Investigate with dedicated read/search tools, maintain Todo/Task "
    "state when useful, and produce a decision-complete implementation plan. Do not edit files "
    "with ordinary file tools. Ordinary "
    "project edits, shell commands, and tests are prohibited. Save the complete plan with "
    "write_plan, then call exit_plan to request approval and restore the previous mode."
)
_MODE_CONTEXT_START = "\n\n<permission-mode-context>\n"
_MODE_CONTEXT_END = "\n</permission-mode-context>"

# Bound on the reactive 413 recovery loop: after summarizing once, we peel the oldest
# whole API rounds and retry ``complete`` at most this many times. This is the guard that
# prevents a 413 → compact → 413 → … infinite loop — once the retries are exhausted (or
# nothing is left to drop) the overflow error propagates instead of spinning forever.
MAX_PTL_RETRIES = 5


def _child_permission_mode(parent_mode: PermissionMode | str, preset: str) -> PermissionMode:
    """The permission mode a spawned child runs under (decision: no privilege escalation).

    Children never inherit ``auto``/``dontask``/``bypass`` — spawning must not launder a
    broad parent grant into an unattended child. The preset maps to the narrowest mode
    that lets it do its declared job: ``full`` (READ+WRITE tools) runs ``acceptedits``
    (writes allowed, DANGEROUS still asks → denied in the child's non-interactive NullUI);
    everything else (``read_only``, ``hook``) runs ``default``. This is safe because the
    spawn call itself already passed the parent's permission gate (user confirmation,
    allow rule, or an explicitly-broad parent mode) — that grant covers the preset's
    declared capability, nothing more. A ``plan``-mode parent never reaches this point:
    the spawn tool call is dry-run by the parent's own gate.
    """
    parent = PermissionMode(parent_mode)
    if parent is PermissionMode.PLAN:
        return PermissionMode.PLAN
    if preset != "full":
        return PermissionMode.DEFAULT
    if parent in {PermissionMode.ACCEPTEDITS, PermissionMode.AUTO, PermissionMode.BYPASS}:
        return PermissionMode.ACCEPTEDITS
    if parent is PermissionMode.DONTASK:
        return PermissionMode.DONTASK
    return PermissionMode.DEFAULT


_READ_ONLY_CHILD_TOOLS = frozenset(
    {
        "list_dir",
        "search_text",
        "glob",
        "git_diff",
        "read_text_file",
        "echo",
        "update_todos",
        "sleep",
        "cron_create",
        "cron_list",
        "cron_delete",
    }
)
_FULL_CHILD_TOOLS = _READ_ONLY_CHILD_TOOLS | {
    "edit_file",
    "multi_edit",
    "apply_patch",
    "write_text_file",
}


# Teammates coordinate through the team store (inbox/status/tasks) — those tools write
# shared team state, not the workspace, so they are allow-ruled in every teammate
# regardless of the child's mode. Workspace tools still go through the child's own gate.
_TEAMMATE_COORDINATION_RULES = RuleSet.from_lists(
    allow=["task_update", "team_inbox_read", "team_message_send"],
    source=PermissionRuleSource.SESSION,
)


@dataclass(slots=True)
class ReActConfig:
    model: str = "claude-opus-4-8"
    provider: str = "claude"
    # NOTE: temperature applies only to legacy models (Haiku 4.5, Sonnet, Opus <= 4.6).
    # Opus 4.7+/Fable/Mythos reject sampling params, so the Claude provider drops it for
    # them (see providers/claude.py _is_adaptive_thinking_model). Left for debug runs
    # that override --model to a legacy id.
    temperature: float = 0.2
    # Answer-token cap. 16k is non-truncating headroom for Opus 4.8 (you're billed for
    # tokens actually produced, not the cap) and stays under the SDK's ~16k non-streaming
    # timeout guard while leaving room for streaming runs.
    max_tokens: int = 16000
    # No fixed step cap by default: like Claude Code, the loop runs until the model
    # stops requesting tools. Set an int only if you want a hard ceiling on tool turns.
    max_steps: int | None = None
    # Upper bound on how many times a Stop hook may block the agent's natural
    # termination and force it to keep running ("可阻断/可续跑"). Once this many
    # consecutive blocks have happened, the loop stops regardless, so a misbehaving
    # stop hook can't pin the agent in an infinite continue loop. 0 disables stop-hook
    # blocking entirely (a stop hook can still observe/inject context but never continue).
    max_stop_blocks: int = 3
    # Wall-clock safety net so a runaway/stuck loop can't hang forever. Configurable
    # via the [limits] toml table, AGENT_MAX_WALL_SECONDS, or --max-wall-seconds.
    # None disables the wall cap entirely (cooperative Esc-cancel still applies);
    # the whole sub-agent fan-out shares one budget (see run()'s deadline param).
    max_wall_seconds: float | None = 1800.0
    # Fraction of the run's wall budget after which a one-time "wrap up now" nudge is
    # injected, so the model can return a useful partial answer before the hard stop.
    # 1.0 (or any value >= 1) disables the nudge; ignored when there is no wall cap.
    soft_deadline_fraction: float = 0.9
    # Extended-thinking token budget for the Claude provider. None disables thinking
    # (default); a positive int enables it and is passed through _provider_config().
    thinking_budget: int | None = 4096
    # output_config.effort depth/cost control for effort-capable models (Opus 4.5+,
    # Sonnet 4.6, Fable/Mythos). "high" is the safe agentic default ("xhigh" is Opus
    # 4.8-best but errors on Sonnet 4.6/Opus<=4.6). The provider drops it for models
    # that don't support the level (see providers/claude.py _effort_for_model). None omits it.
    effort: str | None = "high"
    # Stream tokens to a live UI as they arrive. Only takes effect when the UI is
    # live (ConsoleUI); NullUI never streams. CLI exposes this via --no-stream.
    stream: bool = True
    # Tools returned in the same model turn may run concurrently when their declared
    # resources do not conflict.
    parallel_tools: bool = True
    max_tool_workers: int = 4
    # Cap on simultaneous in-flight LLM API calls across the whole multi-agent
    # fan-out (leader + concurrent children), enforced by the shared provider gate.
    max_api_concurrency: int = 8
    # Sustained API request ceiling per minute across that same fan-out; 0 = unlimited.
    api_rate_limit_per_min: int = 0
    permission: PermissionMode | str = PermissionMode.DEFAULT
    run_dir: str = "runs"
    # Root for resumable session transcripts (distinct from the ``run_dir`` event log).
    # Mirrors the reference's ``~/.claude/projects`` so sessions are scoped per project
    # (a sanitized cwd subdir) and listable/resumable across projects. ``~`` is expanded.
    # Empty string disables transcript persistence entirely.
    session_dir: str = "~/.polaris/projects"
    # When True (reference behavior), a context-collapse fold writes a compact boundary +
    # summary into the transcript, so a resume loads the *compacted* state (only messages
    # after the last boundary) instead of replaying the full pre-fold history. When False,
    # the transcript stays a faithful full record and a resume reloads everything, letting
    # the live loop re-compact (cheaper/simpler; the original decoupled behavior).
    persist_compaction_boundary: bool = True
    system_prompt: str = (
        "You are a ReAct agent. Reason briefly, call tools when useful, "
        "and return a final answer when the task is complete. "
        "For non-trivial, multi-step tasks, call update_todos first to lay out a plan, "
        "then keep it current — mark one item in_progress at a time and complete it before "
        "moving on. For self-contained sub-investigations, consider dispatch_agent to run "
        "them in a fresh context. For work that needs a team of cooperating agents, use "
        "the team tools explicitly: team_create, task_create, teammate_spawn, task_update, "
        "and team_status. Use bash or powershell for commands; long-running commands may "
        "continue in the background and can be inspected or stopped with task_output and "
        "task_stop. Use tool_search to discover deferred capabilities such as LSP, notebook, "
        "worktree, configuration, MCP resources, and scheduling tools. Multiple tool calls "
        "in the same turn may run concurrently when "
        "their resources are independent; if an action needs the output from a previous "
        "tool call, wait until the next turn to request it."
    )
    # Discover project instructions (CLAUDE.md) at run start and inject them as the
    # ``claudeMd`` entry of the pinned ``<system-reminder>`` userContext user message.
    # Off when False (or AGENT_DISABLE_CLAUDE_MD is truthy, folded in by
    # config.resolve_context_config). The joined block is truncated to claudemd_max_chars.
    project_instructions: bool = True
    claudemd_max_chars: int = 32000
    # Collect a one-time git snapshot (branch/main/user/status/log) at run start and
    # append it as the ``gitStatus`` entry of the base system block (systemContext). Off
    # when False (or AGENT_DISABLE_GIT_CONTEXT is truthy, folded in by
    # config.resolve_context_config).
    git_context: bool = True
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputLimitConfig = field(default_factory=OutputLimitConfig)
    tool_use_summary: ToolUseSummaryConfig = field(default_factory=ToolUseSummaryConfig)
    # Skill / slash-command subsystem (loaded eagerly at startup when enabled).
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    # Lifecycle-hook subsystem: built-in programmatic hook toggles + config-driven
    # external hooks. Assembled into the shared HookPipeline by _build_hook_pipeline.
    hooks: HooksConfig = field(default_factory=HooksConfig)
    # OS sandbox (enforcement layer): wraps dangerous command execution in bwrap/
    # sandbox-exec on Linux/macOS, no-op on Windows. Disabled by default.
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    # Fine-grained allow/deny/ask rules (policy layer). Empty by default → the run
    # behaves exactly as the bare per-mode ToolRisk gate did.
    permission_rules: RuleSet = field(default_factory=RuleSet)
    # Outbound domain policy for the web tools ([web] table, decision D10):
    # blocked_domains always refuse; in unattended modes (auto/dontask/bypass) a
    # domain not in allowed_domains is refused too (fail-closed exfiltration guard).
    web: WebPolicyConfig = field(default_factory=WebPolicyConfig)
    tools: ToolSuiteConfig = field(default_factory=ToolSuiteConfig)


@dataclass(slots=True)
class AgentRunResult:
    answer: str
    messages: list[Message]
    steps: int
    run_id: str


class ReActAgent:
    def __init__(
        self,
        provider: LLMProvider,
        config: ReActConfig | None = None,
        tools: ToolRegistry | None = None,
        hooks: HookPipeline | None = None,
        logger: JSONLRunLogger | None = None,
        memory_store: MemoryStore | None = None,
        retriever: MemoryRetriever | None = None,
        extractor: MemoryExtractor | None = None,
        team_store: TeamStore | None = None,
        ui: AgentUI | None = None,
        session_id: str | None = None,
        transcript: "TranscriptStore | None" = None,
        sandbox: SandboxManager | None = None,
        permission_classifier: AutoPermissionClassifier | None = None,
        managed_policy_provider: ManagedPolicyProvider | None = None,
        mcp_manager: object | None = None,
        workspace: str | Path | None = None,
        ask_user: Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.config = config or ReActConfig()
        self._initial_workspace = Path(workspace or Path.cwd()).resolve()
        self.managed_policy_provider = managed_policy_provider or FileManagedPolicyProvider()
        # Effective monotonic deadline of the in-flight run(), shared with children
        # spawned by the sub-agent/teammate factories so the whole fan-out is bounded
        # by one budget. Set at the top of run(); None when no run is active or uncapped.
        self._active_deadline: float | None = None
        self._reported_missed_jobs: set[str] = set()
        # Wrap the provider in a shared, bounded concurrency gate (idempotent): the
        # top-level agent creates it from config, and children spawned with
        # ``provider=self.provider`` reuse the same gate, so the whole fan-out shares
        # one budget. Config must be set first so the knobs below resolve.
        self.provider = gated_provider(
            provider,
            max_concurrency=self.config.max_api_concurrency,
            rate_limit=self.config.api_rate_limit_per_min,
        )
        # Interactive /fast is intentionally session-only.
        self.fast_mode = False
        self.session_title: str | None = None
        self._sandbox_cli_locked = False
        self._retired_sandboxes: list[SandboxManager] = []
        self._plugin_tool_names: set[str] = set()
        self._plugin_mcp_manager: Any | None = None
        self.plugin_agents: dict[str, str] = {}
        self.registry = tools or self.default_registry()
        self.registry.rebind_workspace(str(self._initial_workspace))
        self.logger = logger or JSONLRunLogger(self.config.run_dir)
        # Resumable session transcript (distinct from the event logger above). An injected
        # store wins; otherwise build one from config unless ``session_dir`` is disabled
        # (empty). ``parent_uuid`` chaining is tracked across appends by ``_emit``.
        self.session_id = session_id or new_session_id()
        if transcript is not None:
            self.transcript: TranscriptStore | None = transcript
        elif self.config.session_dir:
            self.transcript = TranscriptStore(
                self.config.session_dir, self._initial_workspace, self.session_id
            )
        else:
            self.transcript = None
        # uuid of the last message appended to the transcript, so each new message links
        # back to its predecessor. Reset at the top of every run().
        self._last_message_uuid: str | None = None
        self.compression = CompressionPipeline(self.config.compression)
        # Track A summarizer (or None → deterministic Track B). Built from the gated
        # provider so summary calls share the fan-out's API budget; None for
        # FakeProvider / no key / disabled, keeping offline runs byte-stable.
        self._summarizer = build_summarizer(
            self.provider, self._provider_config(), self.config.compression
        )
        # Async tool-use progress label (UI-only, ephemeral). None when disabled or for
        # FakeProvider (offline byte-stable). The live-UI / leader-only gates are applied
        # at fire time (``_fire_tool_use_summary``) since ``self.ui`` is set just below.
        self._tool_use_summarizer: ToolUseSummarizer | None = build_tool_use_summarizer(
            self.provider, self._provider_config(), self.config.tool_use_summary
        )
        # The in-flight (or just-finished) label task; fired after a tool batch, awaited and
        # emitted on the next turn so the Haiku call overlaps the main model call. Reset per run.
        # ``_pending_tool_use_names`` carries that batch's tool names for the UI emit (the task
        # itself only returns the label string).
        self._pending_tool_use_summary: "asyncio.Task[str | None] | None" = None
        self._pending_tool_use_names: list[str] = []
        # SessionStart fires once per agent (on the first run()); SessionEnd is
        # host-driven — the embedding host calls fire_session_end() at its own
        # session boundary (the CLI does so at run/chat exit). (Distinct from
        # ``_session_started``, the /cost timestamp.)
        self._session_start_fired = False
        self.ui = ui or NullUI()
        self.team_store = team_store or TeamStore(Path(self.config.run_dir) / "teams")
        # Skill / slash-command registry, loaded eagerly from disk (bundled + user +
        # project) per the eager-loading invariant. Failures degrade to an empty registry
        # so a bad skill file can never sink agent construction.
        self.skills = self._load_skills()
        # Per-run shared state for session-aware tools (planning, sub-agents). The
        # registry may have been built before this agent existed (the CLI path), so we
        # rebind every session-aware tool to *this* session below.
        self.session = SessionContext(
            workspace=self._initial_workspace,
            session_id=self.session_id,
            agent_id=self.logger.run_id,
            subagent_factory=self._spawn_subagent,
            teammate_factory=self._spawn_teammate,
            team_store=self.team_store,
            ui_notify=self.ui.on_todos,
            skills=self.skills,
            run_dir=self.config.run_dir,
            run_id=self.logger.run_id,
            permission_mode_setter=self.set_permission_mode,
            tool_suite=self.config.tools,
            logger=self.logger,
            audit_event=self._audit_event,
            mcp_manager=mcp_manager,
            registry=self.registry,
        )
        self.process_supervisor = ProcessSupervisor(
            self.config.tools.shell,
            Path(self.config.run_dir) / "tasks" / self.session_id,
            event_sink=self._audit_event,
        )
        self.session.process_supervisor = self.process_supervisor
        if PermissionMode(self.config.permission) is PermissionMode.PLAN:
            self.session.plan_state.enter(
                PermissionMode.DEFAULT.value,
                self.session.plan_store.path_for(self.session_id, self.session.agent_id),
            )
        # OS sandbox manager (enforcement layer), built eagerly per the eager-loading
        # invariant. Constructed with fail_if_unavailable honored here (raises before the
        # run starts). An injected manager wins (children receive their parent's, see
        # _make_subagent_child); otherwise the process-shared instance for this
        # (config, workspace) is used, so heavyweight readying happens once per process,
        # not once per agent. Bound into every sandbox-aware command tool below.
        self.sandbox = sandbox if sandbox is not None else get_shared_manager(
            self.config.sandbox, workspace=self.session.workspace
        )
        # Ready the active backend now (verify container runtime + image, boot the VM +
        # base snapshot). Idempotent — a shared, already-prepared manager returns
        # immediately. Degrades to passthrough on failure unless fail_if_unavailable.
        self.sandbox.prepare()
        self.session.worktree_manager = WorktreeManager(
            self.session, self.registry, self.sandbox, self.config.tools.worktree
        )
        for tool in self.registry.list():
            if isinstance(tool, SessionAwareMixin):
                tool.bind_session(self.session)
            if isinstance(tool, SandboxAwareMixin):
                tool.bind_sandbox(self.sandbox)
            if isinstance(tool, WebPolicyAwareMixin):
                tool.bind_web_policy(self.config.web, unattended=self._is_unattended_mode())
        self.registry.bind_runtime(
            session=self.session,
            sandbox=self.sandbox,
            web_policy=self.config.web,
            unattended=self._is_unattended_mode(),
        )
        shell_config = self.config.tools.shell
        available_shells = 0
        for name, enabled, resolver, configured in (
            ("bash", shell_config.enabled and shell_config.bash.enabled,
             resolve_bash_executable, shell_config.bash.executable or os.getenv("POLARIS_BASH_PATH")),
            ("powershell", shell_config.enabled and shell_config.powershell.enabled,
             resolve_powershell_executable, shell_config.powershell.executable),
        ):
            if enabled:
                try:
                    resolver(configured)
                except ShellUnavailableError:
                    self.registry.unregister(name)
                else:
                    available_shells += 1
            else:
                self.registry.unregister(name)
        if not available_shells:
            self.registry.unregister("task_output")
            self.registry.unregister("task_stop")
        if ask_user is not None:
            self.session.ask_user = ask_user
        elif self.ui.is_live:
            self.session.ask_user = self.ui.ask_questions
        else:
            self.registry.unregister("ask_user_question")
        if mcp_manager is None:
            self.registry.unregister("list_mcp_resources")
            self.registry.unregister("read_mcp_resource")
        if not self.config.tools.lsp.servers:
            self.registry.unregister("lsp")
        if not self.config.tools.scheduler.enabled:
            for name in ("cron_create", "cron_list", "cron_delete"):
                self.registry.unregister(name)
        blanket_denied = {
            rule.tool_name for rule in self.config.permission_rules.deny if rule.content is None
        }
        try:
            managed_rules = self.managed_policy_provider.load().rules()
            blanket_denied.update(
                rule.tool_name for rule in managed_rules.deny if rule.content is None
            )
        except Exception:
            pass
        for name in blanket_denied:
            if any(item.name == name for item in self.registry.deferred()):
                self.registry.unregister(name)
        # The model-facing ``skill`` tool is dead weight when no skill is model-invocable,
        # so drop it from the advertised tool set in that case (saves tokens, avoids
        # offering the model a tool it can't use).
        if not self.skills.model_invocable():
            self.registry.unregister("skill")
        # Session-only acknowledgement for explicitly accepted unsandboxed
        # unattended operation. It is never persisted or inherited by children.
        self._unsandboxed_permission_ack = False
        # D3: an unattended permission mode with no working sandbox refuses to start
        # (interactive runs get a confirm prompt; the config/env opt-out is explicit).
        self._enforce_unattended_sandbox_requirement(self.config.permission)
        # Only wire an interactive prompter when the UI can actually ask the user;
        # otherwise an "ask" decision collapses to a denial (non-interactive behavior).
        # The fine-grained rule set + sandbox make the policy argument-aware.
        self.permissions = PermissionPolicy(
            self.config.permission,
            prompter=self.ui.request_permission if self.ui.is_live else None,
            rules=self.config.permission_rules,
            sandbox=self.sandbox,
            workspace=self.session.workspace,
            plan_state=self.session.plan_state,
            managed_policy_provider=self.managed_policy_provider,
            allow_unsandboxed_unattended=self._unsandboxed_permission_ack,
        )
        self.session.permission_grant_setter = self.permissions.add_session_rule
        self.session.permission_workspace_setter = lambda path: setattr(
            self.permissions, "workspace", path
        )
        self.session.registered_tool_names = frozenset(tool.name for tool in self.registry.list())
        if PermissionMode(self.config.permission) in self.permissions.managed_policy.forbidden_modes:
            raise ValueError(
                f"permission mode {PermissionMode(self.config.permission).value!r} is forbidden by managed policy"
            )
        # One pipeline shared by the executor (pre/post tool hooks) and the run loop
        # (lifecycle hooks: user-prompt / post-sampling / pre-/post-compact / stop). An
        # injected pipeline (tests/library use) wins; otherwise assemble the default from
        # config — the MaxOutput tool hook plus the enabled built-in + external lifecycle
        # hooks. Built after session/logger exist so hooks can close over them.
        self.hooks = hooks or self._build_hook_pipeline()
        self.permission_classifier = permission_classifier or ProviderAutoPermissionClassifier(
            self.provider,
            self._provider_config,
        )
        self.executor = ToolExecutor(
            self.registry,
            self.permissions,
            self.hooks,
            self.logger,
            self.ui,
            self.permission_classifier,
            parallel_tools=self.config.parallel_tools,
            max_workers=self.config.max_tool_workers,
        )
        # Strong refs to in-flight fire-and-forget PostSampling hook tasks, so they
        # aren't garbage-collected mid-run; reaped best-effort at terminal returns.
        self._background_hook_tasks: set[asyncio.Task[None]] = set()
        self.memory_store, self.retriever, self.extractor = self._build_memory(
            memory_store, retriever, extractor
        )
        # Running prompt-token figure from the last response's usage (Phase 2B). The
        # auto-compact gate thresholds against this (parity with the reference) instead
        # of a char ratio; 0 until the first response with usage arrives.
        self._last_usage_tokens: int = 0
        # Session-cumulative token totals, surfaced by the chat ``/cost``/``/status``
        # commands. Summed across every run() on this agent (a chat session reuses one).
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        # Per-run token totals (reset each run() — see below), surfaced in the final recap.
        # Unlike the ``_session_*`` counters these reflect only the current run.
        self._run_input_tokens: int = 0
        self._run_output_tokens: int = 0
        self._run_context_tokens: int = 0
        # When this agent was created — a chat session builds one agent, so this doubles
        # as the session start for ``/cost`` duration.
        self._session_started: float = time.monotonic()
        # Enabled plugins load eagerly at process/session start. A broken generation
        # never replaces the already-valid built-in registries.
        try:
            from agent_core.plugins import PluginManager, reload_plugins

            if PluginManager(self.session.workspace).enabled_ids():
                reload_plugins(self)
        except Exception as exc:  # noqa: BLE001 - plugins are optional
            logging.getLogger(__name__).warning(
                "enabled plugins failed to load; using built-ins only: %s: %s",
                type(exc).__name__,
                exc,
            )

    def _build_memory(
        self,
        memory_store: MemoryStore | None,
        retriever: MemoryRetriever | None,
        extractor: MemoryExtractor | None,
    ) -> tuple[MemoryStore | None, MemoryRetriever | None, MemoryExtractor | None]:
        """Wire up cross-conversation memory, but only when it's enabled.

        Injected components win (tests/customisation); otherwise the missing pieces
        are built from ``config.memory``. When memory is disabled this returns all
        ``None`` and the run loop behaves exactly as it did before memory existed.
        """
        if not self.config.memory.enabled:
            return None, None, None
        store = memory_store or MemoryStore(Path(self.config.memory.dir) / "memory.jsonl")
        retriever = retriever or MemoryRetriever(store, self.config.memory)
        extractor = extractor or MemoryExtractor(
            self.provider, store, self.config.memory, self._provider_config()
        )
        return store, retriever, extractor

    def _load_skills(self) -> SkillRegistry:
        """Discover and load skills at startup, or return an empty registry.

        Best-effort and eager (per the eager-loading invariant): a malformed skill file
        or a missing directory degrades to fewer/zero skills, never an exception that
        sinks construction. Returns an empty registry when skills are disabled.
        """
        if not self.config.skills.enabled:
            return SkillRegistry()
        try:
            dirs = discover_skill_dirs(self._initial_workspace, self.config.skills)
            markdown = load_skills(dirs, disabled=self.config.skills.disabled)
            # Programmatic (Python) skills register themselves via @programmatic_skill and
            # are merged in alongside markdown ones (same name → markdown wins, since it's
            # added last and the registry's last-add wins). Honour the disabled list too.
            blocked = {name.strip().lower() for name in self.config.skills.disabled}
            programmatic = [s for s in builtin_programmatic_skills() if s.name.lower() not in blocked]
            return SkillRegistry(programmatic + markdown)
        except Exception as exc:  # noqa: BLE001 - skill loading must never crash agent construction
            logger.warning(
                "skill loading failed; continuing with no skills: %s: %s",
                type(exc).__name__, exc,
            )
            return SkillRegistry()

    def _is_unattended_mode(self, mode: PermissionMode | str | None = None) -> bool:
        """True for modes that can execute without per-call human confirmation."""
        try:
            resolved = PermissionMode(self.config.permission if mode is None else mode)
        except ValueError:
            return False
        return resolved in {PermissionMode.AUTO, PermissionMode.DONTASK, PermissionMode.BYPASS}

    def _enforce_unattended_sandbox_requirement(self, mode: PermissionMode | str) -> None:
        """D3: unattended modes (auto/dontask/bypass) must not run with zero isolation.

        Those modes execute WRITE (and for bypass, DANGEROUS) tools without per-call
        confirmation, so the OS sandbox is the only remaining boundary — if it is
        disabled or unavailable, refuse to construct. Interactive runs are offered a
        one-off "continue unsandboxed?" confirmation instead. The explicit opt-out
        (``[sandbox] allow_unattended_unsandboxed`` or ``AGENT_SANDBOX_ALLOW_UNATTENDED``,
        read here so embedded/test constructions honor it too) is logged, never silent.
        """
        try:
            resolved = PermissionMode(mode)
        except ValueError:
            return  # unknown mode string — PermissionPolicy will reject it just below
        if not self._is_unattended_mode(resolved):
            return
        if self.sandbox.is_enabled():
            return
        env = os.getenv("AGENT_SANDBOX_ALLOW_UNATTENDED")
        env_opt_out = env is not None and env.strip().lower() in {"1", "true", "yes", "on"}
        managed_requires_sandbox = False
        if self.managed_policy_provider is not None:
            managed_requires_sandbox = (
                self.managed_policy_provider.load().require_sandbox_for_unattended
            )
        if (
            self.config.sandbox.allow_unattended_unsandboxed or env_opt_out
        ) and not managed_requires_sandbox:
            self._unsandboxed_permission_ack = True
            logger.warning(
                "permission mode %r runs without per-call confirmation AND without a "
                "sandbox (explicit allow_unattended_unsandboxed opt-out)",
                resolved.value,
            )
            return
        if self._unsandboxed_permission_ack:
            return
        reason = (
            self.sandbox.unavailable_reason()
            or "the sandbox is disabled ([sandbox] enabled = false)"
        )
        message = (
            f"Permission mode '{resolved.value}' executes commands without per-call "
            f"confirmation, but {reason}."
        )
        if self.ui.is_live and self.ui.confirm_action(message):
            self._unsandboxed_permission_ack = True
            logger.warning("user confirmed running mode %r without a sandbox", resolved.value)
            return
        raise SandboxRequiredError(
            message
            + " Enable the sandbox ([sandbox] enabled = true / --sandbox), switch to an "
            "attended permission mode, or opt out explicitly with [sandbox] "
            "allow_unattended_unsandboxed = true (env AGENT_SANDBOX_ALLOW_UNATTENDED=1)."
        )

    def set_permission_mode(
        self,
        mode: PermissionMode | str,
        *,
        source: str = "api",
    ) -> PermissionMode:
        """Atomically switch configuration, execution policy, and web egress mode."""
        target = PermissionMode(mode)
        previous = PermissionMode(self.config.permission)
        if target == previous:
            return target

        reload_error = self.permissions.refresh_managed_policy()
        if reload_error is not None:
            raise ValueError(f"managed policy reload failed: {reload_error}")
        if target in self.permissions.managed_policy.forbidden_modes:
            raise ValueError(f"permission mode {target.value!r} is forbidden by managed policy")

        self._enforce_unattended_sandbox_requirement(target)
        self.permissions.allow_unsandboxed_unattended = self._unsandboxed_permission_ack
        if target is PermissionMode.PLAN and previous is not PermissionMode.PLAN:
            self.session.plan_state.enter(
                previous.value,
                self.session.plan_store.path_for(self.session_id, self.session.agent_id),
            )
        self.config.permission = target
        self.permissions.mode = target
        for tool in self.registry.list():
            if isinstance(tool, WebPolicyAwareMixin):
                tool.bind_web_policy(self.config.web, unattended=self._is_unattended_mode(target))
        try:
            self.logger.write_nowait(
                "permission_mode",
                {
                    "from": previous.value,
                    "to": target.value,
                    "source": source,
                    "sandboxed": self.sandbox.is_enabled(),
                    "unsandboxed_ack": self._unsandboxed_permission_ack,
                },
            )
        except OSError as exc:
            # Audit is observational: an unwritable run directory must be visible,
            # but must not leave the already-validated live policy half-switched.
            logger.warning("could not write permission_mode audit event: %s", exc)
        if previous is PermissionMode.PLAN and target is not PermissionMode.PLAN:
            self.session.plan_state.clear()
        return target

    def _sync_permission_mode_context(self, messages: list[Message]) -> None:
        """Refresh the mode system block immediately before every provider call."""
        system = next((message for message in messages if message.role == "system"), None)
        if system is None:
            return
        base = system.content.split(_MODE_CONTEXT_START, 1)[0]
        if PermissionMode(self.config.permission) is not PermissionMode.PLAN:
            system.content = base
            return
        artifact = self.session.plan_state.artifact_path
        detail = PLAN_MODE_TEXT + f"\nPlan artifact: {artifact}"
        system.content = base + _MODE_CONTEXT_START + detail + _MODE_CONTEXT_END

    async def compact_now(self, messages: list[Message]) -> tuple[list[Message], int]:
        """Force a compaction fold of ``messages`` now, ignoring the token gate.

        Reuses the reactive (always-compacts) path and the agent's own summarizer, so a
        chat ``/compact`` produces the same kind of fold the loop would. Returns the
        compacted history and the number of characters saved (0 when nothing folded).
        Best-effort: any failure returns the input unchanged with 0 saved.
        """
        before = sum(len(m.content) for m in messages)
        try:
            compacted, _events = await self.compression.reactive_compact(
                messages,
                model=self.config.model,
                token_estimator=self._estimate_tokens,
                summarizer=self._summarizer,
            )
        except Exception as exc:  # noqa: BLE001 - a manual compact must never crash the chat session
            logger.warning(
                "manual compaction failed; history left unchanged: %s: %s",
                type(exc).__name__, exc,
            )
            return messages, 0
        after = sum(len(m.content) for m in compacted)
        return compacted, max(0, before - after)

    @staticmethod
    def default_registry() -> ToolRegistry:
        # The tool set lives in the tools package (self-registered via @builtin_tool
        # and auto-discovered) — adding a tool there needs no change here.
        registry = ToolRegistry()
        populate_registry(registry)
        return registry

    async def run(
        self,
        task: str,
        should_cancel: Callable[[], bool] | None = None,
        deadline: float | None = None,
        history: list[Message] | None = None,
        midturn_drain: Callable[[], list[Message]] | None = None,
        *,
        _user_messages: list[Message] | None = None,
    ) -> AgentRunResult:
        """Drive the ReAct loop to completion and return the final answer.

        The single (async) entry point: synchronous callers wrap the coroutine in
        one top-level ``asyncio.run(agent.run(task))``; async callers just await it.

        ``deadline`` is a ``time.monotonic()`` timestamp shared by an enclosing run:
        sub-agents/teammates inherit the parent's deadline so the whole fan-out is
        bounded by one wall-clock budget instead of each child getting a fresh one.
        When ``None``, the deadline is derived from ``config.max_wall_seconds`` (and
        is itself ``None`` when that is unset, disabling the wall cap).

        ``history`` seeds the loop with a prior conversation — the mechanism behind both
        ``chat`` cross-turn memory and ``--resume``/``--continue``. The system prompt and
        project context are rebuilt fresh each call (so any system messages carried in
        ``history`` are dropped), and the history is assumed already persisted, so it is
        only re-linked into the message chain, not re-written to the transcript.
        """
        self._last_message_uuid = None
        self._pending_tool_use_summary = None
        self._background_hook_tasks = set()
        self._run_start_time = time.monotonic()
        self._run_input_tokens = 0
        self._run_output_tokens = 0
        self._run_context_tokens = 0
        # Per-task VM rollback: restore the base snapshot so each run starts from a clean
        # guest (no-op for native/container tiers and when reset_each_task is off).
        self.sandbox.reset()
        # A live UI's interactive permission prompt runs on a worker thread; give it
        # the running loop so it can bridge the prompt back onto the main thread.
        if self.ui.is_live:
            self.ui.bind_event_loop(asyncio.get_running_loop())
        user_messages = list(_user_messages) if _user_messages is not None else [Message("user", task)]
        if not user_messages:
            raise ValueError("at least one user message is required")
        if any(message.role != "user" for message in user_messages):
            raise ValueError("run user messages must all have role='user'")
        user_message = user_messages[0]
        recalled_task = "\n".join(message.content for message in user_messages)
        system_prompt = self.config.system_prompt
        messages: list[Message] = [Message("system", system_prompt), *user_messages]
        for submitted in user_messages:
            await self.logger.write(
                "user",
                {
                    "content": submitted.content,
                    "uuid": submitted.uuid,
                    **self._trace_fields(),
                },
            )
        # ``_recall``/``_inject_project_context`` position the recall and pinned
        # userContext blocks relative to the trailing user task, so the task must already
        # be in place. Final front order: system(+gitStatus) → (recall) → userContext → task.
        await self._recall(recalled_task, messages)
        await self._inject_project_context(messages)
        self._sync_permission_mode_context(messages)
        # Splice prior conversation in just before the new task (after the pinned context),
        # so the order is [system, (recall), userContext, ...history..., task]. System and
        # pinned context are rebuilt fresh above, so any carried in ``history`` are dropped
        # to avoid stacking a new copy on every turn.
        if history:
            insert_at = messages.index(user_message)
            for past in history:
                if past.role == "system" or past.metadata.get("pinned") == "user_context":
                    continue
                messages.insert(insert_at, past)
                insert_at += 1
                self._last_message_uuid = past.uuid
        # The new batch was present while recall/project context found its anchor.
        # Re-add each message sequentially below so hook-added context and transcript
        # parent links stay in exact per-command order.
        for submitted in user_messages:
            messages.remove(submitted)
        # SessionStart (observational) fires once per agent, on its first run —
        # before UserPromptSubmit so subscribers see the session open first.
        if not self._session_start_fired:
            self._session_start_fired = True
            await self._fire_observational(
                HookEvent.SESSION_START,
                {"run_id": self.logger.run_id, **self._trace_fields()},
                messages,
            )
        # UserPromptSubmit fires once the task is in place but before the first model
        # call, so a hook can abort the run (block), rewrite the task to neutralize untrusted
        # framing (transformed_prompt), or inject extra grounding (additional_context). The new
        # task is chained + persisted inside the helper, *after* the hook, so a rewrite is what
        # gets recorded and sent to the model.
        batch_mode = _user_messages is not None
        first_should_query = True
        first_block_reason = ""
        for index, submitted in enumerate(user_messages):
            messages.append(submitted)
            blocked, was_blocked = await self._run_user_prompt_hooks(
                messages,
                submitted,
                submitted.content,
                blocked_as_warning=batch_mode or index > 0,
            )
            if index == 0 and was_blocked:
                first_should_query = False
                first_block_reason = (
                    messages[-1].metadata.get("reason", "")
                    if messages
                    else ""
                )
            if blocked is not None:
                return blocked
        if not first_should_query:
            answer = str(first_block_reason or "First queued prompt was blocked by a hook.")
            self._emit_recap(messages, 0, "blocked")
            await self.logger.write("final", {"answer": answer, "stopped": "blocked"})
            return AgentRunResult(answer, messages, 0, self.logger.run_id)
        # Exposed read-only to the persistent terminal's Ctrl+O transcript snapshot.
        self._active_messages = messages

        cancelled = should_cancel or (lambda: False)
        start = time.monotonic()
        if deadline is None and self.config.max_wall_seconds is not None:
            deadline = start + self.config.max_wall_seconds
        # Stash for the sub-agent/teammate factories so children share this budget.
        self._active_deadline = deadline
        # Soft deadline: a fraction of *this run's* window (not max_wall_seconds, which
        # may differ from the inherited budget), after which we nudge the model once.
        soft_threshold: float | None = None
        if deadline is not None and self.config.soft_deadline_fraction < 1.0:
            soft_threshold = start + (deadline - start) * self.config.soft_deadline_fraction
        wrapup_sent = False
        # How many times a Stop hook has blocked termination so far this run; bounds
        # the "可阻断/可续跑" continuation so a stop hook can't pin the loop forever.
        stop_blocks = 0
        step = 0
        while True:
            # The natural exit below — the model returning no tool calls — is the primary
            # stop, so a task can take as many tool turns as it needs. These guards are only
            # a safety net so a runaway or stuck loop can't spin forever: a cooperative
            # cancel signal (e.g. the user pressing Esc), an optional hard step ceiling,
            # and a wall-clock deadline that bounds the whole run.
            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
            if self.config.max_steps is not None and step >= self.config.max_steps:
                return await self._stopped(messages, step, "max_steps", "reaching max_steps")
            if deadline is not None:
                now = time.monotonic()
                if now > deadline:
                    return await self._stopped(messages, step, "deadline", "reaching the wall-clock deadline")
                # One-time soft nudge so the model can wrap up before the hard stop.
                if soft_threshold is not None and not wrapup_sent and now >= soft_threshold:
                    messages.append(Message("system", WRAPUP_TEXT, metadata={"deadline_wrapup": True}))
                    await self.logger.write("deadline_wrapup", {"step": step})
                    wrapup_sent = True
            step += 1

            # Build the post-compact file re-injection attachments once per turn; they
            # are appended to the conversation tail ONLY if a real fold happens (the
            # pipeline forwards them to build_post_compact_messages inside the collapse
            # stage). Empty when nothing has been read yet.
            attachments = self._build_read_attachments()
            # PreCompact fires only when a proactive fold is actually imminent (the gate
            # predicate mirrors auto_compact's own check). A hook may block the fold this
            # turn or inject grounding; PostCompact fires afterwards with the new summary.
            skip_compaction = False
            if self.compression.should_compact(
                messages, model=self.config.model, token_estimator=self._estimate_tokens
            ):
                pre = await self._run_pre_compact_hooks(messages, "auto")
                skip_compaction = pre.block
            if not skip_compaction:
                before_compaction = list(messages)
                messages, events = await self.compression.auto_compact(
                    messages,
                    model=self.config.model,
                    token_estimator=self._estimate_tokens,
                    summarizer=self._summarizer,
                    on_stage=self._compaction_reporter(reactive=False),
                    attachments=attachments,
                )
                self._active_messages = messages
                for event in events:
                    await self.logger.write("compression", asdict(event))
                await self._commit_compaction_boundary(before_compaction, messages)
                if events:
                    await self._run_post_compact_hooks(messages, before_compaction, "auto")

            # Stream tokens to the UI only when it is live and streaming is enabled.
            sink = self.ui if (self.ui.is_live and self.config.stream) else None
            self.ui.on_turn_start()
            try:
                self._sync_permission_mode_context(messages)
                result = await self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                    should_cancel=cancelled,
                )
            except asyncio.CancelledError:
                if cancelled():
                    return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
                raise
            except LLMContextTooLongError as exc:
                # Reactive recovery, bounded so a 413 can never loop forever: summarize
                # aggressively once, then retry ``complete`` up to MAX_PTL_RETRIES times,
                # peeling the oldest whole API rounds before each retry. If a retry still
                # 413s and nothing is left to drop (< 2 rounds), or the retries are
                # exhausted, the overflow propagates.
                gap = parse_prompt_too_long_gap(str(exc))
                before_compaction = list(messages)
                # PreCompact fires for observability/grounding; its block flag is IGNORED
                # here — reactive compaction recovers from an actual overflow and must run.
                await self._run_pre_compact_hooks(messages, "reactive")
                messages, events = await self.compression.reactive_compact(
                    messages,
                    model=self.config.model,
                    token_estimator=self._estimate_tokens,
                    summarizer=self._summarizer,
                    on_stage=self._compaction_reporter(reactive=True),
                    attachments=self._build_read_attachments(),
                )
                self._active_messages = messages
                for event in events:
                    await self.logger.write("compression", {**asdict(event), "reactive": True})
                await self._commit_compaction_boundary(before_compaction, messages)
                if events:
                    await self._run_post_compact_hooks(messages, before_compaction, "reactive")
                result = None
                for _ in range(MAX_PTL_RETRIES):
                    self.ui.on_turn_start()
                    try:
                        self._sync_permission_mode_context(messages)
                        result = await self.provider.complete(
                            messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                            should_cancel=cancelled,
                        )
                        break
                    except asyncio.CancelledError:
                        if cancelled():
                            return await self._stopped(
                                messages, step, "interrupted", "being interrupted by the user (Esc)"
                            )
                        raise
                    except LLMContextTooLongError as exc_retry:
                        gap = parse_prompt_too_long_gap(str(exc_retry)) or gap
                        truncated = truncate_head_for_ptl_retry(
                            messages, token_gap=gap, token_estimator=self._estimate_tokens
                        )
                        if truncated is not None:
                            messages = truncated
                            self._active_messages = messages
                            await self.logger.write(
                                "compression",
                                {"stage": "ptl_head_truncate", "reactive": True, "kept": len(messages)},
                            )
                            continue
                        # No whole round is safe to drop (< 2 rounds) — a single oversized
                        # round/message is the whole overflow. Last resort: head/tail-truncate
                        # the largest non-preserved messages so the prompt finally fits. We
                        # must shed at least the known gap (else a fraction of the estimate).
                        need = gap or max(1, self._estimate_tokens(messages) // 5)
                        shrunk = shrink_oversize_messages(
                            messages, tokens_to_drop=need, token_estimator=self._estimate_tokens
                        )
                        if shrunk is None:
                            # Even the largest messages are already at their floor — nothing
                            # left to shrink. Surface the overflow rather than spin.
                            raise
                        messages = shrunk
                        self._active_messages = messages
                        await self.logger.write(
                            "compression",
                            {"stage": "ptl_shrink", "reactive": True, "kept": len(messages)},
                        )
                if result is None:
                    # Exhausted MAX_PTL_RETRIES without a successful completion.
                    raise exc
            except Exception:
                # A fast request may already have transparently fallen back before the
                # normal-speed retry failed. Reflect that session-state transition even
                # though this provider call itself did not complete.
                self._consume_fast_mode_fallback()
                raise

            # Track the running prompt token count from the response usage, when the
            # provider reports it, so the next turn's auto-compact gate thresholds against
            # real usage (parity with the reference) rather than only a char estimate.
            # Also accumulate session-wide input/output totals for the chat ``/cost`` and
            # ``/status`` commands (cheap counters; not part of any contract).
            if result is None:
                raise RuntimeError("provider returned no result after context recovery")
            self._consume_fast_mode_fallback()
            if result.usage is not None:
                # ``total_tokens`` (prompt incl. cache + this turn's output) is the anchor
                # the gate carries forward — once this turn is in history its output counts
                # toward the next request's prompt.
                self._last_usage_tokens = result.usage.total_tokens
                self._session_input_tokens += result.usage.input_tokens
                self._session_output_tokens += result.usage.output_tokens
                self._run_input_tokens += result.usage.input_tokens
                self._run_output_tokens += result.usage.output_tokens
                self._run_context_tokens = result.usage.context_tokens

            # Re-poll after the turn completes: an Esc pressed *during* the model
            # call (including a single-turn final answer that requests no tools, and
            # the non-streaming path where deltas can't be polled) is honored here at
            # the next safe point instead of being silently swallowed.
            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")

            # Emit the previous tool batch's progress label here — the model call just
            # finished, so the background Haiku task fired last turn has had that whole
            # call to resolve (near-zero added latency). This runs before both the final-
            # answer return and the next tool execution, so neither path drops a label.
            await self._flush_pending_tool_use_summary()

            await self.logger.write(
                "llm",
                {
                    "content": result.content,
                    "tool_calls": [asdict(tool_call) for tool_call in result.tool_calls],
                    "stop_reason": result.stop_reason,
                },
            )
            if result.thinking:
                self.ui.on_thinking(result.thinking)

            tool_call_payloads = [asdict(tool_call) for tool_call in result.tool_calls]
            assistant_metadata: dict[str, object] = {}
            if tool_call_payloads:
                assistant_metadata["tool_calls"] = tool_call_payloads
            # Preserve the raw thinking blocks so the provider can replay them on the
            # next turn (required by the API when thinking and tool use span turns).
            if result.thinking_blocks:
                assistant_metadata["thinking_blocks"] = result.thinking_blocks
            # Preserve provider-owned opaque state so the provider can replay future turns
            # without leaking protocol-specific shapes into the core loop.
            if result.provider_state:
                assistant_metadata["provider_state"] = result.provider_state
            # Anchor this turn's full token footprint on the message itself so the gate
            # estimate can walk back to it and add only the rough cost of messages added
            # since (ports the reference's per-message ``tokenCountWithEstimation``).
            if result.usage is not None:
                assistant_metadata["usage_tokens"] = result.usage.total_tokens
            await self._emit(messages, Message("assistant", result.content, metadata=assistant_metadata))

            # Surface the running token usage to a live UI, once per turn (the only
            # granularity available — usage is known only after the response). Lands right
            # under the streamed assistant block; NullUI ignores it.
            if result.usage is not None:
                # Split the real prompt total into the per-run baseline (system prompt,
                # gitStatus, recall, pinned CLAUDE.md/userContext, tool schemas) vs. what the
                # conversation itself contributes, so a fresh or /clear'd session reads ~0 chat.
                conversation = min(
                    self._conversation_token_estimate(messages), self._run_context_tokens
                )
                self.ui.on_token_usage(
                    {
                        "context_tokens": self._run_context_tokens,
                        "conversation_tokens": conversation,
                        "window": tokens.context_window_for_model(self.config.model),
                        "input_tokens": self._run_input_tokens,
                        "output_tokens": self._run_output_tokens,
                    }
                )

            # PostSampling: fire-and-forget after the assistant turn is in the history, so
            # an observational hook (e.g. background extraction) overlaps the next model
            # call instead of blocking the loop. Awaited only at terminal returns.
            self._fire_post_sampling(messages, result.content)

            # Natural termination: the model stopped requesting tools, so this is the answer.
            if not result.tool_calls:
                # Stop hook: a hook may BLOCK this stop and force the loop to keep running
                # ("可阻断/可续跑"), bounded by config.max_stop_blocks so it can't loop
                # forever. A block injects the continuation directive and re-enters the loop.
                stop = await self._run_stop_hooks(messages, result.content, stop_blocks)
                if stop.block and stop_blocks < self.config.max_stop_blocks:
                    stop_blocks += 1
                    directive = (
                        stop.additional_context
                        or stop.reason
                        or "A stop hook requested that you keep working instead of stopping."
                    )
                    await self._emit(
                        messages,
                        Message(
                            "user",
                            f"<system-reminder>\n{directive}\n</system-reminder>",
                            metadata={"stop_hook": "continue"},
                        ),
                    )
                    await self.logger.write(
                        "hook",
                        {"event": "Stop", "decision": "continue", "block_count": stop_blocks,
                         "reason": stop.reason},
                    )
                    self.ui.on_reasoning(result.content)
                    continue
                await self._reap_background_hooks()
                self.ui.on_final(result.content)
                self._emit_recap(messages, step, "completed")
                await self.logger.write("final", {"answer": result.content})
                await self._extract_memories(messages)
                return AgentRunResult(result.content, messages, step, self.logger.run_id)

            # Intermediate turn: show the reasoning that precedes the tool calls.
            self.ui.on_reasoning(result.content)

            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
            tool_results = await self.executor.execute_many(
                result.tool_calls,
                should_cancel=cancelled,
                messages=messages,
            )
            for tool_call, tool_result in zip(result.tool_calls, tool_results, strict=True):
                observation = f"{tool_result.name}: {tool_result.content}"
                await self._emit(
                    messages,
                    Message(
                        "tool",
                        observation,
                        name=tool_result.name,
                        metadata={**tool_result.metadata, "ok": tool_result.ok, "tool_call_id": tool_call.id},
                    ),
                )
                # Record read-file state HERE (not in the read tool — its mixin shape
                # conflicts with SessionAwareMixin) so it can be re-injected after a
                # post-compaction fold. Defensive: odd/missing args just skip.
                self._record_read_result(tool_call, tool_result)
                await self._notify_lsp_edit(tool_call, tool_result)

            # Interactive input entered while tools were running is inserted only after
            # the complete tool-result batch, immediately before the next provider call.
            # These messages have already been classified by the host as ordinary
            # prompts (slash commands remain queued), and intentionally bypass
            # UserPromptSubmit hooks because this is a current-turn steering seam.
            if midturn_drain is not None:
                for queued_message in midturn_drain():
                    await self._emit(messages, queued_message)
                    await self.logger.write(
                        "queued_prompt",
                        {
                            "content": queued_message.content,
                            "delivery": "midturn",
                            **self._trace_fields(),
                        },
                    )

            # Fire (fire-and-forget) the tool-use progress label for this batch. It runs in
            # the background during next turn's model call and is awaited/emitted there.
            self._fire_tool_use_summary(
                list(zip(result.tool_calls, tool_results, strict=True)), result.content
            )

    async def run_messages(
        self,
        user_messages: list[Message],
        should_cancel: Callable[[], bool] | None = None,
        deadline: float | None = None,
        history: list[Message] | None = None,
        midturn_drain: Callable[[], list[Message]] | None = None,
    ) -> AgentRunResult:
        """Run a between-turn batch while preserving each prompt's UUID and hooks."""

        if not user_messages:
            raise ValueError("at least one user message is required")
        return await self.run(
            user_messages[0].content,
            should_cancel=should_cancel,
            deadline=deadline,
            history=history,
            midturn_drain=midturn_drain,
            _user_messages=user_messages,
        )

    def _emit_recap(self, messages: list[Message], step: int, reason: str) -> None:
        duration = time.monotonic() - getattr(self, "_run_start_time", time.monotonic())
        tool_counts: dict[str, int] = {}
        for msg in messages:
            if msg.role == "tool" and msg.name:
                tool_counts[msg.name] = tool_counts.get(msg.name, 0) + 1
        stats = {
            "duration": duration,
            "steps": step,
            "reason": reason,
            "tool_counts": tool_counts,
            "input_tokens": self._run_input_tokens,
            "output_tokens": self._run_output_tokens,
            "context_tokens": self._run_context_tokens,
        }
        self.ui.on_run_completed(stats)

    async def _emit(self, messages: list[Message], message: Message) -> None:
        """Append a real conversation turn: link it into the chain and persist it.

        The transcript is the faithful, append-only record of the conversation; ``uuid``/
        ``parent_uuid`` chain each turn to its predecessor. Compaction is a separate,
        in-memory-only optimization — its summary messages never come through here, so the
        transcript always reflects the true history and a resume reconstructs it intact
        (the live loop re-compacts as needed). Persistence is best-effort and never raises.
        """
        message.parent_uuid = self._last_message_uuid
        messages.append(message)
        if self.transcript is not None:
            await self.transcript.append_message(message)
        self._last_message_uuid = message.uuid

    async def _commit_compaction_boundary(
        self, before: list[Message], after: list[Message]
    ) -> None:
        """Persist a compaction boundary when a context-collapse fold actually happened.

        Mirrors the reference: the new summary becomes a transcript root
        (``parent_uuid=None`` + ``compact_boundary`` tag), the kept tail's first message is
        relinked onto it, and post-compact file attachments are chained on — so a
        ``--resume`` loads only the *compacted* state (turns after the last boundary), not
        the full pre-fold history. The fold is detected by diffing message uuids, so
        snip/microcompact (which truncate content in place, same uuids) are correctly
        ignored. No-op when persistence or the boundary feature is off; best-effort and
        decoupled from the loop's correctness, exactly like ``_emit``.
        """
        if self.transcript is None or not self.config.persist_compaction_boundary:
            return
        before_uuids = {m.uuid for m in before}
        new_msgs = [m for m in after if m.uuid not in before_uuids]
        if not new_msgs:
            return  # snip/microcompact only, or nothing changed — no boundary to write.
        summary = next((m for m in new_msgs if is_summary_message(m)), None)
        if summary is None:
            # A drop with no summary (e.g. emergency PTL head-truncation): don't write a
            # boundary — let a resume reload the full history and re-compact.
            return

        # The summary becomes a new root; record the real predecessor for forensics.
        if self._last_message_uuid is not None:
            summary.metadata["logical_parent_uuid"] = self._last_message_uuid
        summary.metadata["compact_boundary"] = True
        summary.parent_uuid = None
        await self.transcript.append_message(summary)

        # Everything after the summary in the folded list is either kept tail (already on
        # disk, uuid in ``before``) or new attachments. Preserved front matter sits before
        # the summary, so it is excluded here.
        tail = after[after.index(summary) + 1 :]
        recent = [m for m in tail if m.uuid in before_uuids]
        attachments = [m for m in tail if m.uuid not in before_uuids]

        if recent:
            # Re-point the kept tail's head onto the summary via an append-only relink
            # (the original line can't be mutated); keep the in-memory chain in sync.
            recent[0].parent_uuid = summary.uuid
            await self.transcript.append_relink(recent[0].uuid, summary.uuid)
            running = recent[-1].uuid
        else:
            running = summary.uuid

        for attachment in attachments:
            attachment.parent_uuid = running
            await self.transcript.append_message(attachment)
            running = attachment.uuid

        self._last_message_uuid = running

    async def _stopped(self, messages: list[Message], step: int, reason: str, human: str) -> AgentRunResult:
        """Shared exit path for run interruption (cancel / max_steps / deadline)."""
        self._emit_recap(messages, step, reason)
        await self._cancel_pending_tool_use_summary()
        await self._reap_background_hooks()
        answer = f"Stopped after {human} without a final answer."
        self.ui.on_stopped(reason, human)
        await self.logger.write("final", {"answer": answer, "stopped": reason})
        return AgentRunResult(answer, messages, step, self.logger.run_id)

    def _fire_tool_use_summary(
        self, batch: list[tuple[ToolCall, ToolResult]], last_assistant_text: str
    ) -> None:
        """Kick off (fire-and-forget) the async progress label for a finished tool batch.

        No-op unless a summarizer exists (feature on + real provider), the UI is live (no
        one to show a label to otherwise — don't waste an API call), and this is the leader
        (or sub-agent labels are explicitly enabled). The created task is awaited and emitted
        next turn by ``_flush_pending_tool_use_summary``.
        """
        if self._tool_use_summarizer is None or not self.ui.is_live or not batch:
            return
        if self.session.depth != 0 and not self.config.tool_use_summary.include_subagents:
            return
        self._pending_tool_use_names = [call.name for call, _ in batch]

        async def summarize() -> str | None:
            assert self._tool_use_summarizer is not None
            return await self._tool_use_summarizer(batch, last_assistant_text)

        self._pending_tool_use_summary = asyncio.create_task(summarize())

    async def _flush_pending_tool_use_summary(self) -> None:
        """Await the pending label task and emit it to the UI + event log (never the API).

        Best-effort: any failure (including the task having degraded to ``None``) just drops
        the label — it must never sink a run. The label is observability only; it is written
        to ``runs/*.jsonl`` but never to the transcript or the API ``messages``.
        """
        task = self._pending_tool_use_summary
        if task is None:
            return
        names = self._pending_tool_use_names
        self._pending_tool_use_summary = None
        self._pending_tool_use_names = []
        try:
            label = await task
        except Exception:  # noqa: BLE001 - a missing label is non-fatal.
            return
        if not label:
            return
        self.ui.on_tool_use_summary(label, names)
        await self.logger.write("tool_use_summary", {"label": label, "tools": names})

    async def _cancel_pending_tool_use_summary(self) -> None:
        """Cancel and reap the in-flight label task without emitting it (used on abort)."""
        task = self._pending_tool_use_summary
        if task is None:
            return
        self._pending_tool_use_summary = None
        self._pending_tool_use_names = []
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - reap quietly (incl. CancelledError).
            pass

    # --- Lifecycle hook seams --------------------------------------------------
    #
    # Each fires a class of programmatic hook at a specific loop boundary. All are
    # best-effort observability/steering seams: a missing hook list makes them cheap
    # no-ops, and (except for an explicit block decision) they never alter control
    # flow. ``self.hooks`` is the same pipeline the executor uses for tool hooks.

    async def _run_user_prompt_hooks(
        self,
        messages: list[Message],
        user_message: Message,
        task: str,
        *,
        blocked_as_warning: bool = False,
    ) -> "tuple[AgentRunResult | None, bool]":
        """Fire UserPromptSubmit, then chain + persist the (possibly neutralized) task.

        Returns a terminal result if a hook blocked the run, else ``None`` (after persisting
        the task and injecting any additional_context for the model to see). Persistence happens
        here, *after* the hook, so a ``transformed_prompt`` rewrite is what gets recorded + sent;
        with no user-prompt hooks the task is simply persisted unchanged.
        """
        outcome = HookOutcome()
        if self.hooks.user_prompt_hooks:
            ctx = HookContext(
                event=HookEvent.USER_PROMPT_SUBMIT,
                messages=messages,
                session_id=self.session_id,
                prompt=task,
            )
            outcome = await self.hooks.run_user_prompt(ctx)
            if outcome.transformed_prompt is not None:
                user_message.content = outcome.transformed_prompt
        if self.hooks.user_prompt_hooks:
            await self.logger.write(
                "hook",
                {
                    "event": "UserPromptSubmit",
                    "blocked": outcome.block,
                    "reason": outcome.reason,
                    **(outcome.metadata or {}),
                },
            )
        if outcome.block and blocked_as_warning:
            messages.remove(user_message)
            reason = outcome.reason or "Request blocked by a UserPromptSubmit hook."
            await self._emit(
                messages,
                Message(
                    "system",
                    f"Queued prompt was blocked by UserPromptSubmit: {reason}",
                    metadata={
                        "hook_blocked_prompt": True,
                        "reason": reason,
                        "original_prompt": task,
                    },
                ),
            )
            self.ui.on_stopped("blocked_queued_prompt", reason)
            return None, True
        # Chain + persist the new task (parent = last history message, or None).
        user_message.parent_uuid = self._last_message_uuid
        if self.transcript is not None:
            await self.transcript.append_message(user_message)
        self._last_message_uuid = user_message.uuid
        if not self.hooks.user_prompt_hooks:
            return None, False
        if outcome.block:
            answer = outcome.reason or "Request blocked by a UserPromptSubmit hook."
            self._emit_recap(messages, 0, "blocked")
            self.ui.on_stopped("blocked", answer)
            await self.logger.write("final", {"answer": answer, "stopped": "blocked"})
            return AgentRunResult(answer, messages, 0, self.logger.run_id), True
        if outcome.additional_context:
            await self._emit(
                messages,
                Message(
                    "user",
                    f"<system-reminder>\n{outcome.additional_context}\n</system-reminder>",
                    metadata={"hook": "user_prompt_context"},
                ),
            )
        return None, False

    async def _run_pre_compact_hooks(self, messages: list[Message], trigger: str) -> HookOutcome:
        """Fire PreCompact. The caller honors ``block`` only on the proactive (``auto``)
        path; on the forced ``reactive`` path the block is ignored (compaction is mandatory)."""
        if not self.hooks.pre_compact_hooks:
            return HookOutcome()
        ctx = HookContext(
            event=HookEvent.PRE_COMPACT,
            messages=messages,
            session_id=self.session_id,
            trigger=trigger,
        )
        outcome = await self.hooks.run_pre_compact(ctx)
        await self.logger.write(
            "hook",
            {"event": "PreCompact", "trigger": trigger, "blocked": outcome.block,
             "reason": outcome.reason},
        )
        return outcome

    async def _run_post_compact_hooks(
        self, messages: list[Message], before: list[Message], trigger: str
    ) -> None:
        """Fire PostCompact with the new summary (when a prefix fold produced one) and
        inject any returned additional_context to ground the next turn."""
        if not self.hooks.post_compact_hooks:
            return
        before_uuids = {m.uuid for m in before}
        new_msgs = [m for m in messages if m.uuid not in before_uuids]
        summary = next((m.content for m in new_msgs if is_summary_message(m)), None)
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=messages,
            session_id=self.session_id,
            trigger=trigger,
            summary=summary,
        )
        outcome = await self.hooks.run_post_compact(ctx)
        await self.logger.write(
            "hook",
            {"event": "PostCompact", "trigger": trigger, "has_summary": summary is not None},
        )
        if outcome.additional_context:
            await self._emit(
                messages,
                Message(
                    "user",
                    f"<system-reminder>\n{outcome.additional_context}\n</system-reminder>",
                    metadata={"hook": "post_compact_context"},
                ),
            )

    async def _run_stop_hooks(
        self, messages: list[Message], last_assistant_text: str, stop_blocks: int
    ) -> HookOutcome:
        """Fire Stop at natural termination. A returned ``block`` asks the loop to keep
        running; ``stop_hook_active`` tells the hook it has already blocked at least once."""
        if not self.hooks.stop_hooks:
            return HookOutcome()
        ctx = HookContext(
            event=HookEvent.STOP,
            messages=messages,
            session_id=self.session_id,
            last_assistant_message=last_assistant_text,
            stop_hook_active=stop_blocks > 0,
        )
        outcome = await self.hooks.run_stop(ctx)
        await self.logger.write(
            "hook",
            {"event": "Stop", "blocked": outcome.block, "stop_hook_active": stop_blocks > 0,
             "reason": outcome.reason},
        )
        return outcome

    def _fire_post_sampling(self, messages: list[Message], last_assistant_text: str) -> None:
        """Schedule PostSampling fire-and-forget on a snapshot of the history."""
        if not self.hooks.post_sampling_hooks:
            return
        ctx = HookContext(
            event=HookEvent.POST_SAMPLING,
            messages=list(messages),
            session_id=self.session_id,
            last_assistant_message=last_assistant_text,
        )
        task = asyncio.create_task(self._post_sampling_runner(ctx))
        self._background_hook_tasks.add(task)
        task.add_done_callback(self._background_hook_tasks.discard)

    async def _post_sampling_runner(self, ctx: HookContext) -> None:
        try:
            await self.hooks.run_post_sampling(ctx)
        except Exception as exc:  # noqa: BLE001 - observational; must never sink a run
            await self.logger.write(
                "hook", {"event": "PostSampling", "error": f"{type(exc).__name__}: {exc}"}
            )

    async def _fire_observational(
        self, event: HookEvent, detail: dict[str, object], messages: list[Message]
    ) -> None:
        """Fire one C5 observational event: awaited, fail-open, always JSONL-logged.

        Subscribers run to completion (so an external watcher's side effect lands
        before the loop moves on), but any failure is reduced to a log field — these
        events never alter control flow.
        """
        runner = {
            HookEvent.SESSION_START: self.hooks.run_session_start,
            HookEvent.SESSION_END: self.hooks.run_session_end,
            HookEvent.SUBAGENT_START: self.hooks.run_subagent_start,
            HookEvent.SUBAGENT_STOP: self.hooks.run_subagent_stop,
        }[event]
        ctx = HookContext(
            event=event, messages=messages, session_id=self.session_id, detail=detail
        )
        error: str | None = None
        try:
            await runner(ctx)
        except Exception as exc:  # noqa: BLE001 - observational; must never sink a run
            error = f"{type(exc).__name__}: {exc}"
        payload: dict[str, object] = {"event": event.value, **detail}
        if error:
            payload["error"] = error
        await self.logger.write("hook", payload)

    async def fire_session_end(self, reason: str = "host_shutdown") -> None:
        """Fire SessionEnd (observational). The HOST owns the session boundary:
        the CLI calls this after a one-shot run and at chat exit; library embedders
        call it when their session concept closes. No-op before the first run()."""
        await self.process_supervisor.shutdown()
        if self.session.scheduler_store is not None:
            await asyncio.to_thread(
                self.session.scheduler_store.delete_session_jobs,
                self.session.session_id,
                self.session.agent_id,
            )
        lsp_manager = self.session.lsp_manager
        if lsp_manager is not None:
            close = getattr(lsp_manager, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        if not self._session_start_fired:
            return
        self._session_start_fired = False
        await self._fire_observational(
            HookEvent.SESSION_END,
            {"run_id": self.logger.run_id, "reason": reason, **self._trace_fields()},
            [],
        )

    def _scheduler_store(self) -> SchedulerStore | None:
        if not self.config.tools.scheduler.enabled:
            return None
        if self.session.scheduler_store is None:
            config = self.config.tools.scheduler
            self.session.scheduler_store = SchedulerStore(
                config.database_path(), max_jobs=config.max_jobs,
                max_prompt_chars=config.max_prompt_chars,
            )
        return self.session.scheduler_store

    async def scheduler_heartbeat(self, *, ttl: float = 120) -> None:
        """Publish session liveness and make one missed recurring delivery eligible."""
        try:
            store = self._scheduler_store()
        except Exception as exc:  # noqa: BLE001 - scheduling is observational to an agent run
            await self._audit_event(
                "scheduler_delivery", {"state": "store_unavailable", "error": str(exc)}
            )
            return
        if store is None:
            return
        try:
            catchups = await asyncio.to_thread(
                store.heartbeat, self.session.session_id, self.session.agent_id, ttl=ttl
            )
            missed_one_shots = await asyncio.to_thread(
                store.missed_one_shots, self.session.session_id, self.session.agent_id
            )
        except Exception as exc:  # noqa: BLE001 - scheduling cannot sink the foreground turn
            await self._audit_event(
                "scheduler_delivery", {"state": "heartbeat_error", "error": str(exc)}
            )
            return
        for delivery in catchups:
            await self._audit_event("scheduler_missed", delivery)
        for job in missed_one_shots:
            job_id = str(job["id"])
            if self.session.ask_user is None:
                if job_id not in self._reported_missed_jobs:
                    await self._audit_event(
                        "scheduler_missed",
                        {"job_id": job_id, "one_shot": True, "state": "confirmation_required"},
                    )
                    self._reported_missed_jobs.add(job_id)
                continue
            answers = await self.session.ask_user([{
                "id": f"missed-{job_id}",
                "question": f"One-shot job {job_id} was missed. Run it now?",
                "options": [
                    {"label": "Run now", "description": "Queue the missed prompt once."},
                    {"label": "Discard", "description": "Delete it without running."},
                ],
            }])
            answer = str(answers[0].get("answer", "")) if answers else ""
            deliver = answer.strip().casefold() in {"run now", "run", "yes", "y"}
            await asyncio.to_thread(
                store.resolve_missed_one_shot, job_id, deliver=deliver
            )
            await self._audit_event(
                "scheduler_missed",
                {"job_id": job_id, "one_shot": True,
                 "state": "queued" if deliver else "discarded"},
            )

    async def drain_scheduler_deliveries(
        self, history: list[Message] | None,
    ) -> tuple[list[Message], list[AgentRunResult]]:
        """Run queued prompts serially after the current complete agent turn."""
        store = self.session.scheduler_store
        current = list(history or [])
        results: list[AgentRunResult] = []
        if store is None:
            return current, results
        pending = await asyncio.to_thread(
            store.pending, self.session.session_id, self.session.agent_id
        )
        for delivery in pending[:10]:
            await self._audit_event(
                "scheduler_delivery",
                {"delivery_id": delivery["id"], "job_id": delivery["job_id"], "state": "started"},
            )
            result = await self.run(str(delivery["prompt"]), history=current or None)
            current = result.messages
            results.append(result)
            await asyncio.to_thread(store.complete_delivery, int(str(delivery["id"])))
            await self._audit_event(
                "scheduler_delivery",
                {"delivery_id": delivery["id"], "job_id": delivery["job_id"], "state": "completed"},
            )
        return current, results

    async def _reap_background_hooks(self) -> None:
        """Await any in-flight PostSampling tasks at a terminal return (best-effort)."""
        tasks = list(self._background_hook_tasks)
        self._background_hook_tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _recall(self, task: str, messages: list[Message]) -> None:
        """Inject relevant past memories as a pinned system block before the task."""
        if self.retriever is None:
            return
        recalled = await self.retriever.recall(task)
        if not recalled:
            return
        block = self.retriever.format_block(recalled)
        # Right after the main system prompt, before the user task, tagged so
        # extraction skips it and context_collapse keeps it pinned.
        messages.insert(1, Message("system", block, metadata={"memory": "recall"}))
        await self.logger.write("memory_recall", {"count": len(recalled), "ids": [r.id for r in recalled]})

    async def _inject_project_context(self, messages: list[Message]) -> None:
        """Assemble run-start context the reference Open-ClaudeCode way.

        Runs right after ``_recall``, which already put any recall block at index 1, so
        messages are ``[system, (recall), user]`` here. We build two seams:

        - ``system_context`` ``{"gitStatus": <git block>}`` — appended to the *base*
          system message (``messages[0]``) as ``key: value`` lines via
          ``append_system_context``. Git thus rides inside the single system block
          (always preserved by compaction), not as a standalone system message.
        - ``user_context`` ``{"claudeMd": <CLAUDE.md>, "currentDate": <today>}`` —
          rendered as ONE pinned ``<system-reminder>`` user message via
          ``prepend_user_context`` and inserted immediately before the user task.

        Final order becomes ``system(+gitStatus) → (memory recall system) →
        userContext <system-reminder> user (pinned) → user task``.

        Each source is independently best-effort: a failure in one (or its absence)
        degrades to no injection for that part and never sinks the run. Log event
        names/sizes are unchanged (``git_status`` / ``project_instructions``).

        NOTE for Phase 2B: the userContext message is a *pinned user* message.
        Compaction must preserve pinned messages regardless of role (``_context_collapse``
        will enforce this); the ``pinned`` tag is the seam for that.
        """
        system_context: dict[str, str] = {}
        if self.config.git_context:
            try:
                git_block = await build_git_status(self.session.workspace)
            except Exception as exc:  # noqa: BLE001 - build_git_status shouldn't raise; defensive.
                await self.logger.write("git_status", {"error": f"{type(exc).__name__}: {exc}"})
                git_block = None
            if git_block:
                system_context["gitStatus"] = git_block
                await self.logger.write("git_status", {"chars": len(git_block)})

        if system_context:
            base = messages[0]
            messages[0] = Message(
                "system",
                append_system_context(base.content, system_context),
                metadata=base.metadata,
            )

        user_context: dict[str, str] = {}
        if self.config.project_instructions:
            try:
                text = await build_project_instructions(
                    self.session.workspace, max_chars=self.config.claudemd_max_chars
                )
            except Exception as exc:  # noqa: BLE001 - injection must not fail a run
                await self.logger.write("project_instructions", {"error": f"{type(exc).__name__}: {exc}"})
                text = None
            if text:
                user_context["claudeMd"] = text
                await self.logger.write("project_instructions", {"chars": len(text)})

        # currentDate always rides in userContext (cheap, stdlib-only).
        user_context["currentDate"] = current_date_line()

        meta = prepend_user_context(user_context)
        if meta is not None:
            # Insert immediately before the user task message (the last message at this
            # point). Memory recall, if any, stays its own pinned system message ahead
            # of this one.
            messages.insert(len(messages) - 1, meta)

    async def _extract_memories(self, messages: list[Message]) -> None:
        """Extraction at natural termination — goes through ``complete`` (and thus
        the shared ``GatedProvider``) without blocking the event loop. Best-effort; never
        raises: a failed extraction must not sink an otherwise completed run."""
        if self.extractor is None or not self.config.memory.auto_extract:
            return
        try:
            stored = await self.extractor.extract(messages, source_run_id=self.logger.run_id)
        except Exception as exc:  # noqa: BLE001 - extraction must not fail a finished run
            await self.logger.write("memory_extract", {"error": f"{type(exc).__name__}: {exc}"})
            return
        if stored:
            await self.logger.write("memory_extract", {"count": len(stored), "ids": [r.id for r in stored]})

    def _compaction_reporter(self, reactive: bool) -> Callable[[int, int, CompressionEvent], None]:
        """Build the per-stage callback that drives the UI's compaction progress bar.

        Only fires when compaction actually runs (the threshold gate lives in
        ``auto_compact``). Accumulates the overall before/after size and the
        non-empty stage details, emitting start → progress* → end. ``NullUI`` makes
        all three hooks no-ops, so non-interactive runs stay silent."""
        state: dict[str, Any] = {"started": False, "before": 0, "after": 0, "details": []}

        def on_stage(done: int, total: int, event: CompressionEvent) -> None:
            if not state["started"]:
                self.ui.on_compaction_start(reactive)
                state["started"] = True
                state["before"] = event.before_chars
            state["after"] = event.after_chars
            if event.detail:
                state["details"].append(event.detail)
            self.ui.on_compaction_progress(done / total, event.stage)
            if done == total:
                self.ui.on_compaction_end(
                    state["before"], state["after"], ", ".join(state["details"]), reactive
                )

        return on_stage

    def _record_read_result(self, tool_call: ToolCall, tool_result: ToolResult) -> None:
        """Record a successful ``read_text_file`` result into the session read-state.

        The key is the workspace-resolved path string (stable across relative spellings);
        the value is the file content snapshot. Re-injection (after a fold) reads this back.
        Fully defensive — any odd/missing argument or resolve failure just skips, never
        raises, so a malformed call can't break the loop. (Deferred-tool delta
        re-announcements are intentionally NOT done here: this framework re-sends every
        tool schema each turn via ``registry.schemas_for_llm()``, so tools are never lost
        after compaction.)
        """
        if tool_call.name != "read_text_file" or not tool_result.ok:
            return
        try:
            raw_path = tool_call.arguments.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                return
            # Skip CLAUDE.md — it is already injected (and pinned) as the userContext
            # system-reminder, so re-attaching it after a fold is pure duplication that
            # wastes the re-injection budget. Mirrors the reference's CLAUDE.md exclusion.
            if Path(raw_path).name == "CLAUDE.md":
                return
            key = str((self.session.workspace / raw_path).resolve())
            self.session.record_read(key, tool_result.content)
            if tool_result.metadata.get("notebook"):
                self.session.notebook_reads[key] = {
                    "sha256": tool_result.metadata.get("fingerprint"),
                    "mtime_ns": tool_result.metadata.get("mtime_ns"),
                    "size": tool_result.metadata.get("size"),
                }
        except Exception:  # noqa: BLE001 - read-state recording is best-effort, never fatal
            return

    async def _notify_lsp_edit(self, tool_call: ToolCall, tool_result: ToolResult) -> None:
        if not tool_result.ok or self.session.lsp_manager is None:
            return
        path_keys = {
            "edit_file": "path", "multi_edit": "path", "write_text_file": "path",
            "apply_patch": "path", "notebook_edit": "notebook_path",
        }
        key = path_keys.get(tool_call.name)
        if key is None:
            return
        raw = tool_call.arguments.get(key)
        if not isinstance(raw, str) or not raw:
            return
        try:
            await self.session.lsp_manager.notify_saved(raw)
        except Exception as exc:  # observational integration: edits already succeeded
            await self.logger.write("lsp_server_state", {"state": "notify_failed", "error": str(exc)})

    def _build_read_attachments(self) -> list[Message]:
        """Build the post-compact file re-injection message from session read-state.

        Takes the most-recently-read files (newest first) up to
        ``post_compact_max_files``, each truncated to ``post_compact_max_chars_per_file``,
        within a total ``post_compact_total_budget_chars`` budget. Returns ``[]`` when
        nothing has been read. Emits ONE combined ``user`` message framed as untrusted
        situational context (avoids role-alternation worries and is cheaper than many
        messages), tagged ``metadata={"post_compact_attachment": True}`` so it is foldable
        conversation in the tail — NOT pinned (pinning would break the preserved-front
        invariant in ``_context_collapse``).
        """
        state = self.session.read_file_state
        if not state:
            return []
        config = self.config.compression
        max_files = config.post_compact_max_files
        # Budgets are token-based (char/4, matching the auto-compact gate); convert to a
        # char ceiling for the actual truncation, which operates on the raw string.
        per_file = config.post_compact_max_tokens_per_file * 4
        total_budget = config.post_compact_total_budget_tokens * 4
        if max_files <= 0 or per_file <= 0 or total_budget <= 0:
            return []

        sections: list[str] = []
        spent = 0
        # Newest-last dict → reverse for newest-first.
        for key, content in reversed(list(state.items())):
            if len(sections) >= max_files:
                break
            rel = self._relativize(key)
            body = content[:per_file]
            if len(content) > per_file:
                body = f"{body}\n[truncated {len(content) - per_file} chars]"
            remaining = total_budget - spent
            if remaining <= 0:
                break
            if len(body) > remaining:
                body = f"{body[:remaining]}\n[truncated to fit budget]"
            sections.append(f"## {rel}\n{body}")
            spent += len(body)
        if not sections:
            return []
        joined = "\n\n".join(sections)
        text = (
            "<system-reminder>\n"
            "Files you read earlier, re-attached after the conversation was compacted. "
            "This is a snapshot — the file may have changed since; re-read it if you need "
            "the current contents.\n"
            f"{joined}\n"
            "</system-reminder>"
        )
        return [Message("user", text, metadata={"post_compact_attachment": True})]

    def _relativize(self, key: str) -> str:
        """Workspace-relative path for the attachment heading; fall back to the raw key."""
        try:
            return str(Path(key).relative_to(self.session.workspace))
        except Exception:  # noqa: BLE001 - outside the workspace or unrelativizable
            return key

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate the prompt token footprint for the auto-compact gate.

        Ports the reference ``tokenCountWithEstimation``: walk back to the most recent
        assistant turn that carries a real per-response token count (anchored in
        ``metadata['usage_tokens']``) and add only a cheap char-based estimate of the
        messages appended since it. So the gate reflects real API usage once a response
        arrives, charges the not-yet-sent delta at ~4 chars/token, and falls back to a
        full rough estimate offline or after a fold drops the anchor.
        """
        for i in range(len(messages) - 1, -1, -1):
            anchor = messages[i].metadata.get("usage_tokens")
            if isinstance(anchor, int):
                return anchor + tokens.rough_token_estimate_for_messages(messages[i + 1:])
        return tokens.rough_token_estimate_for_messages(messages)

    def _conversation_token_estimate(self, messages: list[Message]) -> int:
        """Rough token size of just the conversation, excluding the fixed run-start
        context (system prompt, gitStatus, memory recall, pinned userContext/CLAUDE.md).

        Lets the live gauge separate the per-run baseline overhead from what the
        conversation itself contributes, so a fresh or ``/clear``'d session reads as
        ~0 chat. The baseline predicate mirrors the history-splice skip rule in
        ``run()`` (``role == 'system'`` or the pinned ``user_context`` message);
        ``rough_token_estimate_for_messages`` is per-message additive, so subtracting
        this from the real prompt total leaves the baseline.
        """
        convo = [
            m for m in messages
            if not (m.role == "system" or m.metadata.get("pinned") == "user_context")
        ]
        return tokens.rough_token_estimate_for_messages(convo)

    def _provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_budget=self.config.thinking_budget,
            effort=self.config.effort,
            speed="fast" if self.fast_mode else None,
            stream=self.config.stream,
        )

    def _consume_fast_mode_fallback(self) -> None:
        """Reflect a provider's transparent normal-speed retry in session state."""

        provider = self.provider
        while hasattr(provider, "inner"):
            provider = provider.inner
        consume = getattr(provider, "consume_fast_disabled_reason", None)
        if not callable(consume):
            return
        reason = consume()
        if reason:
            self.fast_mode = False
            print(f"[fast] disabled for this session ({reason}); retried at normal speed.")

    def _build_hook_pipeline(self) -> HookPipeline:
        """Assemble the default HookPipeline from ``config.hooks``.

        Always carries the ``MaxOutputPostHook`` tool post-hook (the prior default). When
        the hook subsystem is enabled, layers in the toggled built-in programmatic
        lifecycle hooks (closing over this run's session/logger) and any config-driven
        external adapters (each appended to the pipeline list for its event). A bad spec or
        a transport whose dependency is unavailable drops that one hook; assembly never
        raises into construction, so a misconfigured ``[hooks]`` table degrades gracefully.
        """
        pipeline = HookPipeline(
            post_hooks=[
                MaxOutputPostHook.from_config(
                    self.config.output, spill_dir=str(Path(self.config.run_dir) / "outputs")
                )
            ]
        )
        hooks_config = self.config.hooks
        if not hooks_config.enabled:
            return pipeline

        builtin = hooks_config.builtin
        if builtin.stop_completion:
            pipeline.stop_hooks.append(StopCompletionHook(self.session))
        if builtin.post_sampling_observer:
            pipeline.post_sampling_hooks.append(PostSamplingObserverHook(self.logger))
        if builtin.compaction_logger:
            # One instance serves both Pre and PostCompact (it implements both methods).
            compaction_hook = CompactionLoggerHook(self.logger)
            pipeline.pre_compact_hooks.append(compaction_hook)
            pipeline.post_compact_hooks.append(compaction_hook)
        if hooks_config.prompt_validation.enabled:
            pipeline.user_prompt_hooks.append(
                PromptValidationHook(hooks_config.prompt_validation)
            )

        for spec in hooks_config.external:
            attr = LIFECYCLE_EVENT_ATTRS.get(spec.event)
            if attr is None:
                continue  # non-lifecycle event (e.g. a tool event) — not handled here.
            try:
                adapter = build_external_adapter(
                    spec,
                    logger=self.logger,
                    provider=self.provider,
                    base_config=self._provider_config(),
                    subagent_factory=self.session.subagent_factory,
                )
            except Exception as exc:  # noqa: BLE001 - a bad spec must not sink construction.
                logger.warning(
                    "dropping external hook %s/%s (failed to build): %s: %s",
                    spec.event, spec.type, type(exc).__name__, exc,
                )
                adapter = None
            if adapter is not None:
                getattr(pipeline, attr).append(adapter)
        return pipeline

    def _trace_fields(self) -> dict[str, object]:
        """Tracing metadata stamped on a run's opening log event.

        Lets concurrent fan-out be reconstructed from ``runs/*.jsonl``: a child's log
        carries the ``parent_run_id`` that spawned it plus its agent/team identity.
        """
        return {
            "agent_name": self.session.agent_name,
            "team_id": self.session.team_id,
            "parent_run_id": self.session.parent_run_id,
        }

    async def _audit_event(self, kind: str, payload: dict[str, object]) -> None:
        await self.logger.write(kind, {**payload, **self._trace_fields()})

    async def _git_capture(self, cwd: Path, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        output, error = await asyncio.wait_for(process.communicate(), 30)
        text = output.decode("utf-8", errors="replace").strip()
        if process.returncode:
            detail = error.decode("utf-8", errors="replace").strip() or text
            raise RuntimeError(f"git {' '.join(args)} failed: {detail[:2000]}")
        return text

    async def _create_agent_worktree(self, prefix: str) -> dict[str, Any]:
        workspace = self.session.workspace.resolve()
        repo = Path(await self._git_capture(workspace, "rev-parse", "--show-toplevel")).resolve()
        base_sha = await self._git_capture(workspace, "rev-parse", "HEAD")
        slug = f"{prefix}-{new_session_id()[:12]}"
        root = (repo / self.config.tools.worktree.root).resolve()
        path = (root / slug).resolve()
        if path.parent != root or path.exists():
            raise RuntimeError(f"unsafe agent worktree path: {path}")
        branch = f"polaris/ephemeral/{slug}"
        root.mkdir(parents=True, exist_ok=True)
        await self._git_capture(repo, "worktree", "add", "-b", branch, str(path), base_sha)
        return {"path": path, "branch": branch, "base_sha": base_sha, "repo": repo}

    async def _finalize_agent_worktree(self, state: dict[str, Any]) -> dict[str, object]:
        path = Path(state["path"])
        repo = Path(state["repo"])
        base_sha = str(state["base_sha"])
        status = await self._git_capture(path, "status", "--porcelain=v1")
        commits = int(await self._git_capture(path, "rev-list", "--count", f"{base_sha}..HEAD") or "0")
        diff = await self._git_capture(path, "diff", "--stat", base_sha)
        retained = bool(status) or commits > 0
        if not retained:
            await self._git_capture(repo, "worktree", "remove", str(path))
            await self._git_capture(repo, "branch", "-D", str(state["branch"]))
        return {
            "path": str(path), "branch": state["branch"], "base_sha": base_sha,
            "dirty": bool(status), "new_commits": commits, "diff_summary": diff[:8000],
            "retained": retained,
        }

    def _make_subagent_child(
        self, preset: str, model: str | None = None, workspace: Path | None = None
    ) -> "ReActAgent | str":
        """Build the ``dispatch_agent`` child (or a refusal string at the depth ceiling).

        The child reuses this agent's gated provider and scalar config but gets a
        narrowed tool set (``read_only`` = READ tools; ``full`` = READ+WRITE) and —
        crucially — **never** the ``dispatch_agent`` tool itself, so sub-agents can't
        recurse. A depth ceiling is a second guard. The child runs silently (``NullUI``)
        and writes its own run log, tagged with this agent's run id as parent.

        An optional ``model`` overrides the child's model independently of this agent's
        (None → inherit). Compaction and the provider request shape adapt per-model
        automatically (each child builds its own summarizer / compaction threshold from
        its own config, and the shared provider picks the body shape per call), so a
        single leader can fan out a mix of Haiku/Sonnet/Opus children.
        """
        if self.session.depth >= self.session.max_depth:
            return "[dispatch_agent] max sub-agent depth reached; refusing to spawn deeper."
        if model and not is_model_allowed(self.config.provider, model):
            return unsupported_model_message("dispatch_agent", self.config.provider, model)
        sub_registry = ToolRegistry()
        excluded = {
            "dispatch_agent",
            "skill",
            "team_create",
            "task_create",
            "teammate_spawn",
            "task_update",
            "team_status",
            "team_inbox_read",
            "team_message_send",
        }
        allowed_tools = _FULL_CHILD_TOOLS if preset == "full" else _READ_ONLY_CHILD_TOOLS
        child_workspace = (workspace or self.session.workspace).resolve()
        for tool in default_tools(workspace=child_workspace):
            if getattr(tool, "name", "") in excluded:
                continue  # prevent recursive fan-out and team orchestration from ordinary sub-agents
            if getattr(tool, "name", "") not in allowed_tools:
                continue
            sub_registry.register(tool)
        # Disable memory (no independent recall/extract) and skills (the child never gets
        # the ``skill`` tool, so there's nothing to load) in the child. The child's
        # permission mode comes from the preset, never inherited wholesale — see
        # _child_permission_mode (no escalation via spawning).
        child_config = replace(
            self.config,
            permission=_child_permission_mode(self.config.permission, preset),
            memory=replace(self.config.memory, enabled=False),
            skills=replace(self.config.skills, enabled=False),
        )
        if model:
            child_config = replace(child_config, model=model)
        agent_id = new_session_id()
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=self.team_store,
            ui=NullUI(),
            session_id=self.session_id,
            transcript=self._child_transcript(agent_id),
            managed_policy_provider=self.managed_policy_provider,
            sandbox=self.sandbox if child_workspace == self.session.workspace else None,
            workspace=child_workspace,
        )
        child.session.depth = self.session.depth + 1
        child.session.max_depth = self.session.max_depth
        child.session.parent_run_id = self.logger.run_id
        child.session.agent_id = agent_id
        child.session.parent_agent_id = self.session.agent_id
        child.permissions.is_subagent = True
        child.permissions.parent_mode = PermissionMode(self.config.permission)
        child.permissions.parent_agent_id = self.session.agent_id
        child.permissions.tool_source = ToolCallSource.SUBAGENT
        child.permissions.inherit_scoped_session_grants(
            self.permissions, frozenset(tool.name for tool in child.registry.list())
        )
        return child

    def _child_transcript(self, agent_id: str) -> "TranscriptStore | None":
        """A sidechain transcript for a spawned child, nested under this session.

        Children write to ``{session_id}/subagents/agent-{id}.jsonl`` so their turns are
        preserved but never surface as standalone resumable sessions (``list_sessions``
        only globs top-level ``*.jsonl``). ``None`` when persistence is disabled.
        """
        if self.transcript is None:
            return None
        return TranscriptStore(
            self.config.session_dir, self.session.workspace, self.session_id, agent_id=agent_id
        )

    async def _spawn_subagent(
        self, task: str, preset: str = "read_only", model: str | None = None,
        isolation: str = "shared",
    ) -> str:
        """``dispatch_agent`` factory — awaits the child on the shared event loop.

        Fires SubagentStart/SubagentStop (observational) around the child run; a
        cancellation skips the Stop event (best-effort — never await new work while
        being torn down).
        """
        isolated = await self._create_agent_worktree("agent") if isolation == "worktree" else None
        child = (
            self._make_subagent_child(preset, model, isolated["path"])
            if isolated is not None else self._make_subagent_child(preset, model)
        )
        if isinstance(child, str):
            if isolated is not None:
                await self._finalize_agent_worktree(isolated)
            return child
        # getattr-guarded: the factory seam may be stubbed (tests/embedding) and an
        # observational payload must never be the thing that breaks spawning.
        detail: dict[str, object] = {
            "kind": "subagent",
            "preset": preset,
            "model": model or self.config.model,
            "depth": getattr(getattr(child, "session", None), "depth", None),
            "child_run_id": getattr(getattr(child, "logger", None), "run_id", None),
            "isolation": isolation,
        }
        await self._fire_observational(HookEvent.SUBAGENT_START, detail, [])
        started = time.monotonic()
        try:
            result = await child.run(task, deadline=self._active_deadline)
            drain = getattr(child, "drain_scheduler_deliveries", None)
            _history, scheduled = await drain(result.messages) if drain is not None else ([], [])
            if scheduled:
                result.answer += "\n\n" + "\n\n".join(item.answer for item in scheduled)
        except Exception:
            end = getattr(child, "fire_session_end", None)
            if end is not None:
                await end("subagent_error")
            await self._fire_observational(
                HookEvent.SUBAGENT_STOP,
                {**detail, "ok": False, "duration_s": round(time.monotonic() - started, 3)},
                [],
            )
            if isolated is not None:
                try:
                    await self._finalize_agent_worktree(isolated)
                except Exception as cleanup_exc:  # noqa: BLE001 - retain unknown state fail-closed
                    logger.warning("sub-agent worktree finalization failed: %s", cleanup_exc)
            raise
        finally:
            end = getattr(child, "fire_session_end", None)
            if end is not None:
                await end("subagent_exit")
            child_sandbox = getattr(child, "sandbox", self.sandbox)
            if child_sandbox is not self.sandbox:
                child_sandbox.teardown()
            child_logger = getattr(child, "logger", None)
            if child_logger is not None:
                child_logger.close()
        await self._fire_observational(
            HookEvent.SUBAGENT_STOP,
            {**detail, "ok": True, "duration_s": round(time.monotonic() - started, 3)},
            [],
        )
        if isolated is None:
            return result.answer
        summary = await self._finalize_agent_worktree(isolated)
        return result.answer + "\n\nWorktree summary:\n" + json.dumps(summary, ensure_ascii=False, indent=2)

    async def _make_teammate_child(
        self,
        team_id: str,
        name: str,
        role: str,
        task_id: str | None,
        preset: str,
        model: str | None = None,
        workspace: Path | None = None,
    ) -> "tuple[ReActAgent, str] | str":
        """Build a teammate child and its prompt (or a refusal string at the ceiling).

        An optional ``model`` overrides the teammate's model independently (None →
        inherit), so one team can mix Haiku/Sonnet/Opus teammates; compaction and the
        provider shape adapt per-model automatically.
        """
        if self.session.depth >= self.session.max_depth:
            return "[teammate_spawn] max sub-agent depth reached; refusing to spawn deeper."
        if model and not is_model_allowed(self.config.provider, model):
            return unsupported_model_message("teammate_spawn", self.config.provider, model)

        store = self.team_store
        await store.add_member(team_id, name, role)
        team = await store.get_team(team_id)
        assigned_tasks = [
            task
            for task in await store.list_tasks(team_id)
            if task.get("owner") == name and task.get("status") != "completed"
        ]
        focus_task = await store.get_task(team_id, task_id) if task_id else None

        sub_registry = ToolRegistry()
        excluded = {"dispatch_agent", "skill", "team_create", "task_create", "teammate_spawn", "team_status"}
        allowed_tools = _FULL_CHILD_TOOLS if preset == "full" else _READ_ONLY_CHILD_TOOLS
        child_workspace = (workspace or self.session.workspace).resolve()
        for tool in default_tools(workspace=child_workspace):
            tool_name = getattr(tool, "name", "")
            if tool_name in excluded:
                continue
            if tool_name != "task_update" and tool_name not in allowed_tools:
                continue
            sub_registry.register(tool)
        sub_registry.register(TeamInboxReadTool())
        sub_registry.register(TeamMessageSendTool())

        # Teammates used to run a blanket ``permission="auto"`` — a child-side privilege
        # escalation. Now they get the preset-mapped mode like any child, plus explicit
        # allow rules for the team-coordination tools (which write team state, not the
        # workspace), so a read_only teammate can still claim tasks and report back.
        child_config = replace(
            self.config,
            permission=_child_permission_mode(self.config.permission, preset),
            permission_rules=self.config.permission_rules.merge(_TEAMMATE_COORDINATION_RULES),
            memory=replace(self.config.memory, enabled=False),
            skills=replace(self.config.skills, enabled=False),
        )
        if model:
            child_config = replace(child_config, model=model)
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=store,
            ui=NullUI(),
            session_id=self.session_id,
            transcript=self._child_transcript(new_session_id()),
            managed_policy_provider=self.managed_policy_provider,
            sandbox=self.sandbox if child_workspace == self.session.workspace else None,
            workspace=child_workspace,
        )
        child.session.depth = self.session.depth + 1
        child.session.max_depth = self.session.max_depth
        child.session.agent_name = name
        child.session.team_id = team_id
        child.session.parent_run_id = self.logger.run_id
        child.session.parent_agent_id = self.session.agent_id
        child.permissions.is_subagent = True
        child.permissions.parent_mode = PermissionMode(self.config.permission)
        child.permissions.parent_agent_id = self.session.agent_id
        child.permissions.tool_source = ToolCallSource.TEAM
        child.permissions.inherit_scoped_session_grants(
            self.permissions, frozenset(tool.name for tool in child.registry.list())
        )
        prompt = self._teammate_prompt(team, name, role, focus_task, assigned_tasks)
        return child, prompt

    async def _spawn_teammate(
        self,
        team_id: str,
        name: str,
        role: str,
        task_id: str | None = None,
        preset: str = "read_only",
        model: str | None = None,
        isolation: str = "shared",
    ) -> str:
        """Teammate factory — awaits the teammate turn on the shared event loop.

        Fires SubagentStart/SubagentStop (observational, ``kind="teammate"``) around
        the child run; a cancellation skips the Stop event (best-effort)."""
        isolated = await self._create_agent_worktree("teammate") if isolation == "worktree" else None
        built = await self._make_teammate_child(
            team_id, name, role, task_id, preset, model, isolated["path"] if isolated else None
        )
        if isinstance(built, str):
            if isolated is not None:
                await self._finalize_agent_worktree(isolated)
            return built
        child, prompt = built
        detail: dict[str, object] = {
            "kind": "teammate",
            "team_id": team_id,
            "name": name,
            "preset": preset,
            "model": model or self.config.model,
            "depth": getattr(getattr(child, "session", None), "depth", None),
            "child_run_id": getattr(getattr(child, "logger", None), "run_id", None),
            "isolation": isolation,
        }
        await self._fire_observational(HookEvent.SUBAGENT_START, detail, [])
        started = time.monotonic()
        try:
            result = await child.run(prompt, deadline=self._active_deadline)
            drain = getattr(child, "drain_scheduler_deliveries", None)
            _history, scheduled = await drain(result.messages) if drain is not None else ([], [])
            if scheduled:
                result.answer += "\n\n" + "\n\n".join(item.answer for item in scheduled)
        except Exception:
            end = getattr(child, "fire_session_end", None)
            if end is not None:
                await end("teammate_error")
            await self._fire_observational(
                HookEvent.SUBAGENT_STOP,
                {**detail, "ok": False, "duration_s": round(time.monotonic() - started, 3)},
                [],
            )
            if isolated is not None:
                try:
                    await self._finalize_agent_worktree(isolated)
                except Exception as cleanup_exc:  # noqa: BLE001 - retain unknown state fail-closed
                    logger.warning("teammate worktree finalization failed: %s", cleanup_exc)
            raise
        finally:
            end = getattr(child, "fire_session_end", None)
            if end is not None:
                await end("teammate_exit")
            child_sandbox = getattr(child, "sandbox", self.sandbox)
            if child_sandbox is not self.sandbox:
                child_sandbox.teardown()
            child_logger = getattr(child, "logger", None)
            if child_logger is not None:
                child_logger.close()
        await self._fire_observational(
            HookEvent.SUBAGENT_STOP,
            {**detail, "ok": True, "duration_s": round(time.monotonic() - started, 3)},
            [],
        )
        if isolated is None:
            return result.answer
        summary = await self._finalize_agent_worktree(isolated)
        return result.answer + "\n\nWorktree summary:\n" + json.dumps(summary, ensure_ascii=False, indent=2)

    @staticmethod
    def _teammate_prompt(
        team: dict[str, object],
        name: str,
        role: str,
        focus_task: dict[str, object] | None,
        assigned_tasks: list[dict[str, object]],
    ) -> str:
        task_block = focus_task if focus_task is not None else assigned_tasks
        return (
            f"You are teammate '{name}' in team '{team['id']}'.\n"
            f"Role: {role}\n"
            f"Team goal: {team['goal']}\n"
            f"Leader: {team['leader']}\n"
            "Use team_inbox_read to read your inbox. Work only on tasks assigned to you "
            "or tasks you explicitly claim with task_update. When you finish or become "
            "blocked, call task_update with the new status and then call team_message_send "
            "to notify the leader.\n"
            f"Current task context:\n{json.dumps(task_block, ensure_ascii=False, indent=2, default=str)}"
        )

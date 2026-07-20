"""Slash-command autocompletion for the interactive ``polaris chat`` input.

When the user types ``/`` the prompt_toolkit dropdown lists every built-in command and
user-invocable skill (with descriptions); when they are completing the argument of a
command with a discrete value set it lists those values. For ``/resume`` the candidates
are past sessions shown by their **summary phrase** (title / first prompt), not the raw
UUID — the inserted text is still the session id so ``/resume <id>`` resolves correctly.

The completer holds the live :class:`~agent_core.react.ReActAgent` (stable for the chat
process) so it can read the skill registry and session list at completion time. Every
branch is defensive: any failure yields no completions rather than breaking input.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion

from agent_core.chat_commands import _COMMANDS, _COMMAND_HELP
from agent_core.transcript import list_sessions, project_dir, session_label

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from agent_core.react import ReActAgent
    from agent_core.skills import Skill

# ``/name`` with no space yet → still completing the command/skill name.
_FIRST_TOKEN = re.compile(r"^/(?P<name>\S*)$")
# ``/name <arg...>`` → completing the argument of an already-typed command.
_ARG = re.compile(r"^/(?P<name>\S+)\s+(?P<arg>.*)$")


class SlashCompleter(Completer):
    """prompt_toolkit completer for chat slash-commands, skills, and their arguments."""

    def __init__(self, agent: "ReActAgent") -> None:
        self._agent = agent

    def get_completions(
        self, document: "Document", complete_event: "CompleteEvent"
    ) -> "Iterable[Completion]":
        text = document.text_before_cursor
        # Only complete a single-line buffer that starts with '/': a multi-line message
        # that merely contains '/' must never trigger command completion.
        if "\n" in text or not text.startswith("/"):
            return

        first = _FIRST_TOKEN.match(text)
        if first is not None:
            yield from self._complete_commands(first.group("name"))
            return

        arg_match = _ARG.match(text)
        if arg_match is None:
            return
        name = arg_match.group("name").lower()
        arg = arg_match.group("arg")
        if name in {"resume", "continue"}:
            yield from self._complete_sessions(arg)
        elif name == "permissions":
            yield from self._complete_permission_modes(arg)
        # `/model` has no inline completion: bare `/model` opens the interactive picker.

    # -- name completion ------------------------------------------------------

    def _complete_commands(self, prefix: str) -> "Iterable[Completion]":
        prefix_l = prefix.lower()
        start = -(len(prefix) + 1)  # replace the leading '/' plus the typed name

        for cmd, (_, summary) in sorted(_COMMAND_HELP.items()):
            if cmd.startswith(prefix_l):
                yield Completion(
                    f"/{cmd} ", start_position=start, display=f"/{cmd}", display_meta=summary
                )

        try:
            skills: list[Skill] = sorted(
                self._agent.skills.user_invocable(), key=lambda s: s.name
            )
        except Exception:  # noqa: BLE001 - completion must never break input
            return
        for skill in skills:
            name_l = skill.name.lower()
            if name_l in _COMMANDS or not name_l.startswith(prefix_l):
                continue  # a built-in command of the same name already shadows it
            hint = f" {skill.argument_hint}" if skill.argument_hint else ""
            yield Completion(
                f"/{skill.name} ",
                start_position=start,
                display=f"/{skill.name}{hint}",
                display_meta=skill.description or "skill",
            )

    # -- argument completion --------------------------------------------------

    def _complete_sessions(self, arg: str) -> "Iterable[Completion]":
        agent = self._agent
        session_dir = getattr(agent.config, "session_dir", None)
        if not session_dir:
            return
        try:
            infos = list_sessions(project_dir(session_dir, agent.session.workspace))
        except Exception:  # noqa: BLE001 - never break input on a listing error
            return
        needle = arg.strip().lower()
        start = -len(arg)
        for info in infos[:10]:
            label = session_label(info)
            if needle and needle not in label.lower() and needle not in info.session_id.lower():
                continue
            when = _dt.datetime.fromtimestamp(info.modified).strftime("%Y-%m-%d %H:%M")
            branch = f" [{info.git_branch}]" if info.git_branch else ""
            meta = f"{when} · {info.message_count} msgs{branch}"
            # Inserted text is the id; the user only ever reads the summary phrase.
            yield Completion(
                info.session_id, start_position=start, display=label[:60], display_meta=meta
            )

    def _complete_permission_modes(self, arg: str) -> "Iterable[Completion]":
        from agent_core.permissions import PermissionMode, permission_mode_label

        needle = arg.strip().lower()
        start = -len(arg)
        for mode in PermissionMode:
            if mode.value.startswith(needle):
                yield Completion(
                    mode.value,
                    start_position=start,
                    display=mode.value,
                    display_meta=permission_mode_label(mode),
                )

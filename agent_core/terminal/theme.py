from rich.theme import Theme

# A flat, low-chrome palette inspired by Claude Code: dim scaffolding, a single
# accent per state, risk colors that escalate green → yellow → red.
claude_theme = Theme({
    "info": "cyan",
    "dim": "dim",
    "warning": "yellow",
    "danger": "red",
    "success": "green",
    "thinking": "dim",
    "reasoning": "cyan",
    "answer": "default",
    "tool_name": "bold",
    "branch": "dim",
    "risk_read": "green",
    "risk_write": "yellow",
    "risk_dangerous": "red",
})


def completion_menu_style():
    """prompt_toolkit ``Style`` for the interactive chat chrome.

    White-on-black options with the highlighted command rendered blue+bold so the
    current selection stands out; meta (the grey description) and scrollbar kept low
    chrome. The bottom toolbar uses the terminal palette's subdued grey (matching the
    dimmed thinking text) on the terminal's own background, instead of prompt_toolkit's
    default reverse-video style. prompt_toolkit merges this over its default style, so
    only these classes change. Imported lazily so ``theme`` stays rich-only at module
    load.
    """
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "bottom-toolbar": "bg:default ansibrightblack noreverse",
            "completion-menu": "bg:#000000 #ffffff",
            "completion-menu.completion": "bg:#000000 #ffffff",
            "completion-menu.completion.current": "bg:#000000 #5fafff bold",
            "completion-menu.meta.completion": "bg:#000000 #808080",
            "completion-menu.meta.completion.current": "bg:#1c1c1c #5fafff",
            "scrollbar.background": "bg:#1c1c1c",
            "scrollbar.button": "bg:#5fafff",
        }
    )


def risk_style(risk: str) -> str:
    """Theme style name for a tool risk, falling back to ``dim`` when unknown.

    NOTE: ``Theme.styles`` is a plain dict, so membership must be tested with
    ``in`` — ``hasattr`` is always False on a dict and silently drops the color.
    """
    name = f"risk_{risk}"
    return name if name in claude_theme.styles else "dim"


# Glyphs, kept ASCII-light so they survive a non-UTF console. The compact tool
# stream uses ``●`` (call header) + ``⎿`` (result branch), mirroring Claude Code.
SYMBOLS = {
    "thinking": "✻",
    "answer": "●",
    "reasoning": "·",
    "tool_call": "●",
    "branch": "⎿",
    "tool_ok": "ok",
    "tool_err": "error",
    "plan": "☰",
    "plan_pending": "○",
    "plan_progress": "◐",
    "plan_completed": "●",
    "stopped": "■",
    "compacting": "⊙",
    "warning": "⚠",
    "ok": "✓",
    "fail": "✗",
}

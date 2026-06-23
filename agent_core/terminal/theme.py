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

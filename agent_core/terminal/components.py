"""Block-level renderables used by the compact terminal stream.

The per-tool stream is flat (``●``/``⎿`` lines emitted by ``TerminalRenderer``),
so the only block component left is ``DiffBlock`` — a syntax-highlighted unified
diff shown under a write/edit tool's result branch.
"""
from rich.console import RenderableType
from rich.syntax import Syntax


class DiffBlock:
    """A unified-diff string rendered with diff syntax highlighting."""

    def __init__(self, content: str):
        self.content = content

    def __rich__(self) -> RenderableType:
        return Syntax(
            self.content,
            "diff",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )

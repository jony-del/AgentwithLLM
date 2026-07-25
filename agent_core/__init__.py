"""Python ReAct agent framework."""

from typing import Any

__all__ = ["ReActAgent", "ReActConfig"]


def __getattr__(name: str) -> Any:
    """Load the public agent API only when it is requested.

    Lightweight maintenance entry points such as the standalone uninstaller
    must be importable from a stdlib-only bootstrap Python.
    """

    if name in __all__:
        from agent_core.react import ReActAgent, ReActConfig

        exports = {"ReActAgent": ReActAgent, "ReActConfig": ReActConfig}
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Cross-kernel command contracts for sandboxed process execution.

Host executables are not meaningful inside a Linux guest.  ``SandboxInvocation``
therefore carries both spellings and lets the selected backend choose exactly one.
Guest command aliases (``@python``, ``@bash``...) are resolved only from the probed
image manifest; they are never looked up on the host or on an ambient guest ``PATH``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping


GUEST_PROTOCOL_VERSION = 1
REQUIRED_GUEST_TOOLS = frozenset(
    {"bash", "pwsh", "python", "node", "npm", "npx", "pyright-langserver"}
)
_ABSOLUTE_GUEST_PATH = re.compile(r"^/[A-Za-z0-9._+/@-]+(?:/[A-Za-z0-9._+@ -]+)*$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_EXECUTABLE = re.compile(r"\.(?:exe|cmd|bat)(?:$|[\s\"'])", re.IGNORECASE)


class GuestCapabilityUnavailable(RuntimeError):
    """A command cannot be represented safely in the selected Linux guest."""

    error_type = "guest_capability_unavailable"


@dataclass(frozen=True, slots=True)
class GuestRuntimeManifest:
    """Validated, immutable capabilities reported by a sandbox image probe."""

    protocol_version: int
    guest_os: str
    architecture: str
    _tools: tuple[tuple[str, str], ...] = field(repr=False)

    @property
    def tools(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._tools))

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(name for name, _ in self._tools)

    def resolve(self, capability: str) -> str:
        value = self.tools.get(capability)
        if value is None:
            raise GuestCapabilityUnavailable(
                f"guest_capability_unavailable: image does not declare {capability!r}"
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GuestRuntimeManifest":
        try:
            protocol = int(value.get("protocol_version", value.get("protocol", 0)))
        except (TypeError, ValueError) as exc:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: invalid guest protocol version"
            ) from exc
        if protocol != GUEST_PROTOCOL_VERSION:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: unsupported guest protocol "
                f"{protocol!r}; expected {GUEST_PROTOCOL_VERSION}"
            )
        guest_os = str(value.get("guest_os", value.get("os", ""))).casefold()
        if guest_os != "linux":
            raise GuestCapabilityUnavailable(
                f"guest_capability_unavailable: guest OS must be linux, got {guest_os or 'missing'!r}"
            )
        architecture = str(value.get("architecture", value.get("arch", ""))).strip()
        if not architecture:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: guest architecture is missing"
            )
        raw_tools = value.get("tools", value.get("commands", value.get("capabilities")))
        if not isinstance(raw_tools, Mapping):
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: guest tool table is missing"
            )
        tools: list[tuple[str, str]] = []
        for raw_name, raw_path in raw_tools.items():
            name = str(raw_name).strip()
            path = str(raw_path).strip()
            if _WINDOWS_DRIVE.match(path) or _WINDOWS_EXECUTABLE.search(path):
                raise GuestCapabilityUnavailable(
                    f"guest_capability_unavailable: Windows executable in guest manifest: {path!r}"
                )
            if not name or not _ABSOLUTE_GUEST_PATH.fullmatch(path):
                raise GuestCapabilityUnavailable(
                    f"guest_capability_unavailable: invalid guest tool declaration {name!r}"
                )
            tools.append((name, path))
        return cls(protocol, guest_os, architecture, tuple(sorted(tools)))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "GuestRuntimeManifest":
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: image probe returned invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: image probe did not return an object"
            )
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class SandboxInvocation:
    """Immutable host/guest argv pair plus capabilities and per-call scope."""

    host_argv: tuple[str, ...]
    guest_argv: tuple[str, ...] | None = None
    required_guest_capabilities: frozenset[str] = frozenset()
    scope: object | None = None

    @classmethod
    def create(
        cls,
        host_argv: Iterable[object],
        *,
        guest_argv: Iterable[object] | None = None,
        required_guest_capabilities: Iterable[str] = (),
        scope: object | None = None,
    ) -> "SandboxInvocation":
        host = tuple(str(item) for item in host_argv)
        guest = None if guest_argv is None else tuple(str(item) for item in guest_argv)
        if not host:
            raise ValueError("host argv must not be empty")
        if guest is not None and not guest:
            raise ValueError("guest argv must not be empty")
        return cls(host, guest, frozenset(required_guest_capabilities), scope)

    def guest_command(self, manifest: GuestRuntimeManifest) -> list[str]:
        if self.guest_argv is None:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: this invocation has no Linux guest argv"
            )
        missing = self.required_guest_capabilities - manifest.capabilities
        if missing:
            raise GuestCapabilityUnavailable(
                "guest_capability_unavailable: image is missing " + ", ".join(sorted(missing))
            )
        result: list[str] = []
        for item in self.guest_argv:
            if item.startswith("@") and len(item) > 1:
                item = manifest.resolve(item[1:])
            if _WINDOWS_DRIVE.match(item) or _WINDOWS_EXECUTABLE.search(item):
                raise GuestCapabilityUnavailable(
                    f"guest_capability_unavailable: Windows-only argument cannot enter guest: {item!r}"
                )
            result.append(item)
        return result


def guest_alias(command: str) -> str:
    """Normalize a declared command name to the manifest alias spelling."""

    return command if command.startswith("@") else f"@{command}"

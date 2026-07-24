from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SecretFinding:
    rule: str


_SECRET_RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"']?[^\s\"']{12,}"
    ),
}


class SecretDetectedError(ValueError):
    def __init__(self, rules: list[str]) -> None:
        self.rules = sorted(set(rules))
        super().__init__("memory rejected by secret scanning: " + ", ".join(self.rules))


def scan_secrets(text: str) -> list[SecretFinding]:
    """Return rule names only; callers must never log matching secret values."""
    return [SecretFinding(name) for name, pattern in _SECRET_RULES.items() if pattern.search(text)]


def require_secret_free(*values: str) -> None:
    findings = scan_secrets("\n".join(values))
    if findings:
        raise SecretDetectedError([finding.rule for finding in findings])

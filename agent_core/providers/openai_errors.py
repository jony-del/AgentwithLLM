from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenAIErrorInfo:
    message: str
    code: str = ""
    param: str = ""
    type: str = ""


def parse_openai_error(text: str) -> OpenAIErrorInfo:
    """Parse the common OpenAI error envelope, tolerating plain-text bodies."""

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return OpenAIErrorInfo(message=str(text)[:300])
    if not isinstance(payload, dict):
        return OpenAIErrorInfo(message=str(payload)[:300])
    error = payload.get("error")
    if isinstance(error, dict):
        return OpenAIErrorInfo(
            message=str(error.get("message") or text)[:300],
            code=str(error.get("code") or ""),
            param=str(error.get("param") or ""),
            type=str(error.get("type") or ""),
        )
    return OpenAIErrorInfo(message=str(payload.get("message") or text)[:300])


def is_parameter_error(info: OpenAIErrorInfo) -> bool:
    haystack = " ".join(part for part in (info.message, info.code, info.type, info.param) if part).lower()
    return any(marker in haystack for marker in ("unsupported", "invalid", "unknown parameter", "unrecognized"))


def format_openai_error(protocol: str, model: str | None, status: int, info: OpenAIErrorInfo) -> str:
    """Return an actionable non-retryable OpenAI error message."""

    model_part = f" model {model!r}" if model else ""
    details = info.message or "request rejected"
    bits = [f"{protocol} API error {status}{model_part}: {details}"]
    structured = []
    if info.param:
        structured.append(f"param={info.param!r}")
    if info.code:
        structured.append(f"code={info.code!r}")
    if info.type:
        structured.append(f"type={info.type!r}")
    if structured:
        bits.append(f"({', '.join(structured)})")
    if is_parameter_error(info):
        bits.append(
            "Check the explicit provider/protocol selection and model capability profile; "
            "Polaris does not infer /v1/responses vs /v1/chat/completions from model IDs or base URLs. "
            "If this is a newly supported model, update the OpenAI capability matrix instead of relying on fallback."
        )
    return " ".join(bits)

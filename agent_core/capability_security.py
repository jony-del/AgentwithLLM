"""Security-owned identities and immutable activation data for capabilities.

The model-facing catalog is deliberately descriptive.  Values in this module are
constructed by the host from normalized source configuration and verified bytes; a
publisher cannot grant itself a stronger trust tier by adding fields to a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from agent_core.plugin_spec import MarketplaceSourceConfig, PluginSourceConfig


CAPABILITY_STATE_SCHEMA_VERSION = 3


class TrustTier(str, Enum):
    ANTHROPIC_FIRST_PARTY = "anthropic_first_party"
    VERIFIED_PUBLISHER = "verified_publisher"
    COMMUNITY = "community"
    LOCAL_USER_DECLARED = "local_user_declared"


# Reserved display names cannot be rebound to another source.  An omitted ref is
# normalized to main for these three bundled defaults; any other ref is a distinct,
# non-reserved configuration and therefore rejected under the reserved name.
OFFICIAL_MARKETPLACES: Mapping[str, tuple[str, str]] = {
    "claude-plugins-official": ("anthropics/claude-plugins-official", "main"),
    "anthropic-agent-skills": ("anthropics/skills", "main"),
    "knowledge-work-plugins": ("anthropics/knowledge-work-plugins", "main"),
}


def _canonical_url(raw: str) -> str:
    value = raw.strip()
    parsed = urlsplit(value)
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    if not scheme or not host:
        return value
    port = parsed.port
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    user = parsed.username or ""
    if user:
        user += "@"
    netloc = user + host + (f":{port}" if port and not default_port else "")
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def canonical_marketplace_source(source: MarketplaceSourceConfig) -> str:
    """Return a stable, secret-free identity for one marketplace source."""

    if source.kind == "github":
        ref = source.ref or (
            OFFICIAL_MARKETPLACES.get(source.name, ("", ""))[1] if source.name else ""
        )
        return f"github:{source.repo.casefold()}@{ref or 'HEAD'}:{source.path}"
    if source.kind in {"git", "url"}:
        return f"{source.kind}:{_canonical_url(source.url)}@{source.ref or 'HEAD'}:{source.path}"
    if source.kind in {"file", "directory"}:
        path = os.path.normcase(str(Path(source.path).expanduser().resolve()))
        return f"{source.kind}:{path}"
    if source.kind == "settings":
        body = json.dumps(
            {"name": source.name, "plugins": list(source.plugins)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return f"settings:{source.name}:{hashlib.sha256(body).hexdigest()}"
    return f"{source.kind}:{source.display()}"


def marketplace_source_id(source: MarketplaceSourceConfig) -> str:
    return hashlib.sha256(canonical_marketplace_source(source).encode("utf-8")).hexdigest()


def validate_reserved_marketplace(name: str, source: MarketplaceSourceConfig) -> None:
    expected = OFFICIAL_MARKETPLACES.get(name)
    if expected is None:
        return
    expected_repo, expected_ref = expected
    if (
        source.kind != "github"
        or source.repo.casefold() != expected_repo.casefold()
        or (source.ref or expected_ref) != expected_ref
        or bool(source.path)
    ):
        raise ValueError(
            f"reserved marketplace {name!r} must use github source "
            f"{expected_repo}@{expected_ref}"
        )


def marketplace_trust_tier(
    name: str, source: MarketplaceSourceConfig
) -> TrustTier:
    """Classify a source using host-owned facts, never catalog metadata."""

    validate_reserved_marketplace(name, source)
    if source.kind in {"file", "directory", "settings"}:
        return TrustTier.LOCAL_USER_DECLARED
    if source.kind == "github" and source.repo.casefold().startswith("anthropics/"):
        return TrustTier.ANTHROPIC_FIRST_PARTY
    if source.kind == "git":
        parsed = urlsplit(source.url)
        path = parsed.path.strip("/")
        if parsed.hostname and parsed.hostname.casefold() == "github.com":
            if path.removesuffix(".git").casefold().startswith("anthropics/"):
                return TrustTier.ANTHROPIC_FIRST_PARTY
    return TrustTier.COMMUNITY


def plugin_trust_tier(
    marketplace_tier: TrustTier,
    source: PluginSourceConfig,
) -> TrustTier:
    """A trusted catalog does not automatically make an external artifact trusted."""

    if marketplace_tier is TrustTier.LOCAL_USER_DECLARED:
        return marketplace_tier
    if source.kind == "relative":
        return marketplace_tier
    if source.kind == "github" and source.repo.casefold().startswith("anthropics/"):
        return TrustTier.ANTHROPIC_FIRST_PARTY
    if source.kind in {"url", "git-subdir"}:
        parsed = urlsplit(source.url)
        path = parsed.path.strip("/").removesuffix(".git").casefold()
        if parsed.hostname and parsed.hostname.casefold() == "github.com" and path.startswith(
            "anthropics/"
        ):
            return TrustTier.ANTHROPIC_FIRST_PARTY
    return TrustTier.COMMUNITY


@dataclass(frozen=True, slots=True)
class MarketplaceIdentity:
    name: str
    source_id: str
    canonical_source: str
    snapshot: str
    trust_tier: TrustTier

    @classmethod
    def create(
        cls,
        name: str,
        source: MarketplaceSourceConfig,
        snapshot: str = "",
    ) -> "MarketplaceIdentity":
        return cls(
            name=name,
            source_id=marketplace_source_id(source),
            canonical_source=canonical_marketplace_source(source),
            snapshot=snapshot,
            trust_tier=marketplace_trust_tier(name, source),
        )


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    kind: str
    source: str
    digest: str
    path: str = ""
    commit: str = ""
    version: str = ""
    platform: str = ""


@dataclass(frozen=True, slots=True)
class ActivationPlan:
    plan_id: str
    capability_id: str
    catalog_digest: str
    plan_digest: str
    created_at: float
    expires_at: float
    trust_tier: TrustTier
    risk: str
    components: tuple[str, ...]
    artifacts: tuple[ResolvedArtifact, ...] = ()
    dependencies: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    configuration: tuple[str, ...] = ()
    requires_approval: bool = True
    installable: bool = True
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def issue(
        *,
        capability_id: str,
        catalog_digest: str,
        trust_tier: TrustTier,
        risk: str,
        components: tuple[str, ...],
        artifacts: tuple[ResolvedArtifact, ...] = (),
        dependencies: tuple[str, ...] = (),
        permissions: tuple[str, ...] = (),
        configuration: tuple[str, ...] = (),
        requires_approval: bool = True,
        installable: bool = True,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
        lifetime_seconds: int = 900,
    ) -> "ActivationPlan":
        now = time.time()
        expires_at = now + max(60, lifetime_seconds)
        plan_id = secrets.token_urlsafe(18)
        payload = {
            "plan_id": plan_id,
            "capability_id": capability_id,
            "catalog_digest": catalog_digest,
            "created_at": now,
            "expires_at": expires_at,
            "trust_tier": trust_tier.value,
            "risk": risk,
            "components": list(components),
            "artifacts": [asdict(item) for item in artifacts],
            "dependencies": list(dependencies),
            "permissions": list(permissions),
            "configuration": list(configuration),
            "requires_approval": requires_approval,
            "installable": installable,
            "reason": reason,
            "metadata": dict(metadata or {}),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        return ActivationPlan(
            plan_id=plan_id,
            capability_id=capability_id,
            catalog_digest=catalog_digest,
            plan_digest=digest,
            created_at=now,
            expires_at=expires_at,
            trust_tier=trust_tier,
            risk=risk,
            components=components,
            artifacts=artifacts,
            dependencies=dependencies,
            permissions=permissions,
            configuration=configuration,
            requires_approval=requires_approval,
            installable=installable,
            reason=reason,
            metadata=dict(metadata or {}),
        )

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value["trust_tier"] = self.trust_tier.value
        return value

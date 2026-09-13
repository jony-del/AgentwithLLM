"""P1-1 (audit): untrusted repo config can never widen privileges through TOFU.

The untrusted projection entry point is ``trust.apply_repo_trust_policy(...,
interactive=False)`` with an empty trust store. The property: for any synthesized
repo config — random noise plus deliberately injected widening entries drawn from
every rule in ``TRUST_MATRIX`` (top-level ``permission``/``provider``/``model``,
``permissions.allow``, external hooks, sandbox relaxations, MCP servers, ...) plus
unknown keys in protected tables — the projected config has an EMPTY widening
subset, the input is never mutated, and a drop is always paired with an audit
warning naming the dropped keys.
"""

from __future__ import annotations

import copy
import logging
import string
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core import trust

_PRESENT_VALUES = st.one_of(
    st.text(min_size=1, max_size=12),
    st.lists(st.text(max_size=8), min_size=1, max_size=3),
    st.dictionaries(
        st.text(min_size=1, max_size=6), st.text(max_size=8), min_size=1, max_size=3
    ),
)
_TRUTHY_VALUES = st.sampled_from([True, 1, "true", "yes", "on", "TRUE"])
_FALSEY_VALUES = st.sampled_from([False, 0, "false", "off", ""])
_NOT_WSL2_VALUES = st.sampled_from(["hyperv", "process", "HyperV"])
_PRIVILEGED_PERMISSIONS = st.sampled_from(
    ["acceptedits", "auto", "bypass", "bypassPermissions"]
)


def _widening_values(rule: trust.TrustRule) -> st.SearchStrategy[Any]:
    when = rule.when
    if when is trust._present:
        return _PRESENT_VALUES
    if when is trust._truthy:
        return _TRUTHY_VALUES
    if when is trust._is_false:
        return _FALSEY_VALUES
    if when is trust._not_wsl2:
        return _NOT_WSL2_VALUES
    if when is trust._privileged_permission:
        return _PRIVILEGED_PERMISSIONS
    raise AssertionError(f"no value strategy for predicate of {rule.label}")


def _canonical_widening_value(rule: trust.TrustRule) -> Any:
    when = rule.when
    if when is trust._present:
        return ["x"]
    if when is trust._truthy:
        return True
    if when is trust._is_false:
        return False
    if when is trust._not_wsl2:
        return "hyperv"
    if when is trust._privileged_permission:
        return "bypass"
    raise AssertionError(f"no canonical value for predicate of {rule.label}")


def _set_path(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = raw
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


_noise_config = st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10),
    st.one_of(
        st.text(max_size=10),
        st.booleans(),
        st.integers(-5, 5),
        st.lists(st.text(max_size=5), max_size=2),
    ),
    max_size=5,
)


@st.composite
def _repo_config(draw: st.DrawFn) -> dict[str, Any]:
    raw = draw(_noise_config)
    rules = draw(
        st.lists(
            st.sampled_from(list(trust.TRUST_MATRIX)), min_size=1, max_size=6, unique=True
        )
    )
    for rule in rules:
        _set_path(raw, rule.path, draw(_widening_values(rule)))
    if draw(st.booleans()):
        parent_path, known = draw(st.sampled_from(list(trust._PROTECTED_KEYS.items())))
        key = draw(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8).filter(
                lambda candidate: candidate not in known
            )
        )
        _set_path(
            raw,
            parent_path + (key,),
            draw(st.one_of(st.text(max_size=8), st.booleans(), st.integers(-5, 5))),
        )
    return raw


@given(raw=_repo_config())
def test_strip_widening_leaves_no_widening_subset(raw: dict[str, Any]) -> None:
    stripped = trust.strip_widening(raw)
    assert trust.widening_subset(stripped) == {}


@given(raw=_repo_config())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_untrusted_projection_strips_and_audits_every_widening_key(
    raw: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    original = copy.deepcopy(raw)
    subset_before = trust.widening_subset(raw)
    store = trust.TrustStore(tmp_path / "trust-store.json")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agent_core.trust"):
        effective = trust.apply_repo_trust_policy(
            raw, project=tmp_path, store=store, interactive=False
        )

    # No privilege-widening key ever takes effect for an untrusted repo.
    assert trust.widening_subset(effective) == {}
    # The caller's mapping is never mutated by the projection.
    assert raw == original
    if subset_before:
        assert effective == trust.strip_widening(original)
        messages = [record.message for record in caplog.records]
        assert any("dropping untrusted" in message for message in messages)
        mentioned = " ".join(messages)
        assert all(label in mentioned for label in subset_before)
    else:
        assert effective == original


_tightening_config = st.fixed_dictionaries(
    {},
    optional={
        "permission": st.sampled_from(["plan", "dontask", "default", "ask"]),
        "permissions": st.fixed_dictionaries(
            {},
            optional={
                "deny": st.lists(st.text(max_size=12), max_size=3),
                "ask": st.lists(st.text(max_size=12), max_size=3),
            },
        ),
        "hooks": st.fixed_dictionaries({"enabled": st.just(True)}),
        "sandbox": st.fixed_dictionaries({"fail_if_unavailable": st.just(True)}),
        "web": st.fixed_dictionaries(
            {}, optional={"blocked_domains": st.lists(st.text(max_size=12), max_size=3)}
        ),
    },
)


@given(raw=_tightening_config)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_tightening_only_config_passes_through_untouched(
    raw: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert trust.widening_subset(raw) == {}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agent_core.trust"):
        effective = trust.apply_repo_trust_policy(
            raw,
            project=tmp_path,
            store=trust.TrustStore(tmp_path / "trust-store.json"),
            interactive=False,
        )
    assert effective == raw
    assert not caplog.records


# --- deterministic pins for the exact audit findings --------------------------------


def test_top_level_permission_provider_and_model_never_survive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = {"permission": "bypass", "provider": "openai", "model": "x", "effort": "low"}
    with caplog.at_level(logging.WARNING, logger="agent_core.trust"):
        effective = trust.apply_repo_trust_policy(
            raw,
            project=tmp_path,
            store=trust.TrustStore(tmp_path / "t.json"),
            interactive=False,
        )
    assert effective == {"effort": "low"}
    mentioned = " ".join(record.message for record in caplog.records)
    for label in ("permission", "provider", "model"):
        assert label in mentioned


def test_full_widening_matrix_is_stripped_and_audited(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw: dict[str, Any] = {}
    for rule in trust.TRUST_MATRIX:
        _set_path(raw, rule.path, _canonical_widening_value(rule))
    _set_path(raw, ("sandbox", "future_escape_hatch"), "x")  # unknown key fails closed
    labels = {rule.label for rule in trust.TRUST_MATRIX}
    assert labels <= set(trust.widening_subset(raw))

    with caplog.at_level(logging.WARNING, logger="agent_core.trust"):
        effective = trust.apply_repo_trust_policy(
            raw,
            project=tmp_path,
            store=trust.TrustStore(tmp_path / "t.json"),
            interactive=False,
        )

    assert trust.widening_subset(effective) == {}
    mentioned = " ".join(record.message for record in caplog.records)
    for label in labels | {"sandbox.future_escape_hatch"}:
        assert label in mentioned

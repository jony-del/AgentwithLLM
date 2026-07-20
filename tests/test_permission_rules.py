"""Tests for the fine-grained rule engine (policy layer)."""

from agent_core.permission_rules import (
    BINARY_HIJACK_VARS,
    ParsedRule,
    RuleSet,
    _normalize_subcommand,
    _split_subcommands,
    parse_rule,
)
from agent_core.permission_types import PermissionBehavior, PermissionRuleSource


# -- rule parsing --------------------------------------------------------------------


def test_parse_whole_tool_rule() -> None:
    assert parse_rule("WebFetch") == ParsedRule("WebFetch", None)


def test_parse_rule_with_content() -> None:
    assert parse_rule("bash(git *)") == ParsedRule("bash", "git *")


def test_parse_bad_rule_degrades_to_none() -> None:
    assert parse_rule("bash(unterminated") is None
    assert parse_rule("") is None
    assert parse_rule("()") is None


def test_from_lists_drops_unparseable_entries() -> None:
    rules = RuleSet.from_lists(allow=["EchoTool", "bad("], deny=[], ask=[])
    assert len(rules.allow) == 1
    assert rules.allow[0].tool_name == "EchoTool"


# -- shell command matching: exact / prefix / wildcard -------------------------------


def test_exact_match() -> None:
    rules = RuleSet.from_lists(allow=["bash(npm run build)"])
    assert rules.allow_matches("bash", {"command": "npm run build"})
    assert not rules.allow_matches("bash", {"command": "npm run test"})


def test_prefix_match() -> None:
    rules = RuleSet.from_lists(allow=["bash(npm:*)"])
    assert rules.allow_matches("bash", {"command": "npm run build"})
    assert rules.allow_matches("bash", {"command": "npm"})
    assert not rules.allow_matches("bash", {"command": "npmx evil"})


def test_wildcard_match_and_bare_trailing() -> None:
    rules = RuleSet.from_lists(deny=["bash(git * --force)"])
    assert rules.deny_matches("bash", {"command": "git push --force"})
    rules2 = RuleSet.from_lists(allow=["bash(git *)"])
    assert rules2.allow_matches("bash", {"command": "git"})  # bare command matches "git *"
    assert rules2.allow_matches("bash", {"command": "git status"})


# -- compound command semantics: allow needs ALL, deny needs ANY ---------------------


def test_compound_allow_requires_every_subcommand() -> None:
    rules = RuleSet.from_lists(allow=["bash(git *)"])
    assert rules.allow_matches("bash", {"command": "git status && git diff"})
    # One sub-command not covered → not allowed.
    assert not rules.allow_matches("bash", {"command": "git status && rm x"})


def test_compound_deny_triggers_on_any_subcommand() -> None:
    rules = RuleSet.from_lists(deny=["bash(rm *)"])
    assert rules.deny_matches("bash", {"command": "git status && rm -rf /"})
    assert rules.deny_matches("bash", {"command": "rm x | cat"})


def test_split_respects_quotes() -> None:
    # A ';' inside quotes must not split into a fake sub-command.
    parts = _split_subcommands("echo 'a;b' && ls")
    assert parts == ["echo 'a;b'", "ls"]


# -- anti-evasion --------------------------------------------------------------------


def test_safe_env_var_prefix_is_stripped_for_allow() -> None:
    rules = RuleSet.from_lists(allow=["bash(npm run test)"])
    assert rules.allow_matches("bash", {"command": "LANG=C npm run test"})


def test_binary_hijack_var_is_never_stripped() -> None:
    # PATH= must NOT be stripped, so the command no longer matches a naive allow.
    assert BINARY_HIJACK_VARS.match("PATH")
    rules = RuleSet.from_lists(allow=["bash(npm run test)"])
    assert not rules.allow_matches("bash", {"command": "PATH=/tmp npm run test"})
    assert not rules.allow_matches("bash", {"command": "LD_PRELOAD=x npm run test"})


def test_safe_wrapper_stripping() -> None:
    rules = RuleSet.from_lists(allow=["bash(npm run test)"])
    assert rules.allow_matches("bash", {"command": "timeout 300 npm run test"})
    assert rules.allow_matches("bash", {"command": "nohup npm run test"})


def test_unsafe_wrapper_flag_halts_stripping() -> None:
    # An injected flag on the wrapper must NOT be silently stripped (fail safe).
    normalized = _normalize_subcommand("timeout -k$(id) 10 npm run test")
    assert normalized.startswith("timeout")


def test_deny_matches_despite_wrapper_evasion() -> None:
    # Deny is checked against the raw form too, so wrapping can't dodge it.
    rules = RuleSet.from_lists(deny=["bash(rm *)"])
    assert rules.deny_matches("bash", {"command": "timeout 5 rm -rf /"})


# -- scalar matching: paths and domains ----------------------------------------------


def test_path_glob_match() -> None:
    rules = RuleSet.from_lists(allow=["read_text_file(src/**)"])
    assert rules.allow_matches("read_text_file", {"path": "src/pkg/mod.py"})
    assert not rules.allow_matches("read_text_file", {"path": "secrets/key.pem"})


def test_domain_match() -> None:
    rules = RuleSet.from_lists(deny=["web_fetch(domain:evil.example)"])
    assert rules.deny_matches("web_fetch", {"url": "https://evil.example/x"})
    assert rules.deny_matches("web_fetch", {"url": "https://api.evil.example/x"})
    assert not rules.deny_matches("web_fetch", {"url": "https://good.example/x"})


def test_whole_tool_rule_matches_any_call() -> None:
    rules = RuleSet.from_lists(deny=["web_fetch"])
    assert rules.deny_matches("web_fetch", {"url": "https://anything.example"})


def test_rule_scoped_to_its_tool() -> None:
    rules = RuleSet.from_lists(deny=["bash(rm *)"])
    # A different tool with the same-looking arg is unaffected.
    assert not rules.deny_matches("read_text_file", {"path": "rm x"})


def test_merge_appends_rules() -> None:
    base = RuleSet.from_lists(deny=["bash(rm *)"])
    extra = RuleSet.from_lists(allow=["bash(git *)"])
    merged = base.merge(extra)
    assert merged.deny_matches("bash", {"command": "rm x"})
    assert merged.allow_matches("bash", {"command": "git status"})
    # Originals are untouched.
    assert not base.allow


def test_rule_provenance_survives_parse_merge_and_match() -> None:
    project = RuleSet.from_lists(
        deny=["bash(rm *)"], source=PermissionRuleSource.PROJECT
    )
    cli = RuleSet.from_lists(
        allow=["bash(git *)"], source=PermissionRuleSource.CLI
    )
    merged = project.merge(cli)

    denied = merged.deny_match("bash", {"command": "rm x"})
    allowed = merged.allow_match("bash", {"command": "git status"})

    assert denied is not None and denied.source is PermissionRuleSource.PROJECT
    assert denied.behavior is PermissionBehavior.DENY
    assert allowed is not None and allowed.source is PermissionRuleSource.CLI
    assert allowed.raw == "bash(git *)"

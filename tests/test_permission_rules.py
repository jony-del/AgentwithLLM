"""Tests for the fine-grained rule engine (policy layer)."""

from agent_core.permission_rules import (
    BINARY_HIJACK_VARS,
    ParsedRule,
    RuleSet,
    _normalize_subcommand,
    _split_subcommands,
    parse_rule,
)


# -- rule parsing --------------------------------------------------------------------


def test_parse_whole_tool_rule() -> None:
    assert parse_rule("WebFetch") == ParsedRule("WebFetch", None)


def test_parse_rule_with_content() -> None:
    assert parse_rule("run_command(git *)") == ParsedRule("run_command", "git *")


def test_parse_bad_rule_degrades_to_none() -> None:
    assert parse_rule("run_command(unterminated") is None
    assert parse_rule("") is None
    assert parse_rule("()") is None


def test_from_lists_drops_unparseable_entries() -> None:
    rules = RuleSet.from_lists(allow=["EchoTool", "bad("], deny=[], ask=[])
    assert len(rules.allow) == 1
    assert rules.allow[0].tool_name == "EchoTool"


# -- shell command matching: exact / prefix / wildcard -------------------------------


def test_exact_match() -> None:
    rules = RuleSet.from_lists(allow=["run_command(npm run build)"])
    assert rules.allow_matches("run_command", {"command": "npm run build"})
    assert not rules.allow_matches("run_command", {"command": "npm run test"})


def test_prefix_match() -> None:
    rules = RuleSet.from_lists(allow=["run_command(npm:*)"])
    assert rules.allow_matches("run_command", {"command": "npm run build"})
    assert rules.allow_matches("run_command", {"command": "npm"})
    assert not rules.allow_matches("run_command", {"command": "npmx evil"})


def test_wildcard_match_and_bare_trailing() -> None:
    rules = RuleSet.from_lists(deny=["run_command(git * --force)"])
    assert rules.deny_matches("run_command", {"command": "git push --force"})
    rules2 = RuleSet.from_lists(allow=["run_command(git *)"])
    assert rules2.allow_matches("run_command", {"command": "git"})  # bare command matches "git *"
    assert rules2.allow_matches("run_command", {"command": "git status"})


# -- compound command semantics: allow needs ALL, deny needs ANY ---------------------


def test_compound_allow_requires_every_subcommand() -> None:
    rules = RuleSet.from_lists(allow=["run_command(git *)"])
    assert rules.allow_matches("run_command", {"command": "git status && git diff"})
    # One sub-command not covered → not allowed.
    assert not rules.allow_matches("run_command", {"command": "git status && rm x"})


def test_compound_deny_triggers_on_any_subcommand() -> None:
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    assert rules.deny_matches("run_command", {"command": "git status && rm -rf /"})
    assert rules.deny_matches("run_command", {"command": "rm x | cat"})


def test_split_respects_quotes() -> None:
    # A ';' inside quotes must not split into a fake sub-command.
    parts = _split_subcommands("echo 'a;b' && ls")
    assert parts == ["echo 'a;b'", "ls"]


# -- anti-evasion --------------------------------------------------------------------


def test_safe_env_var_prefix_is_stripped_for_allow() -> None:
    rules = RuleSet.from_lists(allow=["run_command(npm run test)"])
    assert rules.allow_matches("run_command", {"command": "LANG=C npm run test"})


def test_binary_hijack_var_is_never_stripped() -> None:
    # PATH= must NOT be stripped, so the command no longer matches a naive allow.
    assert BINARY_HIJACK_VARS.match("PATH")
    rules = RuleSet.from_lists(allow=["run_command(npm run test)"])
    assert not rules.allow_matches("run_command", {"command": "PATH=/tmp npm run test"})
    assert not rules.allow_matches("run_command", {"command": "LD_PRELOAD=x npm run test"})


def test_safe_wrapper_stripping() -> None:
    rules = RuleSet.from_lists(allow=["run_command(npm run test)"])
    assert rules.allow_matches("run_command", {"command": "timeout 300 npm run test"})
    assert rules.allow_matches("run_command", {"command": "nohup npm run test"})


def test_unsafe_wrapper_flag_halts_stripping() -> None:
    # An injected flag on the wrapper must NOT be silently stripped (fail safe).
    normalized = _normalize_subcommand("timeout -k$(id) 10 npm run test")
    assert normalized.startswith("timeout")


def test_deny_matches_despite_wrapper_evasion() -> None:
    # Deny is checked against the raw form too, so wrapping can't dodge it.
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    assert rules.deny_matches("run_command", {"command": "timeout 5 rm -rf /"})


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
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    # A different tool with the same-looking arg is unaffected.
    assert not rules.deny_matches("read_text_file", {"path": "rm x"})


def test_merge_appends_rules() -> None:
    base = RuleSet.from_lists(deny=["run_command(rm *)"])
    extra = RuleSet.from_lists(allow=["run_command(git *)"])
    merged = base.merge(extra)
    assert merged.deny_matches("run_command", {"command": "rm x"})
    assert merged.allow_matches("run_command", {"command": "git status"})
    # Originals are untouched.
    assert not base.allow

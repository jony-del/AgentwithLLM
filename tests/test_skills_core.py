"""Unit tests for the skill subsystem's pure pieces: frontmatter, loader, registry,
dispatch parsing/rendering, and config resolution."""

from __future__ import annotations

from pathlib import Path

from agent_core.config import resolve_skills_config
from agent_core.skills import (
    Skill,
    SkillContext,
    SkillRegistry,
    SkillsConfig,
    discover_skill_dirs,
    fork_preset,
    load_skill_file,
    load_skills,
    looks_like_command,
    parse_frontmatter,
    parse_slash_command,
    render_skill_prompt,
)


# --- frontmatter -------------------------------------------------------------


def test_frontmatter_parses_scalars_bools_and_lists() -> None:
    text = (
        "---\n"
        "name: my-skill\n"
        "description: Does a thing.\n"
        "when-to-use: When asked.\n"
        "allowed-tools: [read_file, search_text]\n"
        "user-invocable: true\n"
        "disable-model-invocation: false\n"
        "context: fork\n"
        "aliases:\n"
        "  - foo\n"
        "  - bar\n"
        "---\n"
        "Body line one.\nBody line two.\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "my-skill"
    assert meta["when_to_use"] == "When asked."  # hyphen normalised to underscore
    assert meta["allowed_tools"] == ["read_file", "search_text"]
    assert meta["user_invocable"] is True
    assert meta["disable_model_invocation"] is False
    assert meta["context"] == "fork"
    assert meta["aliases"] == ["foo", "bar"]
    assert body.strip() == "Body line one.\nBody line two."


def test_frontmatter_absent_returns_whole_text_as_body() -> None:
    meta, body = parse_frontmatter("Just a body, no fence.\n")
    assert meta == {}
    assert body == "Just a body, no fence.\n"


def test_frontmatter_unterminated_fence_is_all_body() -> None:
    text = "---\nname: x\nno closing fence here\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_frontmatter_strips_quotes() -> None:
    meta, _ = parse_frontmatter('---\ndescription: "Quoted value"\n---\nbody')
    assert meta["description"] == "Quoted value"


def test_frontmatter_numeric_and_null_scalars() -> None:
    # PyYAML types these; the loader stringifies where it needs to, so they stay safe.
    meta, _ = parse_frontmatter("---\ncount: 5\nmodel: ~\n---\nbody")
    assert meta["count"] == 5
    assert meta["model"] is None


def test_frontmatter_bad_yaml_degrades_to_no_metadata() -> None:
    # Unbalanced/invalid YAML must not raise — it degrades to "no metadata".
    meta, body = parse_frontmatter("---\nfoo: [unclosed\n---\nthe body")
    assert meta == {}
    assert body == "---\nfoo: [unclosed\n---\nthe body"


def test_frontmatter_nested_mapping_supported() -> None:
    meta, _ = parse_frontmatter("---\nhooks:\n  Stop: echo hi\n---\nbody")
    assert meta["hooks"] == {"Stop": "echo hi"}


# --- loader ------------------------------------------------------------------


def _write_skill(directory: Path, name: str, body: str = "Do the thing.", **front: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---"] + [f"{key.replace('_', '-')}: {value}" for key, value in front.items()] + ["---", body]
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_bundled_skills_load() -> None:
    skills = load_skills(discover_skill_dirs(Path.cwd(), SkillsConfig()))
    names = {skill.name for skill in skills}
    assert {"commit", "review"} <= names
    review = next(skill for skill in skills if skill.name == "review")
    assert review.context is SkillContext.FORK
    assert review.allowed_tools  # declared read-only tools


def test_load_skill_file_directory_form_defaults_name_to_folder(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "hello", description="Greet")
    skill = load_skill_file(path)
    assert skill is not None
    assert skill.name == "hello"
    assert skill.description == "Greet"
    assert skill.context is SkillContext.INLINE  # default


def test_load_skill_file_loose_form_defaults_name_to_stem(tmp_path: Path) -> None:
    loose = tmp_path / "greet.md"
    loose.write_text("---\ndescription: hi\n---\nSay hi.", encoding="utf-8")
    skill = load_skill_file(loose)
    assert skill is not None
    assert skill.name == "greet"


def test_load_skill_file_skips_empty_body(tmp_path: Path) -> None:
    loose = tmp_path / "empty.md"
    loose.write_text("---\nname: empty\n---\n", encoding="utf-8")
    assert load_skill_file(loose) is None


def test_project_dir_overrides_user_dir_on_name_collision(tmp_path: Path) -> None:
    user = tmp_path / "user"
    project = tmp_path / "proj"
    _write_skill(user, "dup", body="USER VERSION", description="user")
    _write_skill(project, "dup", body="PROJECT VERSION", description="project")
    # discover order is low->high precedence: [bundled, user, project, extra]
    skills = load_skills([user, project])
    dup = next(skill for skill in skills if skill.name == "dup")
    assert dup.body == "PROJECT VERSION"


def test_disabled_names_are_filtered(tmp_path: Path) -> None:
    _write_skill(tmp_path, "keep")
    _write_skill(tmp_path, "drop")
    skills = load_skills([tmp_path], disabled=("drop",))
    names = {skill.name for skill in skills}
    assert names == {"keep"}


def test_bad_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    good = _write_skill(tmp_path, "good")
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_bytes(b"\xff\xfe not utf-8 \x00")
    skills = load_skills([tmp_path])
    assert {skill.name for skill in skills} == {"good"}
    assert good.exists()


def test_discover_skill_dirs_order_and_resolution(tmp_path: Path) -> None:
    config = SkillsConfig(user_dir="~/x", project_dir=".polaris/skills", skills_dirs=(str(tmp_path / "extra"),))
    dirs = discover_skill_dirs(tmp_path, config)
    # bundled first, project resolved under workspace, extra last.
    assert dirs[0].name == "bundled"
    assert dirs[-1] == tmp_path / "extra"
    assert (tmp_path / ".polaris" / "skills") in dirs


# --- registry ----------------------------------------------------------------


def test_registry_lookup_by_name_and_alias() -> None:
    skill = Skill(name="deploy", description="d", body="b", aliases=("ship", "release"))
    registry = SkillRegistry([skill])
    assert registry.get("deploy") is skill
    assert registry.get("DEPLOY") is skill  # case-insensitive
    assert registry.get("ship") is skill  # alias
    assert registry.get("missing") is None


def test_registry_invocability_partitions() -> None:
    human = Skill(name="h", description="", body="b", user_invocable=True, disable_model_invocation=True)
    model = Skill(name="m", description="", body="b", user_invocable=False, disable_model_invocation=False)
    registry = SkillRegistry([human, model])
    assert [s.name for s in registry.user_invocable()] == ["h"]
    assert [s.name for s in registry.model_invocable()] == ["m"]


# --- dispatch ----------------------------------------------------------------


def test_parse_slash_command_splits_name_and_args() -> None:
    parsed = parse_slash_command("/commit fix login bug")
    assert parsed is not None
    assert parsed.name == "commit"
    assert parsed.args == "fix login bug"


def test_parse_slash_command_no_args() -> None:
    parsed = parse_slash_command("/skills")
    assert parsed is not None and parsed.name == "skills" and parsed.args == ""


def test_parse_slash_command_non_command_returns_none() -> None:
    assert parse_slash_command("hello world") is None


def test_looks_like_command_rejects_paths() -> None:
    assert looks_like_command("commit")
    assert looks_like_command("plugin:name")
    assert not looks_like_command("path/to/file")
    assert not looks_like_command("file.txt")


def test_render_skill_prompt_substitutes_placeholder() -> None:
    skill = Skill(name="s", description="", body="Before $ARGUMENTS after")
    assert render_skill_prompt(skill, "MID") == "Before MID after"


def test_render_skill_prompt_appends_when_no_placeholder() -> None:
    skill = Skill(name="s", description="", body="Body")
    assert render_skill_prompt(skill, "extra") == "Body\n\nextra"
    assert render_skill_prompt(skill, "") == "Body"


def test_fork_preset_maps_read_only_vs_full() -> None:
    assert fork_preset(("read_text_file", "search_text")) == "read_only"
    assert fork_preset(("edit_file",)) == "full"
    assert fork_preset(()) == "full"  # nothing declared -> full


# --- config ------------------------------------------------------------------


def test_skills_config_defaults() -> None:
    config = SkillsConfig()
    assert config.enabled is True
    assert config.user_dir == "~/.polaris/skills"
    assert config.project_dir == ".polaris/skills"


def test_resolve_skills_config_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text(
        "[skills]\n"
        "enabled = false\n"
        'user_dir = "~/custom"\n'
        'skills_dirs = ["a", "b"]\n'
        'disabled = ["commit"]\n',
        encoding="utf-8",
    )
    config = resolve_skills_config(toml)
    assert config.enabled is False
    assert config.user_dir == "~/custom"
    assert config.skills_dirs == ("a", "b")
    assert config.disabled == ("commit",)


def test_resolve_skills_config_unknown_keys_ignored(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text("[skills]\nbogus = 1\nenabled = true\n", encoding="utf-8")
    config = resolve_skills_config(toml)
    assert config.enabled is True  # unknown key ignored, valid one honoured


def test_resolve_skills_config_env_override(tmp_path: Path, monkeypatch) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text("[skills]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_SKILLS", "0")
    assert resolve_skills_config(toml).enabled is False

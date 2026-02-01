from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from upskill.cli import main


def test_delete_command_removes_skill_with_force() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        skill_dir = Path("skills/git-commit-messages")
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: git-commit-messages\n---\n")

        result = runner.invoke(main, ["delete", "git-commit-messages", "--force"])

        assert result.exit_code == 0
        assert not skill_dir.exists()


def test_delete_command_requires_confirmation_by_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        skill_dir = Path("skills/my-skill")
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n")

        result = runner.invoke(main, ["delete", "my-skill"], input="y\n")

        assert result.exit_code == 0
        assert not skill_dir.exists()


def test_delete_command_rejects_path_like_name() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["delete", "../oops", "--force"])

        assert result.exit_code == 1
        assert "must be a directory name, not a path" in result.output


def test_delete_command_rejects_non_skill_dir() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        not_skill = Path("skills/random-folder")
        not_skill.mkdir(parents=True)

        result = runner.invoke(main, ["delete", "random-folder", "--force"])

        assert result.exit_code == 1
        assert "does not look like a skill directory" in result.output

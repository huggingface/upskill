"""Verify the public CLI help text mentions CLI agent providers."""

from __future__ import annotations

from click.testing import CliRunner

from upskill.cli import main


def test_eval_help_mentions_cli_providers() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--help"])
    assert result.exit_code == 0
    assert "cli.claude-code" in result.output
    assert "cli.copilot" in result.output
    assert "cli.kiro" in result.output


def test_benchmark_help_mentions_cli_providers() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "cli.claude-code" in result.output


def test_generate_help_directs_users_for_cli_models() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["generate", "--help"])
    assert result.exit_code == 0
    # --eval-model is the v1 entry point for cli.* during generate.
    assert "cli.<provider>" in result.output or "cli.claude-code" in result.output
    assert "--eval-model" in result.output

"""Tests that verify CLI runners strip API-key env vars from subprocesses.

This is what keeps `claude` (and friends) on the user's subscription rather
than silently falling back to per-token API billing when an `*_API_KEY` env
var is set in the parent shell.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from upskill.cli_agents import CliProviderConfig, default_cli_providers
from upskill.cli_agents.base import (
    _resolve_subprocess_env,
    _warn_about_stripped_env,
    _warned_stripped_env,
)
from upskill.cli_agents.claude_code import ClaudeCodeRunner

if TYPE_CHECKING:
    from pathlib import Path


def _reset_warning_cache() -> None:
    _warned_stripped_env.clear()


def test_default_claude_provider_strips_anthropic_env() -> None:
    providers = default_cli_providers()
    claude = providers["claude-code"]
    assert "ANTHROPIC_API_KEY" in claude.unset_env
    assert "CLAUDE_CODE_USE_BEDROCK" in claude.unset_env
    assert "CLAUDE_CODE_USE_VERTEX" in claude.unset_env
    # Other providers stay quiet by default.
    assert providers["copilot"].unset_env == []
    assert providers["kiro"].unset_env == []


def test_resolve_subprocess_env_strips_listed_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OTHER_KEY", "stays")
    env = _resolve_subprocess_env(
        extra_env={"FOO": "bar"},
        unset_env=["ANTHROPIC_API_KEY"],
    )
    assert "ANTHROPIC_API_KEY" not in env
    assert env["OTHER_KEY"] == "stays"
    assert env["FOO"] == "bar"


def test_resolve_subprocess_env_also_strips_extra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if the user re-supplies the var via env=, unset_env wins."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = _resolve_subprocess_env(
        extra_env={"ANTHROPIC_API_KEY": "sk-ant-leak"},
        unset_env=["ANTHROPIC_API_KEY"],
    )
    assert "ANTHROPIC_API_KEY" not in env


def test_warn_about_stripped_env_writes_once_per_var(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reset_warning_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    _warn_about_stripped_env(runner_label="claude-code", unset_env=["ANTHROPIC_API_KEY"])
    _warn_about_stripped_env(runner_label="claude-code", unset_env=["ANTHROPIC_API_KEY"])

    captured = capsys.readouterr()
    # Exactly one stderr line, even after the second call.
    assert captured.err.count("unsetting ANTHROPIC_API_KEY") == 1
    assert "subscription" in captured.err
    assert "claude-code" in captured.err


def test_warn_does_nothing_when_var_not_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reset_warning_cache()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _warn_about_stripped_env(runner_label="claude-code", unset_env=["ANTHROPIC_API_KEY"])

    assert capsys.readouterr().err == ""


class _FakeProcess:
    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        del input
        return self._stdout, self._stderr

    def kill(self) -> None:
        return None


@pytest.mark.asyncio
async def test_claude_runner_actually_strips_anthropic_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ClaudeCodeRunner must not pass ANTHROPIC_API_KEY to claude."""
    _reset_warning_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    captured_envs: list[dict[str, str]] = []

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProcess:
        del args
        env = kwargs.get("env")
        if isinstance(env, dict):
            captured_envs.append(cast("dict[str, str]", env))
        return _FakeProcess(
            stdout=b'{"result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}',
            stderr=b"",
            returncode=0,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    runner = ClaudeCodeRunner()
    provider = default_cli_providers()["claude-code"]
    result = await runner.run(
        prompt="hi",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=provider,
        timeout=10,
    )

    assert result.error is None
    assert captured_envs, "subprocess was not invoked"
    forwarded = captured_envs[0]
    assert "ANTHROPIC_API_KEY" not in forwarded, (
        "Claude runner must strip ANTHROPIC_API_KEY so subscription billing is preserved."
    )


def test_user_can_override_unset_env_to_keep_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Power users running against a custom API endpoint may want to keep the key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = CliProviderConfig(command="claude", unset_env=[])
    env = _resolve_subprocess_env(
        extra_env=provider.env or None,
        unset_env=provider.unset_env or None,
    )
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"

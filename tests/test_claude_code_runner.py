"""Unit tests for the Claude Code CLI runner."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from upskill.cli_agents import CliProviderConfig, CliRunResult
from upskill.cli_agents.claude_code import ClaudeCodeRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_provider(args: list[str] | None = None) -> CliProviderConfig:
    return CliProviderConfig(command="claude", args=args or [], timeout_seconds=60)


class _FakeProcess:
    """Stand-in for ``asyncio.subprocess.Process``."""

    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        del input
        return self._stdout, self._stderr

    def kill(self) -> None:
        return None


def _patch_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    raise_file_not_found: bool = False,
    captured_argv: list[list[str]] | None = None,
) -> None:
    async def fake_create_subprocess_exec(*args: str, **_: object) -> _FakeProcess:
        if captured_argv is not None:
            captured_argv.append(list(args))
        if raise_file_not_found:
            raise FileNotFoundError(args[0])
        return _FakeProcess(stdout=stdout, stderr=stderr, returncode=returncode)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    return workspace, artifact


@pytest.mark.asyncio
async def test_claude_runner_parses_success_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    captured: list[list[str]] = []
    payload = json.dumps(
        {
            "result": "the answer",
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }
    ).encode("utf-8")
    _patch_subprocess(monkeypatch, stdout=payload, captured_argv=captured)

    runner = ClaudeCodeRunner()
    result = await runner.run(
        prompt="hello",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(args=["--model", "sonnet"]),
        timeout=10,
    )

    assert isinstance(result, CliRunResult)
    assert result.error is None
    assert result.output_text == "the answer"
    assert result.stats.input_tokens == 7
    assert result.stats.output_tokens == 11
    assert result.stats.total_tokens == 18
    assert result.return_code == 0

    # argv shape
    assert captured == [["claude", "-p", "hello", "--output-format", "json", "--model", "sonnet"]]
    # debug artifacts
    assert (artifact / "command.json").exists()
    assert (artifact / "stdout.txt").exists()
    assert (artifact / "stderr.txt").exists()


@pytest.mark.asyncio
async def test_claude_runner_handles_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(monkeypatch, stdout=b"not-json")

    runner = ClaudeCodeRunner()
    result = await runner.run(
        prompt="hello",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.output_text is None
    assert result.error is not None
    assert "valid JSON" in result.error
    assert result.return_code == 0


@pytest.mark.asyncio
async def test_claude_runner_surfaces_auth_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(
        monkeypatch,
        stdout=b"",
        stderr=b"Error: not authenticated. Run `claude login`.\n",
        returncode=1,
    )

    runner = ClaudeCodeRunner()
    result = await runner.run(
        prompt="hello",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.error is not None
    assert "exited with code 1" in result.error
    assert "claude login" in result.error
    assert result.return_code == 1


@pytest.mark.asyncio
async def test_claude_runner_handles_missing_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(monkeypatch, raise_file_not_found=True)

    runner = ClaudeCodeRunner()
    result = await runner.run(
        prompt="hello",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.output_text is None
    assert result.error is not None
    assert "not found on PATH" in result.error
    assert "claude login" in result.error  # auth hint appended
    assert result.return_code == -1

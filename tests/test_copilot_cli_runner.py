"""Unit tests for the Copilot CLI runner."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from upskill.cli_agents import CliProviderConfig
from upskill.cli_agents.copilot import CopilotCliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_provider(args: list[str] | None = None) -> CliProviderConfig:
    return CliProviderConfig(command="copilot", args=args or [], timeout_seconds=60)


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
async def test_copilot_runner_returns_stdout_and_marks_tokens_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    captured: list[list[str]] = []
    _patch_subprocess(
        monkeypatch,
        stdout=b"Here is the answer.\n",
        captured_argv=captured,
    )

    runner = CopilotCliRunner()
    result = await runner.run(
        prompt="explain rebase",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(args=["--allow-all-tools"]),
    )

    assert result.error is None
    assert result.output_text == "Here is the answer."
    assert result.stats.total_tokens == 0
    assert result.metadata["tokens_unavailable"] is True
    assert captured == [["copilot", "-p", "explain rebase", "--allow-all-tools"]]
    assert (artifact / "stdout.txt").exists()


@pytest.mark.asyncio
async def test_copilot_runner_reports_non_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(
        monkeypatch,
        stdout=b"",
        stderr=b"login required\n",
        returncode=2,
    )

    runner = CopilotCliRunner()
    result = await runner.run(
        prompt="x",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.output_text is None
    assert result.error is not None
    assert "exited with code 2" in result.error
    assert "gh auth login" in result.error
    assert result.metadata["tokens_unavailable"] is True


@pytest.mark.asyncio
async def test_copilot_runner_handles_missing_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(monkeypatch, raise_file_not_found=True)

    runner = CopilotCliRunner()
    result = await runner.run(
        prompt="x",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.error is not None
    assert "not found on PATH" in result.error
    assert "gh auth login" in result.error
    assert result.metadata["tokens_unavailable"] is True
    assert result.return_code == -1

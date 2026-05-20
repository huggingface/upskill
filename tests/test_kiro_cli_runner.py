"""Unit tests for the Kiro CLI runner."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from upskill.cli_agents import CliProviderConfig
from upskill.cli_agents.kiro import KiroCliRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_provider(args: list[str] | None = None) -> CliProviderConfig:
    return CliProviderConfig(command="kiro-cli", args=args or [], timeout_seconds=60)


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
async def test_kiro_runner_returns_stdout_with_default_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    captured: list[list[str]] = []
    _patch_subprocess(monkeypatch, stdout=b"Kiro response\n", captured_argv=captured)

    runner = KiroCliRunner()
    result = await runner.run(
        prompt="summarize this repo",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.error is None
    assert result.output_text == "Kiro response"
    assert result.metadata["tokens_unavailable"] is True
    # The default argv must NOT include --trust-all-tools (opt-in only).
    assert captured == [["kiro-cli", "chat", "--no-interactive", "summarize this repo"]]


@pytest.mark.asyncio
async def test_kiro_runner_includes_trust_all_tools_when_opted_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    captured: list[list[str]] = []
    _patch_subprocess(monkeypatch, stdout=b"ok\n", captured_argv=captured)

    runner = KiroCliRunner()
    await runner.run(
        prompt="task",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(args=["--trust-all-tools"]),
    )

    assert captured == [["kiro-cli", "chat", "--no-interactive", "--trust-all-tools", "task"]]


@pytest.mark.asyncio
async def test_kiro_runner_reports_non_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(
        monkeypatch,
        stdout=b"",
        stderr=b"not authenticated\n",
        returncode=3,
    )

    runner = KiroCliRunner()
    result = await runner.run(
        prompt="x",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.error is not None
    assert "exited with code 3" in result.error
    assert "kiro-cli login" in result.error


@pytest.mark.asyncio
async def test_kiro_runner_handles_missing_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, artifact = _make_dirs(tmp_path)
    _patch_subprocess(monkeypatch, raise_file_not_found=True)

    runner = KiroCliRunner()
    result = await runner.run(
        prompt="x",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=_make_provider(),
    )

    assert result.error is not None
    assert "not found on PATH" in result.error
    assert "kiro-cli login" in result.error

"""End-to-end tests that drive the CLI executor stack via fake CLI binaries.

These tests prove the full path -- router -> CliAgentExecutor -> runner ->
real subprocess -> result parsing -- works without depending on the real
``claude``/``copilot``/``kiro-cli`` binaries being present in CI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from upskill.cli_agents import CliProviderConfig
from upskill.config import Config
from upskill.executors.contracts import ExecutionRequest
from upskill.executors.router import build_cli_executor
from upskill.models import Skill

if TYPE_CHECKING:
    from upskill.executors.cli_agent import CliAgentExecutor


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "fake_cli_agents"


def _ensure_executable(*names: str) -> None:
    """Make sure fixture scripts have the executable bit set on this checkout."""
    for name in names:
        path = FIXTURE_DIR / name
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)


def _build_config_with_fakes() -> Config:
    """Build a Config whose cli_providers point at the fake fixture scripts."""
    _ensure_executable("fake_claude.py", "fake_copilot.py", "fake_kiro.py")
    config = Config()
    config.cli_providers = {
        "claude-code": CliProviderConfig(
            command=str(FIXTURE_DIR / "fake_claude.py"),
            timeout_seconds=30,
        ),
        "copilot": CliProviderConfig(
            command=str(FIXTURE_DIR / "fake_copilot.py"),
            timeout_seconds=30,
        ),
        "kiro": CliProviderConfig(
            command=str(FIXTURE_DIR / "fake_kiro.py"),
            timeout_seconds=30,
        ),
    }
    return config


def _make_request(*, model: str, tmp_path: Path) -> ExecutionRequest:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir(exist_ok=True)
    (cards_dir / "evaluator.md").write_text("---\nrole: evaluator\n---\nstub\n", encoding="utf-8")
    return ExecutionRequest(
        prompt="Translate 'hello' into a friendly message.",
        model=model,
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=tmp_path / "artifacts" / model,
        cards_source_dir=cards_dir,
        label=f"e2e-{model}",
        skill=Skill(
            name="example-skill",
            description="Greet users politely.",
            body="Always start with 'Hello,'.",
        ),
        workspace_files={},
        metadata={"phase": "e2e"},
    )


def _skip_if_no_fixture_python() -> None:
    if not os.access("/usr/bin/env", os.X_OK):  # pragma: no cover - macOS/linux only
        pytest.skip("Fixture shebangs require a POSIX env interpreter.")


@pytest.mark.asyncio
async def test_e2e_claude_code_runner_against_fake_binary(tmp_path: Path) -> None:
    _skip_if_no_fixture_python()
    config = _build_config_with_fakes()
    executor: CliAgentExecutor = build_cli_executor(config)
    request = _make_request(model="cli.claude-code", tmp_path=tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text is not None
    assert result.output_text.startswith("FAKE-CLAUDE:")
    assert result.stats.input_tokens == 13
    assert result.stats.output_tokens == 7
    assert result.stats.total_tokens == 20
    assert (request.artifact_dir / "stdout.txt").exists()
    assert (request.artifact_dir / "command.json").exists()


@pytest.mark.asyncio
async def test_e2e_copilot_runner_against_fake_binary(tmp_path: Path) -> None:
    _skip_if_no_fixture_python()
    config = _build_config_with_fakes()
    executor: CliAgentExecutor = build_cli_executor(config)
    request = _make_request(model="cli.copilot", tmp_path=tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text is not None
    assert result.output_text.startswith("FAKE-COPILOT:")
    # Copilot does not expose tokens.
    assert result.stats.total_tokens == 0
    assert result.metadata.get("tokens_unavailable") is True


@pytest.mark.asyncio
async def test_e2e_kiro_runner_against_fake_binary(tmp_path: Path) -> None:
    _skip_if_no_fixture_python()
    config = _build_config_with_fakes()
    executor: CliAgentExecutor = build_cli_executor(config)
    request = _make_request(model="cli.kiro", tmp_path=tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text is not None
    assert result.output_text.startswith("FAKE-KIRO:")
    # Kiro does not expose tokens.
    assert result.stats.total_tokens == 0
    assert result.metadata.get("tokens_unavailable") is True


@pytest.mark.asyncio
async def test_e2e_cli_executor_reports_missing_binary_clearly(tmp_path: Path) -> None:
    """If the configured binary is not on disk, the runner returns an actionable error."""
    config = Config()
    config.cli_providers["claude-code"] = CliProviderConfig(
        command="/this/path/does/not/exist/fake-claude",
        timeout_seconds=5,
    )
    executor = build_cli_executor(config)
    request = _make_request(model="cli.claude-code", tmp_path=tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.output_text is None
    assert result.error is not None
    assert "not found on PATH" in result.error
    assert "claude login" in result.error  # auth hint

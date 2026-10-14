"""Tests for ``CliAgentExecutor``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from upskill.cli_agents import (
    CliProviderConfig,
    CliRunResult,
)
from upskill.executors.cli_agent import (
    CliAgentExecutor,
    compose_cli_prompt,
    parse_cli_model,
)
from upskill.executors.contracts import ExecutionRequest
from upskill.models import ConversationStats, Skill

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class _FakeRunner:
    name: str = "claude-code"
    canned_output: str = "fake-output"
    return_code: int = 0
    error: str | None = None
    sleep_seconds: float = 0.0
    invocations: list[dict[str, object]] = field(default_factory=list)
    raise_inside: bool = False

    async def run(
        self,
        *,
        prompt: str,
        workspace_dir: Path,
        artifact_dir: Path,
        provider: CliProviderConfig,
        timeout: float | None = None,
    ) -> CliRunResult:
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_inside:
            raise RuntimeError("runner blew up")
        self.invocations.append(
            {
                "prompt": prompt,
                "workspace_dir": workspace_dir,
                "artifact_dir": artifact_dir,
                "provider_command": provider.command,
                "timeout": timeout,
            }
        )
        # Simulate the runner persisting stdout/stderr/command.json itself.
        (artifact_dir / "stdout.txt").write_text(self.canned_output, encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text("", encoding="utf-8")
        (artifact_dir / "command.json").write_text('{"argv": []}', encoding="utf-8")
        return CliRunResult(
            output_text=self.canned_output if self.return_code == 0 else None,
            stats=ConversationStats(input_tokens=2, output_tokens=3, total_tokens=5),
            return_code=self.return_code,
            stdout=self.canned_output,
            stderr="",
            error=self.error,
            metadata={"runner": self.name, "tokens_unavailable": False},
        )


def _build_request(tmp_path: Path, *, model: str = "cli.claude-code") -> ExecutionRequest:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "evaluator.md").write_text("---\nrole: evaluator\n---\nhi\n", encoding="utf-8")
    skill = Skill(
        name="example-skill",
        description="An example skill.",
        body="Always be concise.",
    )
    return ExecutionRequest(
        prompt="Solve the puzzle.",
        model=model,
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=tmp_path / "artifacts" / "test_1",
        cards_source_dir=cards_dir,
        label="test",
        skill=skill,
        workspace_files={"context.txt": "hello world"},
        metadata={"phase": "with-skill"},
    )


def _make_executor(runner_name: str = "claude-code") -> tuple[CliAgentExecutor, _FakeRunner]:
    runner = _FakeRunner(name=runner_name)
    executor = CliAgentExecutor(
        runners={runner_name: runner},
        providers={runner_name: CliProviderConfig(command="claude")},
    )
    return executor, runner


def test_parse_cli_model_extracts_provider_name() -> None:
    assert parse_cli_model("cli.claude-code") == "claude-code"
    assert parse_cli_model("cli.copilot") == "copilot"
    assert parse_cli_model("cli.kiro") == "kiro"


def test_parse_cli_model_rejects_non_cli_prefix() -> None:
    with pytest.raises(ValueError):
        parse_cli_model("sonnet")


def test_parse_cli_model_rejects_empty_suffix() -> None:
    with pytest.raises(ValueError):
        parse_cli_model("cli.")


def test_compose_cli_prompt_passes_through_when_no_skill(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    request_no_skill = ExecutionRequest(
        prompt="hi",
        model=request.model,
        agent=request.agent,
        fastagent_config_path=request.fastagent_config_path,
        artifact_dir=request.artifact_dir,
        cards_source_dir=request.cards_source_dir,
        label=request.label,
        skill=None,
    )
    assert compose_cli_prompt(request_no_skill) == "hi"


def test_compose_cli_prompt_prepends_skill_block(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    composed = compose_cli_prompt(request)
    assert composed.startswith("You have access to the following skill")
    assert "example-skill" in composed
    assert "Always be concise." in composed
    assert composed.endswith("Solve the puzzle.")


@pytest.mark.asyncio
async def test_executor_runs_request_and_persists_artifacts(tmp_path: Path) -> None:
    executor, runner = _make_executor()
    request = _build_request(tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text == "fake-output"
    assert result.stats.total_tokens == 5
    assert result.metadata["return_code"] == 0
    assert result.metadata["runner"] == "claude-code"
    assert result.metadata["phase"] == "with-skill"  # request metadata preserved

    artifact_dir = request.artifact_dir.resolve()
    assert (artifact_dir / "request.json").exists()
    assert (artifact_dir / "prompt.txt").exists()
    assert (artifact_dir / "stdout.txt").exists()
    assert (artifact_dir / "stderr.txt").exists()
    assert (artifact_dir / "results.json").exists()
    assert (artifact_dir / "workspace" / "context.txt").read_text(encoding="utf-8") == "hello world"
    assert (artifact_dir / "skills" / "example-skill" / "SKILL.md").exists()

    assert len(runner.invocations) == 1
    invoked = runner.invocations[0]
    assert invoked["workspace_dir"] == artifact_dir / "workspace"
    assert invoked["artifact_dir"] == artifact_dir
    assert invoked["provider_command"] == "claude"
    # The composed prompt was passed in
    assert "Solve the puzzle." in invoked["prompt"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_executor_propagates_runner_error(tmp_path: Path) -> None:
    runner = _FakeRunner(
        name="claude-code",
        canned_output="",
        return_code=2,
        error="Claude CLI exited with code 2: not authenticated",
    )
    executor = CliAgentExecutor(
        runners={"claude-code": runner},
        providers={"claude-code": CliProviderConfig(command="claude")},
    )
    request = _build_request(tmp_path)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error == "Claude CLI exited with code 2: not authenticated"
    assert result.output_text is None
    assert result.metadata["return_code"] == 2


@pytest.mark.asyncio
async def test_executor_returns_clear_error_for_unknown_provider(tmp_path: Path) -> None:
    executor, _ = _make_executor()
    request = _build_request(tmp_path, model="cli.kiro")

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.output_text is None
    assert result.error is not None
    assert "No CLI runner registered" in result.error
    assert "claude-code" in result.error
    # Even with an unknown provider, the canonical artifact layout exists.
    assert (request.artifact_dir.resolve() / "stdout.txt").exists()


@pytest.mark.asyncio
async def test_executor_supports_cancellation(tmp_path: Path) -> None:
    runner = _FakeRunner(name="claude-code", sleep_seconds=5)
    executor = CliAgentExecutor(
        runners={"claude-code": runner},
        providers={"claude-code": CliProviderConfig(command="claude")},
    )
    request = _build_request(tmp_path)

    handle = await executor.execute(request)
    await asyncio.sleep(0)
    await executor.cancel(handle)

    assert handle.task.cancelled() or handle.task.done()


def test_executor_requires_at_least_one_runner() -> None:
    with pytest.raises(ValueError):
        CliAgentExecutor(runners={}, providers={})

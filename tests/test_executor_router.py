"""Tests for the per-model executor router."""

from __future__ import annotations

import pytest

from upskill.cli_agents.claude_code import ClaudeCodeRunner
from upskill.cli_agents.copilot import CopilotCliRunner
from upskill.cli_agents.kiro import KiroCliRunner
from upskill.config import Config
from upskill.executors.cli_agent import CliAgentExecutor
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.executors.remote_fast_agent import RemoteFastAgentExecutor
from upskill.executors.router import (
    build_cli_executor,
    build_default_cli_runners,
    is_cli_model,
    select_executor_for_model,
)
from upskill.hf_jobs import JobsConfig


def _local_config() -> Config:
    return Config(executor="local")


def _jobs_config() -> JobsConfig:
    return JobsConfig(
        artifact_repo="user/repo",
        wait=True,
        jobs_timeout="1h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        jobs_image="ghcr.io/example/image:latest",
    )


def test_is_cli_model_only_matches_cli_prefix() -> None:
    assert is_cli_model("cli.claude-code") is True
    assert is_cli_model("cli.copilot") is True
    assert is_cli_model("cli.kiro") is True
    assert is_cli_model("sonnet") is False
    assert is_cli_model("openai.gpt-4") is False
    assert is_cli_model("generic.llama3") is False
    assert is_cli_model("clientside") is False  # close miss but no `.`


def test_build_default_cli_runners_covers_v1_set() -> None:
    runners = build_default_cli_runners()
    assert isinstance(runners["claude-code"], ClaudeCodeRunner)
    assert isinstance(runners["copilot"], CopilotCliRunner)
    assert isinstance(runners["kiro"], KiroCliRunner)


def test_build_cli_executor_uses_config_providers() -> None:
    config = Config()
    config.cli_providers["claude-code"].args = ["--model", "opus"]

    executor = build_cli_executor(config)

    assert isinstance(executor, CliAgentExecutor)


@pytest.mark.parametrize("model", ["cli.claude-code", "cli.copilot", "cli.kiro"])
def test_router_returns_cli_executor_for_cli_models(model: str) -> None:
    executor = select_executor_for_model(model, config=_local_config())
    assert isinstance(executor, CliAgentExecutor)


def test_router_returns_local_executor_for_fast_agent_alias() -> None:
    executor = select_executor_for_model("sonnet", config=_local_config())
    assert isinstance(executor, LocalFastAgentExecutor)


def test_router_returns_local_executor_for_provider_qualified_alias() -> None:
    executor = select_executor_for_model("openai.gpt-4", config=_local_config())
    assert isinstance(executor, LocalFastAgentExecutor)


def test_router_returns_remote_executor_when_executor_jobs_is_set() -> None:
    config = Config(executor="jobs")
    executor = select_executor_for_model(
        "sonnet",
        config=config,
        jobs_config=_jobs_config(),
    )
    assert isinstance(executor, RemoteFastAgentExecutor)


def test_router_still_routes_cli_models_to_cli_executor_under_jobs_config() -> None:
    """`executor: jobs` only governs fast-agent paths; CLI models stay local."""
    config = Config(executor="jobs")
    executor = select_executor_for_model(
        "cli.claude-code",
        config=config,
        jobs_config=_jobs_config(),
    )
    assert isinstance(executor, CliAgentExecutor)


def test_router_requires_jobs_config_when_executor_jobs_and_non_cli_model() -> None:
    config = Config(executor="jobs")
    with pytest.raises(ValueError):
        select_executor_for_model("sonnet", config=config)


def test_router_returns_clear_error_for_unknown_cli_provider() -> None:
    """Unknown providers do not crash the router; ``CliAgentExecutor`` surfaces the error."""
    executor = select_executor_for_model("cli.unknown-tool", config=_local_config())
    assert isinstance(executor, CliAgentExecutor)
    # The error path is exercised at `execute()` time and unit-tested in
    # tests/test_cli_agent_executor.py — we only assert the router itself is
    # tolerant here so a typo doesn't take the whole CLI down.

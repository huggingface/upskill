"""Per-model executor routing.

Decides which executor handles a given model string at runtime so a single
``upskill eval`` invocation can mix CLI-backed agents (``cli.claude-code``,
``cli.copilot``, ``cli.kiro``) with regular fast-agent models in the same
benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from upskill.cli_agents import CLI_PROVIDER_NAMES, CliAgentRunner, CliProviderConfig
from upskill.cli_agents.claude_code import ClaudeCodeRunner
from upskill.cli_agents.copilot import CopilotCliRunner
from upskill.cli_agents.kiro import KiroCliRunner
from upskill.executors.cli_agent import CLI_MODEL_PREFIX, CliAgentExecutor
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.executors.remote_fast_agent import RemoteFastAgentExecutor

if TYPE_CHECKING:
    from collections.abc import Callable

    from upskill.config import Config
    from upskill.executors.base import Executor
    from upskill.hf_jobs import JobsConfig


def is_cli_model(model: str) -> bool:
    """Return True when the model string targets a CLI agent provider."""
    return model.startswith(CLI_MODEL_PREFIX)


def build_default_cli_runners() -> dict[str, CliAgentRunner]:
    """Return the v1 set of registered CLI runners keyed by provider name."""
    runners: dict[str, CliAgentRunner] = {
        "claude-code": ClaudeCodeRunner(),
        "copilot": CopilotCliRunner(),
        "kiro": KiroCliRunner(),
    }
    # Sanity check: the runner registry must cover every provider name we
    # advertise as part of v1 so router lookups never silently miss.
    missing = set(CLI_PROVIDER_NAMES) - runners.keys()
    if missing:
        raise RuntimeError(f"build_default_cli_runners is missing runners for: {sorted(missing)}")
    return runners


def build_cli_executor(config: Config) -> CliAgentExecutor:
    """Build a ``CliAgentExecutor`` wired with the v1 runners and config providers."""
    providers: dict[str, CliProviderConfig] = dict(config.cli_providers)
    return CliAgentExecutor(runners=build_default_cli_runners(), providers=providers)


def select_executor_for_model(
    model: str,
    *,
    config: Config,
    jobs_config: JobsConfig | None = None,
    jobs_progress_callback: Callable[[str], None] | None = None,
) -> Executor:
    """Resolve the executor that should handle ``model`` for one evaluation pass.

    - ``cli.<provider>`` -> ``CliAgentExecutor``
    - otherwise, fall back to the user's configured fast-agent backend
      (``local`` -> ``LocalFastAgentExecutor`` or ``jobs`` ->
      ``RemoteFastAgentExecutor``).
    """
    if is_cli_model(model):
        return build_cli_executor(config)

    if config.executor == "jobs":
        if jobs_config is None:
            raise ValueError(
                "config.executor='jobs' but no JobsConfig was supplied to the router; "
                "build a JobsConfig from the CLI flags before routing."
            )
        return RemoteFastAgentExecutor(
            jobs_config=jobs_config,
            progress_callback=jobs_progress_callback,
        )

    return LocalFastAgentExecutor()

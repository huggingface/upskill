"""CLI-backed agent runners (Claude Code, GitHub Copilot CLI, Kiro CLI).

These runners shell out to user-installed CLI binaries that are themselves agents,
bypassing fast-agent. They let users run upskill against their existing
subscription-backed CLIs instead of raw Anthropic/OpenAI API keys.
"""

from upskill.cli_agents.base import (
    CLI_PROVIDER_NAMES,
    CliAgentError,
    CliAgentRunner,
    CliProcessOutcome,
    CliProviderConfig,
    CliRunResult,
    default_cli_providers,
    run_cli_subprocess,
)
from upskill.cli_agents.fast_agent_adapter import CliFastAgentAdapter
from upskill.cli_agents.protocols import SkillGenerator

__all__ = [
    "CLI_PROVIDER_NAMES",
    "CliAgentError",
    "CliAgentRunner",
    "CliFastAgentAdapter",
    "CliProcessOutcome",
    "CliProviderConfig",
    "CliRunResult",
    "SkillGenerator",
    "default_cli_providers",
    "run_cli_subprocess",
]

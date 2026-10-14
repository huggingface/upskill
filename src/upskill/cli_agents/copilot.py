"""Runner that drives the GitHub Copilot CLI (``copilot``) in non-interactive mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from upskill.cli_agents.base import (
    CliProviderConfig,
    CliRunResult,
    execute_text_cli,
)

if TYPE_CHECKING:
    from pathlib import Path


_AUTH_HINT = (
    "Verify the GitHub Copilot CLI is installed and signed in: run "
    "`gh auth login` (with Copilot scope) and `copilot --version`, then retry. "
    "See README -> CLI providers for details."
)


def _build_copilot_argv(*, provider: CliProviderConfig, prompt: str) -> list[str]:
    """Build the canonical Copilot CLI argv for a single non-interactive prompt."""
    return [provider.command, "-p", prompt, *provider.args]


@dataclass(slots=True)
class CopilotCliRunner:
    """Invoke ``copilot`` for a single prompt and capture the textual reply."""

    name: str = "copilot"

    async def run(
        self,
        *,
        prompt: str,
        workspace_dir: Path,
        artifact_dir: Path,
        provider: CliProviderConfig,
        timeout: float | None = None,
    ) -> CliRunResult:
        return await execute_text_cli(
            runner_name=self.name,
            display_name="Copilot CLI",
            argv=_build_copilot_argv(provider=provider, prompt=prompt),
            provider=provider,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            timeout=timeout,
            auth_hint=_AUTH_HINT,
        )

"""Runner that drives the Kiro CLI (``kiro-cli``) in non-interactive mode."""

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
    "Verify the Kiro CLI is installed and signed in: run `kiro-cli login` "
    "(or follow the Kiro CLI auth flow) and `kiro-cli --version`, then retry. "
    "See README -> CLI providers for details."
)


def _build_kiro_argv(*, provider: CliProviderConfig, prompt: str) -> list[str]:
    """Build the canonical Kiro CLI argv for a single non-interactive prompt.

    The argv shape is ``kiro-cli chat --no-interactive [provider.args...] <prompt>``.
    Users that need ``--trust-all-tools`` or other flags should set them via
    ``cli_providers.kiro.args`` in ``upskill.config.yaml``.
    """
    return [
        provider.command,
        "chat",
        "--no-interactive",
        *provider.args,
        prompt,
    ]


@dataclass(slots=True)
class KiroCliRunner:
    """Invoke ``kiro-cli chat`` for a single prompt and capture the textual reply."""

    name: str = "kiro"

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
            display_name="Kiro CLI",
            argv=_build_kiro_argv(provider=provider, prompt=prompt),
            provider=provider,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            timeout=timeout,
            auth_hint=_AUTH_HINT,
        )

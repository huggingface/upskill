"""Executor that drives CLI-backed agent runners (Claude Code, Copilot, Kiro)."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from upskill.artifacts import (
    ensure_directory,
    materialize_skill_bundle,
    materialize_workspace,
    write_request_file,
)
from upskill.cli_agents import (
    CliAgentRunner,
    CliProviderConfig,
    CliRunResult,
)
from upskill.executors.contracts import ExecutionHandle, ExecutionRequest, ExecutionResult
from upskill.models import ConversationStats

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

CLI_MODEL_PREFIX = "cli."

ExecutionMetadataValue = str | int | float | bool | None


def parse_cli_model(model: str) -> str:
    """Extract the provider name from a ``cli.<provider>`` model string.

    Raises ``ValueError`` if ``model`` does not use the ``cli.`` prefix.
    """
    if not model.startswith(CLI_MODEL_PREFIX):
        raise ValueError(f"Model {model!r} is not a CLI agent model (expected `cli.<provider>`).")
    suffix = model[len(CLI_MODEL_PREFIX) :]
    if not suffix:
        raise ValueError(f"Model {model!r} is missing the provider suffix after `cli.`.")
    return suffix


def compose_cli_prompt(request: ExecutionRequest) -> str:
    """Build the prompt sent to a CLI runner.

    When the request carries a skill, the rendered SKILL.md is prepended as a
    system-style block; otherwise the request prompt is returned untouched.
    CLI tools don't share a uniform ``system prompt`` flag, so we inline the
    skill content here for parity across providers.
    """
    if request.skill is None:
        return request.prompt

    skill_block = request.skill.render().strip()
    return (
        "You have access to the following skill document. Use it to inform your answer.\n\n"
        f"{skill_block}\n\n"
        "---\n\n"
        f"{request.prompt}"
    )


def _coerce_cli_metadata(
    metadata: Mapping[str, object],
) -> dict[str, ExecutionMetadataValue]:
    """Filter CLI runner metadata down to ``ExecutionResult`` compatible types."""
    coerced: dict[str, ExecutionMetadataValue] = {}
    for key, value in metadata.items():
        # Use a tuple of types instead of the PEP 604 union form for the
        # broadest compatibility with isinstance() across runtimes and
        # static analyzers.
        if value is None or isinstance(value, (str, int, float, bool)):
            coerced[key] = value
    return coerced


class CliAgentExecutor:
    """Execute evaluation requests by shelling out to a CLI agent runner.

    A single executor instance can dispatch any registered ``cli.<provider>``
    model to its matching runner. The router constructed in
    ``upskill.executors.router`` is responsible for wiring the registry.
    """

    def __init__(
        self,
        *,
        runners: Mapping[str, CliAgentRunner],
        providers: Mapping[str, CliProviderConfig],
    ) -> None:
        if not runners:
            raise ValueError("CliAgentExecutor requires at least one runner registration.")
        self._runners = dict(runners)
        self._providers = dict(providers)

    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        """Start a CLI-backed execution for a single request."""
        task = asyncio.create_task(self._run_request(request))
        return ExecutionHandle(request=request, task=task)

    async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        """Wait for and collect the result of a CLI-backed execution."""
        return await handle.task

    async def cancel(self, handle: ExecutionHandle) -> None:
        """Cancel an in-flight CLI-backed execution."""
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            return

    def _resolve_runner(
        self,
        request: ExecutionRequest,
    ) -> tuple[CliAgentRunner, CliProviderConfig] | None:
        """Look up the runner + provider config for a request's model string."""
        try:
            provider_name = parse_cli_model(request.model)
        except ValueError:
            return None
        runner = self._runners.get(provider_name)
        provider = self._providers.get(provider_name)
        if runner is None or provider is None:
            return None
        return runner, provider

    def _materialize_artifacts(
        self,
        request: ExecutionRequest,
    ) -> tuple[Path, Path, str]:
        """Set up the canonical artifact layout and composed prompt for a request."""
        artifact_dir = ensure_directory(request.artifact_dir.resolve())
        workspace_dir = ensure_directory(artifact_dir / "workspace")
        materialize_workspace(workspace_dir, request.workspace_files)
        materialize_skill_bundle(artifact_dir / "skills", request)
        write_request_file(artifact_dir / "request.json", request)
        composed_prompt = compose_cli_prompt(request)
        (artifact_dir / "prompt.txt").write_text(composed_prompt, encoding="utf-8")
        return artifact_dir, workspace_dir, composed_prompt

    def _build_result(
        self,
        *,
        request: ExecutionRequest,
        artifact_dir: Path,
        workspace_dir: Path,
        cli_run: CliRunResult,
    ) -> ExecutionResult:
        """Convert a runner outcome into the canonical ``ExecutionResult``."""
        merged_metadata: dict[str, ExecutionMetadataValue] = {**request.metadata}
        merged_metadata.update(_coerce_cli_metadata(cli_run.metadata))
        merged_metadata["return_code"] = cli_run.return_code

        results_path = artifact_dir / "results.json"
        if cli_run.output_text is not None:
            results_path.write_text(
                json.dumps({"result": cli_run.output_text}, indent=2),
                encoding="utf-8",
            )

        return ExecutionResult(
            output_text=cli_run.output_text,
            raw_results_path=results_path if results_path.exists() else None,
            stdout_path=artifact_dir / "stdout.txt",
            stderr_path=artifact_dir / "stderr.txt",
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            stats=cli_run.stats,
            error=cli_run.error,
            metadata=merged_metadata,
        )

    async def _run_request(self, request: ExecutionRequest) -> ExecutionResult:
        artifact_dir, workspace_dir, composed_prompt = self._materialize_artifacts(request)

        resolved = self._resolve_runner(request)
        if resolved is None:
            error = (
                f"No CLI runner registered for model {request.model!r}. "
                f"Known providers: {sorted(self._runners.keys()) or 'none'}."
            )
            empty_result = CliRunResult(
                output_text=None,
                stats=ConversationStats(),
                return_code=-1,
                error=error,
                metadata={"runner": "missing"},
            )
            # Persist empty stdout/stderr so the canonical artifact layout is preserved.
            (artifact_dir / "stdout.txt").write_text("", encoding="utf-8")
            (artifact_dir / "stderr.txt").write_text("", encoding="utf-8")
            return self._build_result(
                request=request,
                artifact_dir=artifact_dir,
                workspace_dir=workspace_dir,
                cli_run=empty_result,
            )

        runner, provider = resolved
        cli_run = await runner.run(
            prompt=composed_prompt,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            provider=provider,
        )
        return self._build_result(
            request=request,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            cli_run=cli_run,
        )

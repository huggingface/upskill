"""Contract types shared by all CLI-backed agent runners."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from upskill.models import ConversationStats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# Stable identifiers for the v1 CLI providers. The CLI surface and router both
# rely on this set to resolve `cli.<name>` model strings.
CLI_PROVIDER_NAMES = ("claude-code", "copilot", "kiro")

CliMetadataValue = str | int | float | bool | None


class CliAgentError(RuntimeError):
    """Raised when a CLI runner cannot be invoked at all (missing/unauth binary)."""


@dataclass(slots=True, frozen=True)
class CliRunResult:
    """Result of invoking a CLI agent for a single prompt."""

    output_text: str | None
    stats: ConversationStats = field(default_factory=ConversationStats)
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    metadata: dict[str, CliMetadataValue] = field(default_factory=dict)


class CliProviderConfig(BaseModel):
    """Per-provider configuration loaded from ``upskill.config.yaml``."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        ...,
        min_length=1,
        description="Executable to invoke (e.g. 'claude', 'copilot', 'kiro-cli').",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Extra arguments appended to the runner-built argv.",
    )
    timeout_seconds: int | None = Field(
        default=600,
        ge=1,
        description="Per-invocation timeout in seconds (None disables the timeout).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables merged into the subprocess environment.",
    )
    unset_env: list[str] = Field(
        default_factory=list,
        description=(
            "Environment variables to remove from the subprocess environment before "
            "launching the CLI. Used to force CLIs onto subscription auth instead "
            "of API-key auth (e.g. ANTHROPIC_API_KEY for `claude`)."
        ),
    )


@runtime_checkable
class CliAgentRunner(Protocol):
    """Strategy interface implemented by each CLI-specific runner."""

    name: str

    async def run(
        self,
        *,
        prompt: str,
        workspace_dir: Path,
        artifact_dir: Path,
        provider: CliProviderConfig,
        timeout: float | None = None,
    ) -> CliRunResult:
        """Invoke the underlying CLI and return a structured result."""


# Env vars that, if present, route the `claude` CLI to per-token API billing
# instead of the user's Claude Code / Pro subscription. Stripping them by
# default means upskill never silently spends a paid API quota when a user
# explicitly configured cli.claude-code.
_CLAUDE_API_FALLBACK_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)


def default_cli_providers() -> dict[str, CliProviderConfig]:
    """Return the v1 default ``cli_providers`` map.

    Defaults are conservative so users can run ``--model cli.claude-code`` (and
    siblings) with zero config, assuming the binary is on ``PATH`` and already
    authenticated.

    Each provider's ``unset_env`` defaults to the env vars known to override
    the CLI's subscription-based auth, so subscription billing is preserved
    without the user having to think about it.
    """
    return {
        "claude-code": CliProviderConfig(
            command="claude",
            unset_env=list(_CLAUDE_API_FALLBACK_ENV),
        ),
        "copilot": CliProviderConfig(command="copilot"),
        "kiro": CliProviderConfig(command="kiro-cli"),
    }


@dataclass(slots=True, frozen=True)
class CliProcessOutcome:
    """Captured stdout/stderr and exit code from a CLI subprocess."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _resolve_subprocess_env(
    extra_env: Mapping[str, str] | None,
    unset_env: Sequence[str] | None = None,
) -> dict[str, str]:
    """Merge optional provider env additions on top of ``os.environ``.

    Variables listed in ``unset_env`` are removed from the resulting
    environment regardless of whether they came from ``os.environ`` or
    ``extra_env``. This is what keeps subscription-backed CLI runners from
    silently falling back to API-key billing.
    """
    merged = dict(os.environ)
    if extra_env:
        merged.update(extra_env)
    if unset_env:
        for name in unset_env:
            merged.pop(name, None)
    return merged


_warned_stripped_env: set[tuple[str, str]] = set()


def _warn_about_stripped_env(
    *,
    runner_label: str,
    unset_env: Sequence[str] | None,
) -> None:
    """Emit a one-shot stderr note when a runner is stripping a set env var.

    Helps users discover that upskill is preserving subscription billing on
    their behalf (otherwise the disappearance is invisible).
    """
    if not unset_env:
        return
    for name in unset_env:
        if name not in os.environ:
            continue
        signature = (runner_label, name)
        if signature in _warned_stripped_env:
            continue
        _warned_stripped_env.add(signature)
        sys.stderr.write(
            f"upskill: {runner_label}: unsetting {name} for this subprocess so the "
            f"CLI uses your subscription instead of API-key billing. "
            f"Override via cli_providers.<provider>.unset_env in upskill.config.yaml.\n"
        )


async def run_cli_subprocess(
    *,
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
    unset_env: Sequence[str] | None = None,
    timeout_seconds: float | None = None,
    stdin_input: str | None = None,
    artifact_dir: Path | None = None,
    runner_label: str | None = None,
) -> CliProcessOutcome:
    """Run a CLI subprocess, capture stdout/stderr, and persist debug artifacts.

    Raises ``CliAgentError`` when the binary cannot be found or executed.
    """
    if not argv:
        raise CliAgentError("Cannot run CLI subprocess with empty argv.")

    if runner_label is not None:
        _warn_about_stripped_env(runner_label=runner_label, unset_env=unset_env)

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "command.json").write_text(
            json.dumps(
                {
                    "argv": list(argv),
                    "cwd": str(cwd),
                    "unset_env": list(unset_env or []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE
            if stdin_input is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_resolve_subprocess_env(env, unset_env=unset_env),
        )
    except FileNotFoundError as exc:
        raise CliAgentError(
            f"CLI binary {argv[0]!r} was not found on PATH. "
            f"Install it or override the `command` in `cli_providers` config."
        ) from exc

    stdin_bytes = stdin_input.encode("utf-8") if stdin_input is not None else None
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=stdin_bytes),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    return_code = process.returncode if process.returncode is not None else -1

    if artifact_dir is not None:
        (artifact_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

    return CliProcessOutcome(
        return_code=return_code,
        stdout=stdout_text,
        stderr=stderr_text,
        timed_out=timed_out,
    )


def _resolve_effective_timeout(
    *,
    provider: CliProviderConfig,
    explicit_timeout: float | None,
) -> float | None:
    """Return the effective subprocess timeout, preferring explicit > provider config."""
    if explicit_timeout is not None:
        return explicit_timeout
    if provider.timeout_seconds is None:
        return None
    return float(provider.timeout_seconds)


def _build_text_metadata(
    *,
    runner_name: str,
    argv: Sequence[str],
    timed_out: bool,
) -> dict[str, CliMetadataValue]:
    return {
        "runner": runner_name,
        "argv": " ".join(argv),
        "timed_out": timed_out,
        "tokens_unavailable": True,
    }


async def execute_text_cli(
    *,
    runner_name: str,
    display_name: str,
    argv: Sequence[str],
    provider: CliProviderConfig,
    workspace_dir: Path,
    artifact_dir: Path,
    timeout: float | None,
    auth_hint: str,
) -> CliRunResult:
    """Run a CLI that emits a plain-text reply, with no usage/token reporting.

    Used by runners that drive CLIs (Copilot, Kiro) which do not currently
    expose structured tokens or JSON envelopes. Tokens are recorded as zero
    and ``metadata['tokens_unavailable']`` is set so the rest of upskill can
    flag the gap when plotting/aggregating.
    """
    effective_timeout = _resolve_effective_timeout(
        provider=provider,
        explicit_timeout=timeout,
    )

    try:
        outcome = await run_cli_subprocess(
            argv=argv,
            cwd=workspace_dir,
            env=provider.env or None,
            unset_env=provider.unset_env or None,
            timeout_seconds=effective_timeout,
            artifact_dir=artifact_dir,
            runner_label=runner_name,
        )
    except CliAgentError as exc:
        return CliRunResult(
            output_text=None,
            stats=ConversationStats(),
            return_code=-1,
            stdout="",
            stderr="",
            error=f"{exc} {auth_hint}",
            metadata=_build_text_metadata(
                runner_name=runner_name,
                argv=argv,
                timed_out=False,
            ),
        )

    metadata = _build_text_metadata(
        runner_name=runner_name,
        argv=argv,
        timed_out=outcome.timed_out,
    )

    if outcome.timed_out:
        return CliRunResult(
            output_text=None,
            stats=ConversationStats(),
            return_code=outcome.return_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            error=f"{display_name} timed out before producing a response.",
            metadata=metadata,
        )

    if outcome.return_code != 0:
        error_detail = outcome.stderr.strip() or outcome.stdout.strip() or "no output"
        error = (
            f"{display_name} exited with code {outcome.return_code}: {error_detail}. {auth_hint}"
        )
        return CliRunResult(
            output_text=None,
            stats=ConversationStats(),
            return_code=outcome.return_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            error=error,
            metadata=metadata,
        )

    output_text = outcome.stdout.strip() or None
    return CliRunResult(
        output_text=output_text,
        stats=ConversationStats(),
        return_code=outcome.return_code,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        error=None if output_text else f"{display_name} produced no stdout.",
        metadata=metadata,
    )

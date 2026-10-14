"""Runner that drives the Claude Code CLI (``claude``) in non-interactive mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from upskill.cli_agents.base import (
    CliAgentError,
    CliMetadataValue,
    CliProviderConfig,
    CliRunResult,
    run_cli_subprocess,
)
from upskill.models import ConversationStats

if TYPE_CHECKING:
    from pathlib import Path


_AUTH_HINT = (
    "Verify Claude Code is installed and authenticated: run `claude login`, "
    "then retry. See README -> CLI providers for details."
)


def _coerce_int(value: object) -> int:
    """Best-effort int coercion that tolerates float/string usage values."""
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _parse_claude_envelope(stdout: str) -> tuple[str | None, ConversationStats, dict[str, object]]:
    """Parse Claude Code's ``--output-format json`` payload."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CliAgentError(f"Claude CLI did not return valid JSON: {exc}") from exc

    if not isinstance(envelope, dict):
        raise CliAgentError("Claude CLI JSON envelope must be an object.")

    output_text = envelope.get("result")
    if output_text is not None and not isinstance(output_text, str):
        raise CliAgentError("Claude CLI 'result' field must be a string when present.")

    usage_raw = envelope.get("usage")
    stats = ConversationStats()
    if isinstance(usage_raw, dict):
        input_tokens = _coerce_int(usage_raw.get("input_tokens", 0))
        output_tokens = _coerce_int(usage_raw.get("output_tokens", 0))
        stats = ConversationStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    return output_text, stats, envelope


def _build_claude_argv(
    *,
    provider: CliProviderConfig,
    prompt: str,
) -> list[str]:
    """Build the canonical Claude Code CLI argv for a single prompt."""
    return [
        provider.command,
        "-p",
        prompt,
        "--output-format",
        "json",
        *provider.args,
    ]


@dataclass(slots=True)
class ClaudeCodeRunner:
    """Invoke Claude Code (``claude``) for a single prompt and capture its reply."""

    name: str = "claude-code"

    async def run(
        self,
        *,
        prompt: str,
        workspace_dir: Path,
        artifact_dir: Path,
        provider: CliProviderConfig,
        timeout: float | None = None,
    ) -> CliRunResult:
        argv = _build_claude_argv(provider=provider, prompt=prompt)
        effective_timeout = timeout
        if effective_timeout is None and provider.timeout_seconds is not None:
            effective_timeout = float(provider.timeout_seconds)

        try:
            outcome = await run_cli_subprocess(
                argv=argv,
                cwd=workspace_dir,
                env=provider.env or None,
                unset_env=provider.unset_env or None,
                timeout_seconds=effective_timeout,
                artifact_dir=artifact_dir,
                runner_label=self.name,
            )
        except CliAgentError as exc:
            return CliRunResult(
                output_text=None,
                stats=ConversationStats(),
                return_code=-1,
                stdout="",
                stderr="",
                error=f"{exc} {_AUTH_HINT}",
                metadata={"runner": self.name, "argv": " ".join(argv)},
            )

        metadata: dict[str, CliMetadataValue] = {
            "runner": self.name,
            "argv": " ".join(argv),
            "timed_out": outcome.timed_out,
        }

        if outcome.timed_out:
            return CliRunResult(
                output_text=None,
                stats=ConversationStats(),
                return_code=outcome.return_code,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                error="Claude CLI timed out before producing a response.",
                metadata=metadata,
            )

        output_text: str | None = None
        stats = ConversationStats()
        envelope_error: str | None = None
        try:
            output_text, stats, envelope = _parse_claude_envelope(outcome.stdout)
            metadata["envelope_keys"] = ",".join(sorted(envelope.keys()))
        except CliAgentError as exc:
            envelope_error = str(exc)

        if outcome.return_code != 0:
            error_detail = outcome.stderr.strip() or outcome.stdout.strip() or "no output"
            error = (
                f"Claude CLI exited with code {outcome.return_code}: {error_detail}. {_AUTH_HINT}"
            )
        elif envelope_error:
            error = envelope_error
        else:
            error = None

        return CliRunResult(
            output_text=output_text,
            stats=stats,
            return_code=outcome.return_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            error=error,
            metadata=metadata,
        )

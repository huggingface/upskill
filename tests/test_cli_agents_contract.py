"""Contract tests for the CLI agent runner protocol and config schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from upskill.cli_agents import (
    CLI_PROVIDER_NAMES,
    CliAgentRunner,
    CliProviderConfig,
    CliRunResult,
    default_cli_providers,
)
from upskill.config import Config
from upskill.models import ConversationStats

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class FakeCliRunner:
    """Minimal in-memory implementation of the ``CliAgentRunner`` protocol."""

    name: str = "fake"
    canned_output: str = "ok"
    invocations: list[dict[str, object]] = field(default_factory=list)

    async def run(
        self,
        *,
        prompt: str,
        workspace_dir: Path,
        artifact_dir: Path,
        provider: CliProviderConfig,
        timeout: float | None = None,
    ) -> CliRunResult:
        self.invocations.append(
            {
                "prompt": prompt,
                "workspace_dir": workspace_dir,
                "artifact_dir": artifact_dir,
                "provider": provider,
                "timeout": timeout,
            }
        )
        return CliRunResult(
            output_text=self.canned_output,
            stats=ConversationStats(input_tokens=1, output_tokens=2, total_tokens=3),
            return_code=0,
            stdout=self.canned_output,
            stderr="",
            metadata={"runner": self.name},
        )


def test_default_cli_providers_covers_v1_set() -> None:
    providers = default_cli_providers()
    assert set(providers.keys()) == set(CLI_PROVIDER_NAMES)
    assert providers["claude-code"].command == "claude"
    assert providers["copilot"].command == "copilot"
    assert providers["kiro"].command == "kiro-cli"
    for provider in providers.values():
        assert provider.timeout_seconds is not None and provider.timeout_seconds > 0
        assert provider.args == []
        assert provider.env == {}


def test_cli_provider_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CliProviderConfig.model_validate({"command": "claude", "unknown_key": True})


def test_cli_provider_config_rejects_blank_command() -> None:
    with pytest.raises(ValidationError):
        CliProviderConfig.model_validate({"command": ""})


def test_cli_provider_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValidationError):
        CliProviderConfig.model_validate({"command": "claude", "timeout_seconds": 0})


def test_config_defaults_include_cli_providers() -> None:
    config = Config()
    assert set(config.cli_providers.keys()) == set(CLI_PROVIDER_NAMES)
    assert config.cli_providers["claude-code"].command == "claude"


def test_config_load_merges_user_overrides_with_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "upskill.config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "skill_generation_model": "sonnet",
                "cli_providers": {
                    "claude-code": {
                        "command": "claude",
                        "args": ["--model", "opus"],
                        "timeout_seconds": 1200,
                    },
                    "custom-tool": {
                        "command": "/usr/local/bin/custom",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = Config.load()

    assert config.cli_providers["claude-code"].args == ["--model", "opus"]
    assert config.cli_providers["claude-code"].timeout_seconds == 1200
    # Defaults that were not overridden survive.
    assert config.cli_providers["copilot"].command == "copilot"
    assert config.cli_providers["kiro"].command == "kiro-cli"
    # User-defined extra entry is preserved.
    assert config.cli_providers["custom-tool"].command == "/usr/local/bin/custom"


def test_config_load_rejects_malformed_cli_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "upskill.config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cli_providers": {
                    "claude-code": "not-a-mapping",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TypeError):
        Config.load()


def test_config_save_round_trips_cli_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = Config()
    config.cli_providers["claude-code"] = CliProviderConfig(
        command="claude",
        args=["--model", "haiku"],
        timeout_seconds=300,
    )
    config.save()

    saved_path = tmp_path / "upskill.config.yaml"
    saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    assert saved["cli_providers"]["claude-code"]["args"] == ["--model", "haiku"]

    reloaded = Config.load()
    assert reloaded.cli_providers["claude-code"].args == ["--model", "haiku"]
    assert reloaded.cli_providers["claude-code"].timeout_seconds == 300


@pytest.mark.asyncio
async def test_fake_runner_satisfies_protocol(tmp_path: Path) -> None:
    runner: CliAgentRunner = FakeCliRunner(name="fake")
    provider = CliProviderConfig(command="echo", args=[], timeout_seconds=10)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    result = await runner.run(
        prompt="hi",
        workspace_dir=workspace,
        artifact_dir=artifact,
        provider=provider,
        timeout=10,
    )

    assert isinstance(result, CliRunResult)
    assert result.output_text == "ok"
    assert result.stats.total_tokens == 3
    assert result.metadata["runner"] == "fake"

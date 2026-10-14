"""Configuration management for upskill."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from upskill.cli_agents import CliProviderConfig, default_cli_providers

UPSKILL_CONFIG_FILE = "upskill.config.yaml"
LEGACY_CONFIG_FILE = "config.yaml"
UPSKILL_CONFIG_ENV = "UPSKILL_CONFIG"


@dataclass(frozen=True)
class UpskillConfigPathResolution:
    """Resolved upskill config path and where it was found."""

    path: Path | None
    source: str
    exists: bool


def get_config_dir() -> Path:
    """Get the upskill config directory."""
    return Path.home() / ".config" / "upskill"


def get_local_config_path() -> Path:
    """Get the project-local upskill config path."""
    return Path.cwd() / UPSKILL_CONFIG_FILE


def get_legacy_config_path() -> Path:
    """Get the legacy user-level upskill config path."""
    return get_config_dir() / LEGACY_CONFIG_FILE


def find_upskill_config_path() -> Path | None:
    """Find upskill config path in priority order."""
    return resolve_upskill_config_path().path


def resolve_upskill_config_path() -> UpskillConfigPathResolution:
    """Find upskill config path in priority order.

    Priority:
      1. UPSKILL_CONFIG env var
      2. ./upskill.config.yaml (project local)
      3. ~/.config/upskill/config.yaml (legacy)
    """
    config_override = os.getenv(UPSKILL_CONFIG_ENV)
    if config_override:
        override_path = Path(config_override).expanduser()
        return UpskillConfigPathResolution(
            path=override_path,
            source=f"{UPSKILL_CONFIG_ENV} env var",
            exists=override_path.exists(),
        )

    local_config = get_local_config_path()
    if local_config.exists():
        return UpskillConfigPathResolution(
            path=local_config,
            source="project-local config",
            exists=True,
        )

    legacy_config = get_legacy_config_path()
    if legacy_config.exists():
        return UpskillConfigPathResolution(
            path=legacy_config,
            source="legacy user config",
            exists=True,
        )

    return UpskillConfigPathResolution(path=None, source="defaults", exists=False)


def get_default_skills_dir() -> Path:
    """Get the default skills directory (current working directory)."""
    return Path.cwd() / "skills"


def get_default_runs_dir() -> Path:
    """Get the default runs directory for logging."""
    return Path.cwd() / "runs"


def _merge_cli_providers(
    raw: object,
) -> dict[str, CliProviderConfig]:
    """Merge user-supplied ``cli_providers`` mapping with the v1 defaults.

    Users can override a single provider (e.g. `claude-code`) without losing
    the other built-in entries. Unknown keys are accepted so new CLIs can be
    declared in config without code changes (the router still has to know how
    to wire them — see ``executors/router.py``).
    """
    if raw is None:
        return default_cli_providers()
    if not isinstance(raw, dict):
        raise TypeError("`cli_providers` must be a mapping of provider name to config.")

    merged = default_cli_providers()
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError("`cli_providers` keys must be strings.")
        if isinstance(value, CliProviderConfig):
            merged[key] = value
            continue
        if not isinstance(value, dict):
            raise TypeError(
                f"`cli_providers.{key}` must be a mapping of fields, got {type(value).__name__}."
            )
        merged[key] = CliProviderConfig.model_validate(value)
    return merged


def find_config_path() -> Path:
    """Find the fastagent config file, checking cwd first then package root."""
    cwd_config = Path.cwd() / "fastagent.config.yaml"
    if cwd_config.exists():
        return cwd_config
    # Fall back to package-bundled config
    package_config = Path(__file__).parent.parent.parent / "fastagent.config.yaml"
    if package_config.exists():
        return package_config
    return cwd_config  # Return cwd path even if not exists (FastAgent will use defaults)


class Config(BaseModel):
    """upskill configuration."""

    model_config = ConfigDict(populate_by_name=True)

    # Model settings
    skill_generation_model: str = Field(
        default="sonnet",
        validation_alias=AliasChoices("skill_generation_model", "model"),
        description="Model for skill generation (FastAgent format)",
    )
    eval_model: str | None = Field(
        default=None,
        description="Model for evaluation (defaults to skill_generation_model)",
    )
    test_gen_model: str | None = Field(
        default=None,
        description="Model for test generation (defaults to skill generation model)",
    )

    # Directory settings
    skills_dir: Path = Field(
        default_factory=get_default_skills_dir, description="Where to save generated skills"
    )
    runs_dir: Path = Field(
        default_factory=get_default_runs_dir, description="Where to save run logs"
    )

    # Generation settings
    max_refine_attempts: int = Field(default=2, description="Max refinement iterations")

    # Execution settings
    executor: Literal["local", "jobs"] = Field(
        default="local",
        description="Default execution backend for evaluation and refinement",
    )
    num_runs: int | None = Field(
        default=None,
        ge=1,
        description="Default runs per model for eval/benchmark when CLI --runs is omitted",
    )
    max_parallel: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent evaluation executions per phase",
    )
    jobs_secrets: str = Field(
        default="HF_TOKEN",
        description="Comma-separated env var names to forward to HF Jobs when using executor=jobs",
    )
    jobs_image: str = Field(
        default="ghcr.io/astral-sh/uv:python3.13-bookworm",
        description="Container image used for HF Jobs when using executor=jobs",
    )

    # FastAgent settings
    fastagent_config: Path | None = Field(default=None, description="Path to fastagent.config.yaml")

    # CLI-backed agent providers (Claude Code, Copilot CLI, Kiro CLI, ...)
    cli_providers: dict[str, CliProviderConfig] = Field(
        default_factory=default_cli_providers,
        description=(
            "Per-provider configuration for cli.<provider> model aliases. "
            "Keys map to the suffix after `cli.`."
        ),
    )

    @classmethod
    def load(cls) -> Config:
        """Load config from file, or return defaults."""
        config_path = find_upskill_config_path()
        if config_path is None:
            return cls()

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Convert path strings to Path objects
            if "skills_dir" in data and isinstance(data["skills_dir"], str):
                data["skills_dir"] = Path(data["skills_dir"])
            if "runs_dir" in data and isinstance(data["runs_dir"], str):
                data["runs_dir"] = Path(data["runs_dir"])
            if "fastagent_config" in data and isinstance(data["fastagent_config"], str):
                data["fastagent_config"] = Path(data["fastagent_config"])
            if "cli_providers" in data:
                data["cli_providers"] = _merge_cli_providers(data["cli_providers"])
            return cls(**data)

        return cls()

    def save(self) -> None:
        """Save config to file."""
        config_path = find_upskill_config_path() or get_local_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        # Convert Path objects to strings for YAML
        data["skills_dir"] = str(self.skills_dir)
        data["runs_dir"] = str(self.runs_dir)
        if self.fastagent_config:
            data["fastagent_config"] = str(self.fastagent_config)
        data["cli_providers"] = {
            key: provider.model_dump(mode="json") for key, provider in self.cli_providers.items()
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    @property
    def effective_eval_model(self) -> str:
        """Get the model to use for evaluation."""
        return self.eval_model or self.skill_generation_model

    def effective_num_runs(self, command: Literal["eval", "benchmark"]) -> int:
        """Get the number of runs to use when CLI ``--runs`` is omitted."""
        if self.num_runs is not None:
            return self.num_runs
        if command == "benchmark":
            return 3
        return 1

    @property
    def model(self) -> str:
        """Backward-compatible alias for ``skill_generation_model``."""
        return self.skill_generation_model

    @property
    def effective_fastagent_config(self) -> Path:
        """Get the fastagent config path to use."""
        if self.fastagent_config and self.fastagent_config.exists():
            return self.fastagent_config
        return find_config_path()

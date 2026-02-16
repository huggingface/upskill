"""Configuration management for upskill."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

UPSKILL_CONFIG_FILE = "upskill.config.yaml"
LEGACY_CONFIG_FILE = "config.yaml"
UPSKILL_CONFIG_ENV = "UPSKILL_CONFIG"


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
    """Find upskill config path in priority order.

    Priority:
      1. UPSKILL_CONFIG env var
      2. ./upskill.config.yaml (project local)
      3. ~/.config/upskill/config.yaml (legacy)
    """
    config_override = os.getenv(UPSKILL_CONFIG_ENV)
    if config_override:
        override_path = Path(config_override).expanduser()
        if override_path.exists():
            return override_path

    local_config = get_local_config_path()
    if local_config.exists():
        return local_config

    legacy_config = get_legacy_config_path()
    if legacy_config.exists():
        return legacy_config

    return None


def get_default_skills_dir() -> Path:
    """Get the default skills directory (current working directory)."""
    return Path.cwd() / "skills"


def get_default_runs_dir() -> Path:
    """Get the default runs directory for logging."""
    return Path.cwd() / "runs"


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

    # Model settings
    model: str = Field(default="sonnet", description="Model for generation (FastAgent format)")
    eval_model: str | None = Field(default=None, description="Model for eval (defaults to model)")

    # Directory settings
    skills_dir: Path = Field(
        default_factory=get_default_skills_dir, description="Where to save generated skills"
    )
    runs_dir: Path = Field(
        default_factory=get_default_runs_dir, description="Where to save run logs"
    )

    # Generation settings
    auto_eval: bool = Field(default=True, description="Run eval after generation")
    max_refine_attempts: int = Field(default=2, description="Max refinement iterations")

    # FastAgent settings
    fastagent_config: Path | None = Field(default=None, description="Path to fastagent.config.yaml")

    @classmethod
    def load(cls) -> Config:
        """Load config from file, or return defaults."""
        config_path = find_upskill_config_path()
        if config_path is None:
            return cls()

        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            # Convert path strings to Path objects
            if "skills_dir" in data and isinstance(data["skills_dir"], str):
                data["skills_dir"] = Path(data["skills_dir"])
            if "runs_dir" in data and isinstance(data["runs_dir"], str):
                data["runs_dir"] = Path(data["runs_dir"])
            if "fastagent_config" in data and isinstance(data["fastagent_config"], str):
                data["fastagent_config"] = Path(data["fastagent_config"])
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
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @property
    def effective_eval_model(self) -> str:
        """Get the model to use for evaluation."""
        return self.eval_model or self.model

    @property
    def effective_fastagent_config(self) -> Path:
        """Get the fastagent config path to use."""
        if self.fastagent_config and self.fastagent_config.exists():
            return self.fastagent_config
        return find_config_path()

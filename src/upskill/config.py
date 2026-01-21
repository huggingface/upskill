"""Configuration management for upskill."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def get_config_dir() -> Path:
    """Get the upskill config directory."""
    return Path.home() / ".config" / "upskill"


def get_skills_dir() -> Path:
    """Get the default skills directory."""
    return get_config_dir() / "skills"


class Config(BaseModel):
    """upskill configuration."""

    model: str = Field(default="claude-sonnet-4-20250514", description="Model for generation")
    eval_model: str | None = Field(default=None, description="Model for eval (defaults to model)")
    skills_dir: Path = Field(default_factory=get_skills_dir)
    auto_eval: bool = Field(default=True, description="Run eval after generation")
    max_refine_attempts: int = Field(default=3, description="Max refinement iterations")

    @classmethod
    def load(cls) -> Config:
        """Load config from file, or return defaults."""
        config_path = get_config_dir() / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

    def save(self) -> None:
        """Save config to file."""
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    @property
    def effective_eval_model(self) -> str:
        """Get the model to use for evaluation."""
        return self.eval_model or self.model

"""Shared FastAgent wiring helpers for upskill."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fast_agent import FastAgent

if TYPE_CHECKING:
    from upskill.models import Skill

KNOWN_MODEL_PROVIDERS = ("anthropic.", "openai.", "generic.", "google.", "tensorzero.")


def get_servers_from_config(config_path: Path) -> list[str]:
    """Get all MCP server names from fastagent config.

    Args:
        config_path: Path to fastagent.config.yaml

    Returns:
        List of server names defined in the config
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        servers = config.get("mcp", {}).get("servers", {})
        return list(servers.keys())
    except Exception:
        return []


@contextmanager
def resolve_agent_cards_path() -> Iterator[Path]:
    """Resolve the packaged AgentCard directory for upskill."""
    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as path:
        yield path


def load_upskill_agent_cards(fast: FastAgent) -> None:
    """Load upskill's packaged AgentCards into a FastAgent instance."""
    with resolve_agent_cards_path() as cards_path:
        fast.load_agents(cards_path)


def apply_base_url(provider: str | None, base_url: str | None) -> str | None:
    """Apply base URL overrides using FastAgent's provider env vars.

    Returns the effective provider name used for model prefixing.
    """
    effective_provider = provider
    if base_url and not provider:
        effective_provider = "generic"

    if base_url:
        if effective_provider == "openai":
            os.environ["OPENAI_API_BASE"] = base_url
        elif effective_provider == "generic":
            os.environ["GENERIC_BASE_URL"] = base_url
        else:
            os.environ["ANTHROPIC_BASE_URL"] = base_url

    return effective_provider


def normalize_model(model: str | None, provider: str | None) -> str | None:
    """Prefix the model name with the provider when needed."""
    if model and provider and not model.startswith(KNOWN_MODEL_PROVIDERS):
        return f"{provider}.{model}"
    return model


def compose_instruction(instruction: str, skill: Skill | None) -> str:
    """Inject the skill content into an instruction when provided."""
    if not skill:
        return instruction
    return f"{instruction}\n\n## Skill: {skill.name}\n\n{skill.body}"


def build_agent_from_card(
    name: str,
    config_path: Path,
    *,
    agent_name: str,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    servers: Sequence[str] | None = None,
) -> FastAgent:
    """Create a FastAgent instance using a packaged AgentCard."""
    effective_provider = apply_base_url(provider, base_url)
    model_str = normalize_model(model, effective_provider)

    fast = FastAgent(
        name,
        ignore_unknown_args=True,
        parse_cli_args=False,
        config_path=str(config_path),
    )

    load_upskill_agent_cards(fast)

    agent_data = fast.agents.get(agent_name)
    if not agent_data:
        raise ValueError(f"AgentCard '{agent_name}' not found in upskill package")

    if model_str:
        agent_data["model"] = model_str
    if servers:
        agent_data["servers"] = list(servers)

    return fast

"""Shared FastAgent wiring helpers for upskill."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent import FastAgent

if TYPE_CHECKING:
    from upskill.models import Skill


@contextmanager
def resolve_agent_cards_path() -> Iterator[Path]:
    """Resolve the packaged AgentCard directory for upskill."""
    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as path:
        yield path

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
) -> FastAgent:
    """Create a FastAgent instance using a packaged AgentCard."""

    fast = FastAgent(
        name,
        ignore_unknown_args=True,
        parse_cli_args=False,
        config_path=str(config_path),
    )

    # cli argument
    if model:
        fast.args.model = model

    with resolve_agent_cards_path() as cards_path:
        fast.load_agents(cards_path)

    agent_data = fast.agents.get(agent_name)
    if not agent_data:
        raise ValueError(f"AgentCard '{agent_name}' not found in upskill package")

    return fast

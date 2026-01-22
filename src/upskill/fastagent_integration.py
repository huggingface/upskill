"""Shared FastAgent wiring helpers for upskill."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upskill.models import Skill


def compose_instruction(instruction: str, skill: Skill | None) -> str:
    """Inject the skill content into an instruction when provided."""
    if not skill:
        return instruction
    return f"{instruction}\n\n## Skill: {skill.name}\n\n{skill.body}"


"""Skill manifest utilities for upskill."""

from __future__ import annotations

import logging
from pathlib import Path

import frontmatter
from fast_agent.skills.registry import SkillManifest

logger = logging.getLogger(__name__)


def parse_skill_manifest_text(
    manifest_text: str,
    *,
    path: Path | None = None,
) -> tuple[SkillManifest | None, str | None]:
    """Parse a SkillManifest from raw SKILL.md content.

    Args:
        manifest_text: Raw SKILL.md content (frontmatter + body).
        path: Optional path for provenance/logging (defaults to in-memory).

    Returns:
        Tuple of (SkillManifest | None, error message | None).
    """
    manifest_path = path or Path("<in-memory>")
    try:
        post = frontmatter.loads(manifest_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to parse skill manifest",
            extra={"path": str(manifest_path), "error": str(exc)},
        )
        return None, str(exc)

    metadata = post.metadata or {}
    name = metadata.get("name")
    description = metadata.get("description")

    if not isinstance(name, str) or not name.strip():
        logger.warning("Skill manifest missing name", extra={"path": str(manifest_path)})
        return None, "Missing 'name' field"
    if not isinstance(description, str) or not description.strip():
        logger.warning(
            "Skill manifest missing description",
            extra={"path": str(manifest_path)},
        )
        return None, "Missing 'description' field"

    body_text = (post.content or "").strip()

    license_field = metadata.get("license")
    compatibility = metadata.get("compatibility")
    custom_metadata = metadata.get("metadata")
    allowed_tools_raw = metadata.get("allowed-tools")

    allowed_tools: list[str] | None = None
    if isinstance(allowed_tools_raw, str) and allowed_tools_raw.strip():
        allowed_tools = allowed_tools_raw.split()

    typed_metadata: dict[str, str] | None = None
    if isinstance(custom_metadata, dict):
        typed_metadata = {str(k): str(v) for k, v in custom_metadata.items()}

    return SkillManifest(
        name=name.strip(),
        description=description.strip(),
        body=body_text,
        path=manifest_path,
        license=license_field.strip() if isinstance(license_field, str) else None,
        compatibility=compatibility.strip() if isinstance(compatibility, str) else None,
        metadata=typed_metadata,
        allowed_tools=allowed_tools,
    ), None

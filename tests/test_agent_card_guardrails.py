from __future__ import annotations

from pathlib import Path

import pytest

AGENT_CARDS_DIR = Path("src/upskill/agent_cards")
GUARDED_CARDS = ("skill_gen.md", "test_gen.md")
# Intentional exceptions require both allowlist entry and frontmatter annotation.
ALLOWED_MODEL_PIN_OVERRIDES: dict[str, str] = {}


def _parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}

    frontmatter_lines: list[str] = []
    for line in lines[1:]:
        if line == "---":
            break
        frontmatter_lines.append(line)

    data: dict[str, str] = {}
    for line in frontmatter_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip()

    return data


@pytest.mark.parametrize("card_name", GUARDED_CARDS)
def test_guarded_agent_cards_do_not_pin_model_unless_explicitly_allowed(card_name: str) -> None:
    card_path = AGENT_CARDS_DIR / card_name
    assert card_path.exists(), f"Missing guarded agent card: {card_path}"

    frontmatter = _parse_frontmatter(card_path)
    if "model" not in frontmatter:
        return

    assert card_name in ALLOWED_MODEL_PIN_OVERRIDES, (
        f"Unexpected model pin in {card_name}. Remove `model:` from frontmatter or add an "
        "explicit temporary override in ALLOWED_MODEL_PIN_OVERRIDES with a justification."
    )
    assert frontmatter.get("allow_model_pin", "").lower() == "true", (
        f"{card_name} is allowlisted but missing `allow_model_pin: true` annotation in frontmatter."
    )


def test_default_guarded_cards_have_no_model_pin() -> None:
    """Regression guard: current default cards should not define a model pin."""
    for card_name in GUARDED_CARDS:
        frontmatter = _parse_frontmatter(AGENT_CARDS_DIR / card_name)
        assert "model" not in frontmatter, (
            f"Unexpected model pin in guarded card {card_name}. "
            "Model selection should come from runtime resolution."
        )

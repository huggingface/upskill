"""Pydantic models for skill schema and evaluation results."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SkillMetadata(BaseModel):
    """Metadata about how a skill was generated."""

    generated_by: str | None = None  # Model that created it
    generated_at: datetime | None = None
    source_task: str | None = None  # Original task description
    test_pass_rate: float | None = None


class Skill(BaseModel):
    """A generated agent skill following the SKILL.md spec."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    version: str = "1.0"
    license: str | None = None
    compatibility: str | None = Field(None, max_length=500)
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
    allowed_tools: list[str] | None = None

    # Content
    body: str  # Main instructions markdown
    references: dict[str, str] = Field(default_factory=dict)  # filename -> content
    scripts: dict[str, str] = Field(default_factory=dict)  # filename -> code

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError("name must be lowercase alphanumeric with hyphens")
        return v

    def render(self) -> str:
        """Generate valid SKILL.md content."""
        lines = [
            f"# {self.name}",
            "",
            self.description,
            "",
        ]

        if self.compatibility:
            lines.extend([f"Compatibility: {self.compatibility}", ""])

        lines.extend(["## Instructions", "", self.body])

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Write skill directory with all files."""
        path.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        (path / "SKILL.md").write_text(self.render())

        # Write references
        if self.references:
            refs_dir = path / "references"
            refs_dir.mkdir(exist_ok=True)
            for filename, content in self.references.items():
                (refs_dir / filename).write_text(content)

        # Write scripts
        if self.scripts:
            scripts_dir = path / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for filename, content in self.scripts.items():
                (scripts_dir / filename).write_text(content)


class TestCase(BaseModel):
    """A test case for skill evaluation."""

    input: str  # Task/prompt to give the agent
    context: dict | None = None  # Files, env vars, etc.
    expected: dict | None = None  # Expected output checks


class TestResult(BaseModel):
    """Result of running a single test case."""

    test_case: TestCase
    success: bool
    output: str | None = None
    tokens_used: int = 0
    turns: int = 0
    error: str | None = None


class EvalResults(BaseModel):
    """Results comparing skill vs baseline performance."""

    skill_name: str
    model: str

    # With skill
    with_skill_results: list[TestResult] = Field(default_factory=list)
    with_skill_success_rate: float = 0.0
    with_skill_total_tokens: int = 0
    with_skill_avg_turns: float = 0.0

    # Without skill (baseline)
    baseline_results: list[TestResult] = Field(default_factory=list)
    baseline_success_rate: float = 0.0
    baseline_total_tokens: int = 0
    baseline_avg_turns: float = 0.0

    @property
    def skill_lift(self) -> float:
        """Improvement in success rate from using skill."""
        return self.with_skill_success_rate - self.baseline_success_rate

    @property
    def token_savings(self) -> float:
        """Percentage of tokens saved (negative means more tokens used)."""
        if self.baseline_total_tokens == 0:
            return 0.0
        return 1 - (self.with_skill_total_tokens / self.baseline_total_tokens)

    @property
    def is_beneficial(self) -> bool:
        """Skill provides net benefit."""
        # Beneficial if: better success, OR same success with fewer tokens
        return self.skill_lift > 0.05 or (self.skill_lift >= 0 and self.token_savings > 0.2)

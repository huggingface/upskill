"""Pydantic models for skill schema, evaluation results, and run logging."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SkillMetadata(BaseModel):
    """Metadata about how a skill was generated (stored in skill_meta.json)."""

    version: str = "1.0"
    generated_by: str | None = None  # Model that created it
    generated_at: datetime | None = None
    source_task: str | None = None  # Original task description
    test_pass_rate: float | None = None
    license: str | None = None
    compatibility: str | None = None


class TestCase(BaseModel):
    """A test case for skill evaluation."""

    input: str  # Task/prompt to give the agent
    context: dict | None = None  # Files, env vars, etc.
    expected: dict | None = None  # Expected output checks


class Skill(BaseModel):
    """A generated agent skill following the Claude Code SKILL.md spec."""

    # Claude Code frontmatter fields
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    allowed_tools: list[str] | None = None
    argument_hint: str | None = None  # e.g., "[issue-number]"
    user_invocable: bool = True
    disable_model_invocation: bool = False

    # upskill metadata (persisted to skill_meta.json)
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)

    # Content
    body: str  # Main instructions markdown
    references: dict[str, str] = Field(default_factory=dict)  # filename -> content
    scripts: dict[str, str] = Field(default_factory=dict)  # filename -> code

    # Test cases (persisted to skill_meta.json)
    tests: list[TestCase] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError("name must be lowercase alphanumeric with hyphens")
        return v

    def render(self) -> str:
        """Generate Claude Code compatible SKILL.md with YAML frontmatter."""
        # Build frontmatter
        frontmatter_lines = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
        ]

        if self.allowed_tools:
            frontmatter_lines.append(f"allowed-tools: {', '.join(self.allowed_tools)}")

        if self.argument_hint:
            frontmatter_lines.append(f"argument-hint: {self.argument_hint}")

        if not self.user_invocable:
            frontmatter_lines.append("user-invocable: false")

        if self.disable_model_invocation:
            frontmatter_lines.append("disable-model-invocation: true")

        frontmatter_lines.append("---")

        return "\n".join(frontmatter_lines) + "\n\n" + self.body

    def save(self, path: Path, tests: list[TestCase] | None = None) -> None:
        """Write skill directory with all files.

        Args:
            path: Directory to save skill to
            tests: Optional test cases to persist (overrides self.tests if provided)
        """
        path.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md (Claude Code compatible)
        (path / "SKILL.md").write_text(self.render())

        # Write skill_meta.json (upskill-specific metadata + tests)
        tests_to_save = tests if tests is not None else self.tests
        meta_dict = {
            "metadata": self.metadata.model_dump(mode="json"),
            "tests": [t.model_dump(mode="json") for t in tests_to_save],
        }
        (path / "skill_meta.json").write_text(
            json.dumps(meta_dict, indent=2, default=str)
        )

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

    @classmethod
    def load(cls, path: Path) -> Skill:
        """Load a skill from a directory.

        Args:
            path: Directory containing SKILL.md and optionally skill_meta.json

        Returns:
            Loaded Skill instance
        """
        skill_md_path = path / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in {path}")

        content = skill_md_path.read_text()

        # Parse YAML frontmatter
        name = path.name  # Default to directory name
        description = ""
        allowed_tools: list[str] | None = None
        argument_hint: str | None = None
        user_invocable = True
        disable_model_invocation = False
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                body = parts[2].strip()

                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        if key == "name":
                            name = value
                        elif key == "description":
                            description = value
                        elif key == "allowed-tools":
                            allowed_tools = [t.strip() for t in value.split(",")]
                        elif key == "argument-hint":
                            argument_hint = value
                        elif key == "user-invocable":
                            user_invocable = value.lower() != "false"
                        elif key == "disable-model-invocation":
                            disable_model_invocation = value.lower() == "true"

        # Load metadata and tests from skill_meta.json if present
        metadata = SkillMetadata()
        tests: list[TestCase] = []
        meta_path = path / "skill_meta.json"
        if meta_path.exists():
            meta_dict = json.loads(meta_path.read_text())
            if "metadata" in meta_dict:
                metadata = SkillMetadata.model_validate(meta_dict["metadata"])
            if "tests" in meta_dict:
                tests = [TestCase.model_validate(t) for t in meta_dict["tests"]]

        # Load references
        references: dict[str, str] = {}
        refs_dir = path / "references"
        if refs_dir.exists():
            for ref_file in refs_dir.iterdir():
                if ref_file.is_file():
                    references[ref_file.name] = ref_file.read_text()

        # Load scripts
        scripts: dict[str, str] = {}
        scripts_dir = path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.iterdir():
                if script_file.is_file():
                    scripts[script_file.name] = script_file.read_text()

        return cls(
            name=name,
            description=description,
            allowed_tools=allowed_tools,
            argument_hint=argument_hint,
            user_invocable=user_invocable,
            disable_model_invocation=disable_model_invocation,
            metadata=metadata,
            body=body,
            references=references,
            scripts=scripts,
            tests=tests,
        )


# ConversationStats must be defined before TestResult (used in default_factory)
class ConversationStats(BaseModel):
    """Statistics from a conversation/run."""

    # Timing metrics
    conversation_span_ms: float = 0.0
    llm_time_ms: float = 0.0
    tool_time_ms: float = 0.0

    # Conversation metrics
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    # Tool categorization (MCP vs Execute)
    mcp_calls: int = 0  # Tools with "__" in name (MCP server tools)
    execute_calls: int = 0  # Tools without "__" (built-in execute)

    # Per-tool breakdown
    tool_call_map: dict[str, int] = Field(default_factory=dict)  # tool_name -> count
    tool_error_map: dict[str, int] = Field(default_factory=dict)  # tool_name -> error_count

    # Token metrics (detailed)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Legacy field for backwards compatibility
    @property
    def tokens(self) -> int:
        """Legacy accessor for total tokens."""
        return self.total_tokens


class TestResult(BaseModel):
    """Result of running a single test case."""

    test_case: TestCase
    success: bool
    output: str | None = None
    tokens_used: int = 0  # Legacy field, use stats.total_tokens
    turns: int = 0  # Legacy field, use stats.turns
    error: str | None = None
    stats: ConversationStats = Field(default_factory=ConversationStats)


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


# Run logging models (similar to skills-test)


class RunMetadata(BaseModel):
    """Metadata for a single run."""

    model: str
    task: str
    batch_id: str
    run_number: int
    timestamp: datetime = Field(default_factory=datetime.now)


class RunResult(BaseModel):
    """Complete result from a single run."""

    metadata: RunMetadata
    stats: ConversationStats = Field(default_factory=ConversationStats)
    passed: bool = False
    assertions_passed: int = 0
    assertions_total: int = 0
    error_message: str | None = None
    session_id: str | None = None
    session_history_file: str | None = None


class BatchSummary(BaseModel):
    """Summary of a batch of runs."""

    batch_id: str
    model: str
    task: str
    total_runs: int
    passed_runs: int
    results: list[RunResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate for the batch."""
        if self.total_runs == 0:
            return 0.0
        return self.passed_runs / self.total_runs

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens used across runs."""
        if not self.results:
            return 0.0
        return sum(r.stats.total_tokens for r in self.results) / len(self.results)

    @property
    def avg_llm_time_ms(self) -> float:
        """Calculate average LLM time across runs."""
        if not self.results:
            return 0.0
        return sum(r.stats.llm_time_ms for r in self.results) / len(self.results)

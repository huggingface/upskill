"""Skill generation from task descriptions using FastAgent."""

from __future__ import annotations

from datetime import UTC, datetime

from fast_agent.interfaces import AgentProtocol
from fast_agent.skills.registry import SkillManifest

from upskill.manifest_utils import parse_skill_manifest_text
from upskill.models import Skill, SkillMetadata, TestCase, TestCaseSuite

# Few-shot examples for test generation
TEST_EXAMPLES = """
## Example Test Cases

Task: "Write good git commit messages"

Output:
```json
{
  "cases": [
    {
      "input": "Write a commit message for adding a new login feature",
      "expected": {"contains": ["feat", "login"]}
    },
    {
      "input": "Write a commit message for fixing a null pointer bug in the user service",
      "expected": {"contains": ["fix", "bug"]}
    },
    {
      "input": "Write a commit message for updating the README documentation",
      "expected": {"contains": ["docs", "readme"]}
    },
    {
      "input": "Write a commit message for a breaking API change",
      "expected": {"contains": ["BREAKING", "api"]}
    }
  ]
}
```

Task: "Handle API errors gracefully in Python"

Output:
```json
{
  "cases": [
    {
      "input": "Write code to fetch data from an API with retry logic",
      "expected": {"contains": ["retry", "error"]}
    },
    {
      "input": "How should I handle a 500 error from an API?",
      "expected": {"contains": ["backoff", "500"]}
    },
    {
      "input": "Write error handling for a requests.get call",
      "expected": {"contains": ["except", "requests"]}
    }
  ]
}
```
"""

# Use TASK_PLACEHOLDER to avoid .format() issues with JSON braces
TASK_PLACEHOLDER = "___TASK___"

TEST_GENERATION_PROMPT = (
    "Generate 3-5 test cases for evaluating if an AI agent can perform this task well.\n\n"
    f"{TEST_EXAMPLES}\n\n"
    "## Your Task\n\n"
    f"Task: {TASK_PLACEHOLDER}\n\n"
    "Generate test cases that verify the agent can apply the skill correctly.\n\n"

    "Each TestCase MUST include at least a list of expected strings in the expected field.\n"
    "Focus on practical scenarios that test understanding of the core concepts."
)

def _build_skill_from_manifest(
    manifest: SkillManifest,
    *,
    model: str | None,
    source_task: str,
    base_skill: Skill | None = None,
) -> Skill:
    references = base_skill.references if base_skill else {}
    scripts = base_skill.scripts if base_skill else {}
    return Skill(
        name=manifest.name,
        description=manifest.description,
        body=manifest.body,
        ## treating these as future for now as skill generator doesn't generate additional artifacts
        references=references,
        scripts=scripts,
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=source_task,
        ),
    )


async def generate_skill(
    task: str,
    generator: AgentProtocol,
    examples: list[str] | None = None,
    model: str | None = None,
) -> Skill:
    """Generate a skill from a task description using FastAgent."""

    prompt = f"Create a skill document that teaches an AI agent how to: {task}"
    if examples:
        prompt += "\n\nExample input/output pairs for this task:\n" + "\n".join(
            f"- {ex}" for ex in examples
        )


    skill_text = await generator.send(prompt)
    manifest, error = parse_skill_manifest_text(skill_text)
    if manifest is None:
        raise ValueError(f"Skill generator did not return valid SKILL.md: {error}")

    return _build_skill_from_manifest(
        manifest,
        model=model,
        source_task=task,
    )


async def generate_tests(
    task: str,
    generator: AgentProtocol,
    model: str | None = None,
) -> list[TestCase]:
    """Generate synthetic test cases from a task description using FastAgent."""

    prompt = TEST_GENERATION_PROMPT.replace(TASK_PLACEHOLDER, task)

    result, _ = await generator.structured(prompt, TestCaseSuite)
    if result is None:
        raise ValueError("Test generator did not return structured test cases.")

    cases = result.cases
    invalid_expected = 0
    for tc in cases:
        expected_values = [value.strip() for value in tc.expected.contains if value.strip()]
        if len(expected_values) < 2:
            invalid_expected += 1

    print(
        "Generated test cases:",
        f"total={len(cases)}",
        f"invalid_expected={invalid_expected}",
    )
    if invalid_expected:
        print(
            "Warning: some test cases are missing at least two expected strings; "
            "review generated tests."
        )
    return cases


async def refine_skill(
    skill: Skill,
    failures: list[str],
    generator: AgentProtocol,
    model: str | None = None,
) -> Skill:
    """Refine a skill based on evaluation failures using FastAgent."""

    prompt = f"""Improve this skill based on failures:

Name: {skill.name}
Description: {skill.description}
Body: {skill.body[:500]}...

Failures:
{chr(10).join(f"- {f}" for f in failures[:3])}

Output a complete SKILL.md document with YAML frontmatter (name, description) and a markdown body.
Do not wrap the output in code fences.
"""

    skill_text = await generator.send(prompt)
    manifest, error = parse_skill_manifest_text(skill_text)
    if manifest is None:
        raise ValueError(f"Skill refinement did not return valid SKILL.md: {error}")

    return _build_skill_from_manifest(
        manifest,
        model=model,
        source_task=skill.metadata.source_task,
        base_skill=skill,
    )


IMPROVE_PROMPT = """You are improving an existing skill document for AI agents.

Given the current skill and improvement instructions, create an enhanced version.

## Current Skill

Name: {name}
Description: {description}

Body:
{body}

## Improvement Instructions

{instructions}

## Guidelines

1. Preserve what works well in the original skill
2. Address the specific improvement requests
3. Maintain the same general structure and format
4. Add new examples or sections as needed
5. Keep the skill focused and actionable

Output a complete SKILL.md document with YAML frontmatter (name, description) and a markdown body.
Do not wrap the output in code fences or JSON.
"""


async def improve_skill(
    skill: Skill,
    instructions: str,
    generator: AgentProtocol,
    model: str | None = None,
) -> Skill:
    """Improve an existing skill based on instructions.

    Args:
        skill: The existing skill to improve
        instructions: What improvements to make
        model: Model to use for skill generation
        config: Configuration

    Returns:
        Improved Skill object
    """
    # config = config or Config.load()
    # model = model or config.skill_generation_model

    prompt = IMPROVE_PROMPT.format(
        name=skill.name,
        description=skill.description,
        body=skill.body,
        instructions=instructions,
    )


    skill_text = await generator.send(prompt)
    manifest, error = parse_skill_manifest_text(skill_text)
    if manifest is None:
        raise ValueError(f"Skill improvement did not return valid SKILL.md: {error}")

    return _build_skill_from_manifest(
        manifest,
        model=model,
        source_task=f"Improved from {skill.name}: {instructions}",
        base_skill=skill,
    )

"""Skill generation from task descriptions using FastAgent."""

from __future__ import annotations

from datetime import UTC, datetime

from fast_agent.interfaces import AgentProtocol

from upskill.models import Skill, SkillDraft, SkillMetadata, TestCase, TestCaseSuite

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
      "expected": {"contains": "feat"}
    },
    {
      "input": "Write a commit message for fixing a null pointer bug in the user service",
      "expected": {"contains": "fix"}
    },
    {
      "input": "Write a commit message for updating the README documentation",
      "expected": {"contains": "docs"}
    },
    {
      "input": "Write a commit message for a breaking API change",
      "expected": {"contains": "BREAKING"}
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
      "expected": {"contains": "retry"}
    },
    {
      "input": "How should I handle a 500 error from an API?",
      "expected": {"contains": "backoff"}
    },
    {
      "input": "Write error handling for a requests.get call",
      "expected": {"contains": "except"}
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
    "Output ONLY a valid JSON object (no markdown code blocks):\n"
    "{\n"
    '  "cases": [\n'
    '    {"input": "prompt/question for the agent",\n'
    '     "expected": {"contains": "substring that should appear in good response"}}\n'
    "  ]\n"
    "}\n\n"
    "Focus on practical scenarios that test understanding of the core concepts."
)


async def generate_skill(
    task: str,
    generator: AgentProtocol,
    examples: list[str] | None = None,
    model: str | None = None,
) -> Skill:
    """Generate a skill from a task description using FastAgent."""
    # config = config or Config.load()
    # model = model or config.model
    # config_path = config.effective_fastagent_config

    prompt = f"Create a skill document that teaches an AI agent how to: {task}"
    if examples:
        prompt += "\n\nExample input/output pairs for this task:\n" + "\n".join(
            f"- {ex}" for ex in examples
        )


    last_error: Exception | None = None
    result: SkillDraft | None = None
    for attempt in range(2):
        result, _ = await generator.structured(prompt, SkillDraft)

        if result is not None:
            break

        last_error = ValueError("Skill generator did not return structured output.")
        if attempt == 0:
            prompt = f"{prompt}\n\nIMPORTANT: Follow the structured schema exactly."
            continue
        raise last_error

    if result is None:
        raise ValueError("No data returned from skill generator")

    return Skill(
        name=result.name,
        description=result.description,
        body=result.body,
        references=result.references or {},
        scripts=result.scripts or {},
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=task,
        ),
    )


async def generate_tests(
    task: str,
    generator: AgentProtocol,
    model: str | None = None,
) -> list[TestCase]:
    """Generate synthetic test cases from a task description using FastAgent."""
    # config = config or Config.load()
    # model = model or config.model
    # config_path = config.effective_fastagent_config

    prompt = TEST_GENERATION_PROMPT.replace(TASK_PLACEHOLDER, task)


    result, _ = await generator.structured(prompt, TestCaseSuite)

    if result is None:
        raise ValueError("Test generator did not return structured test cases.")

    return result.cases


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

Output improved skill as JSON (same structure, no code blocks)."""

    result, _ = await generator.structured(prompt, SkillDraft)

    if result is None:
        raise ValueError("Skill refinement did not return structured output.")

    return Skill(
        name=result.name or skill.name,
        description=result.description or skill.description,
        body=result.body,
        references=result.references if result.references is not None else skill.references,
        scripts=result.scripts if result.scripts is not None else skill.scripts,
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=skill.metadata.source_task,
        ),
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

Output the improved skill as JSON with this structure:
{{
  "name": "skill-name",
  "description": "Brief description of what this skill teaches.",
  "body": "The full skill document in markdown format..."
}}

Output ONLY the JSON, no code blocks or explanations."""


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
        model: Model to use for generation
        config: Configuration

    Returns:
        Improved Skill object
    """
    # config = config or Config.load()
    # model = model or config.model

    prompt = IMPROVE_PROMPT.format(
        name=skill.name,
        description=skill.description,
        body=skill.body,
        instructions=instructions,
    )


    result, _ = await generator.structured(prompt, SkillDraft)

    if result is None:
        raise ValueError("Skill improvement did not return structured output.")

    return Skill(
        name=result.name or skill.name,
        description=result.description or skill.description,
        body=result.body,
        references=result.references if result.references is not None else skill.references,
        scripts=result.scripts if result.scripts is not None else skill.scripts,
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=f"Improved from {skill.name}: {instructions}",
        ),
    )


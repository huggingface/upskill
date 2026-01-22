"""Skill generation from task descriptions using FastAgent."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from fast_agent import FastAgent

from upskill.config import Config
from upskill.fastagent_integration import build_agent_from_card
from upskill.models import Skill, SkillMetadata, TestCase

# Few-shot examples for test generation
TEST_EXAMPLES = """
## Example Test Cases

Task: "Write good git commit messages"

Output:
```json
[
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
```

Task: "Handle API errors gracefully in Python"

Output:
```json
[
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
    "Output ONLY valid JSON array (no markdown code blocks):\n"
    "[\n"
    '  {"input": "prompt/question for the agent",\n'
    '   "expected": {"contains": "substring that should appear in good response"}}\n'
    "]\n\n"
    "Focus on practical scenarios that test understanding of the core concepts."
)



def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling common issues."""
    content = content.strip()

    # Remove outer markdown code block if present
    if content.startswith("```json"):
        content = content[7:]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
    elif content.startswith("```"):
        content = content[3:]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]

    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to fix common issues: unescaped newlines in strings
    try:
        fixed = re.sub(
            r'(?<=": ")(.*?)(?="[,\}])',
            lambda m: m.group(0).replace("\n", "\\n").replace("\t", "\\t"),
            content,
            flags=re.DOTALL,
        )
        return json.loads(fixed)
    except (json.JSONDecodeError, re.error):
        pass

    # Last resort: try to extract partial JSON
    try:
        start = content.find("{")
        if start >= 0:
            for end in range(len(content), start + 2, -1):
                try:
                    substr = content[start:end]
                    open_braces = substr.count("{") - substr.count("}")
                    substr += "}" * open_braces
                    result = json.loads(substr)
                    if isinstance(result, dict) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON from response:\n{content[:500]}...")


def build_skill_generator(config_path: Path, model: str | None = None) -> FastAgent:
    """Create a FastAgent instance for skill generation."""
    return build_agent_from_card(
        "upskill-generator",
        config_path,
        agent_name="skill_gen",
        model=model,
    )


def build_test_generator(config_path: Path, model: str | None = None) -> FastAgent:
    """Create a FastAgent instance for test generation."""
    return build_agent_from_card(
        "upskill-test-generator",
        config_path,
        agent_name="test_gen",
        model=model,
    )


async def generate_skill(
    task: str,
    examples: list[str] | None = None,
    model: str | None = None,
    config: Config | None = None,
) -> Skill:
    """Generate a skill from a task description using FastAgent."""
    config = config or Config.load()
    model = model or config.model
    config_path = config.effective_fastagent_config

    prompt = f"Create a skill document that teaches an AI agent how to: {task}"
    if examples:
        prompt += "\n\nExample input/output pairs for this task:\n" + "\n".join(
            f"- {ex}" for ex in examples
        )

    fast = build_skill_generator(config_path, model)

    last_error = None
    data: dict[str, object] | None = None
    response = ""
    for attempt in range(2):
        async with fast.run() as agent:
            response = await agent.skill_gen.send(prompt)

        try:
            data = parse_json_response(response)
        except ValueError as e:
            last_error = e
            if attempt == 0:
                prompt = (
                    f"{prompt}\n\nIMPORTANT: Output only raw JSON, no code blocks."
                )
                continue
            raise last_error

        required = ["name", "description", "body"]
        missing = [k for k in required if k not in data]
        if missing:
            if attempt == 0:
                prompt = (
                    f"{prompt}\n\nIMPORTANT: Include all required fields: name, description, body."
                )
                continue
            preview = {
                k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                for k, v in data.items()
            }
            raise ValueError(
                "Missing required fields: "
                f"{missing}\nGot: {preview}\n\nRaw response:\n"
                f"{response[:1000]}"
            )
        break

    if data is None:
        raise ValueError("No data returned from skill generator")

    name = data.get("name")
    description = data.get("description")
    body = data.get("body")
    if not isinstance(name, str) or not isinstance(description, str) or not isinstance(body, str):
        preview = {
            k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
            for k, v in data.items()
        }
        raise ValueError(
            "Skill response contained non-string required fields."
            f"\nGot: {preview}\n\nRaw response:\n{response[:1000]}"
        )

    return Skill(
        name=name,
        description=description,
        body=body,
        references=data.get("references", {}),
        scripts=data.get("scripts", {}),
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=task,
        ),
    )


async def generate_tests(
    task: str, model: str | None = None, config: Config | None = None
) -> list[TestCase]:
    """Generate synthetic test cases from a task description using FastAgent."""
    config = config or Config.load()
    model = model or config.model
    config_path = config.effective_fastagent_config

    prompt = TEST_GENERATION_PROMPT.replace(TASK_PLACEHOLDER, task)

    fast = build_test_generator(config_path, model)

    async with fast.run() as agent:
        response = await agent.test_gen.send(prompt)

    data = parse_json_response(response)

    if isinstance(data, list):
        return [TestCase(**tc) for tc in data]
    return [TestCase(**tc) for tc in data.get("cases", data)]


async def refine_skill(
    skill: Skill,
    failures: list[str],
    model: str | None = None,
    config: Config | None = None,
) -> Skill:
    """Refine a skill based on evaluation failures using FastAgent."""
    config = config or Config.load()
    model = model or config.model
    config_path = config.effective_fastagent_config

    prompt = f"""Improve this skill based on failures:

Name: {skill.name}
Description: {skill.description}
Body: {skill.body[:500]}...

Failures:
{chr(10).join(f"- {f}" for f in failures[:3])}

Output improved skill as JSON (same structure, no code blocks)."""

    fast = build_skill_generator(config_path, model)

    async with fast.run() as agent:
        response = await agent.skill_gen.send(prompt)

    data = parse_json_response(response)

    return Skill(
        name=data.get("name", skill.name),
        description=data.get("description", skill.description),
        body=data["body"],
        references=data.get("references", skill.references),
        scripts=data.get("scripts", skill.scripts),
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
    model: str | None = None,
    config: Config | None = None,
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
    config = config or Config.load()
    model = model or config.model
    config_path = config.effective_fastagent_config

    prompt = IMPROVE_PROMPT.format(
        name=skill.name,
        description=skill.description,
        body=skill.body,
        instructions=instructions,
    )

    fast = build_skill_generator(config_path, model)

    async with fast.run() as agent:
        response = await agent.skill_gen.send(prompt)

    data = parse_json_response(response)

    return Skill(
        name=data.get("name", skill.name),
        description=data.get("description", skill.description),
        body=data["body"],
        references=data.get("references", skill.references),
        scripts=data.get("scripts", skill.scripts),
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(UTC),
            source_task=f"Improved from {skill.name}: {instructions}",
        ),
    )

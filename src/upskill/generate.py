"""Skill generation from task descriptions, tools, and traces."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from anthropic import AsyncAnthropic

from upskill.config import Config
from upskill.models import Skill, SkillMetadata, TestCase

# Meta-prompt for skill generation
GENERATION_PROMPT = """You generate "skills" - instruction documents that teach AI coding agents how to perform tasks.

When given a task description, you create a skill document with clear instructions, examples, and best practices that will help an AI agent complete that type of task reliably.

You must output ONLY a JSON object with this exact structure:
{"name": "skill-name", "description": "What this skill teaches", "body": "Markdown instructions"}

Field requirements:
- name: lowercase with hyphens (e.g., "parse-yaml-files")
- description: one sentence under 100 chars
- body: 200-400 word markdown guide with step-by-step instructions and 2-3 examples

IMPORTANT:
- Output ONLY the JSON object, no other text
- Use \\n for newlines in the body field
- Do NOT actually perform the task - create instructions FOR performing it"""

TEST_GENERATION_PROMPT = """Generate 3-5 test cases for this task.

Task: {task}

Output ONLY valid JSON array (no markdown):
[{{"input": "prompt", "expected": {{"contains": "substring"}}}}]"""


def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling common issues."""
    content = content.strip()

    # Remove outer markdown code block if present (but preserve inner ones in JSON strings)
    # Look for ```json at start and ``` at end
    if content.startswith("```json"):
        # Find the matching closing ``` by looking for the last one
        content = content[7:]  # Remove ```json
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]  # Remove trailing ```
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
    # Find string values and escape unescaped newlines
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
                    # Only return if we got something useful
                    if isinstance(result, dict) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON from response:\n{content[:500]}...")


async def generate_skill(
    task: str,
    examples: list[str] | None = None,
    model: str | None = None,
    config: Config | None = None,
) -> Skill:
    """Generate a skill from a task description."""
    config = config or Config.load()
    model = model or config.model
    client = AsyncAnthropic()

    prompt = f"Create a skill document that teaches an AI agent how to: {task}"
    if examples:
        prompt += "\n\nExample input/output pairs for this task:\n" + "\n".join(f"- {ex}" for ex in examples)

    # Try up to 2 times
    last_error = None
    for attempt in range(2):
        response = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=GENERATION_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        try:
            data = parse_json_response(content)
            break
        except ValueError as e:
            last_error = e
            if attempt == 0:
                # Retry with more explicit instruction
                prompt = f"{prompt}\n\nIMPORTANT: Output only raw JSON, no code blocks."
                continue
            raise last_error

    # Validate required fields
    required = ["name", "description", "body"]
    missing = [k for k in required if k not in data]
    if missing:
        # Show what we got for debugging
        preview = {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                   for k, v in data.items()}
        raise ValueError(f"Missing required fields: {missing}\nGot: {preview}\n\nRaw response:\n{content[:1000]}")

    return Skill(
        name=data["name"],
        description=data["description"],
        body=data["body"],
        references=data.get("references", {}),
        scripts=data.get("scripts", {}),
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(timezone.utc),
            source_task=task,
        ),
    )


async def generate_tests(
    task: str, model: str | None = None, config: Config | None = None
) -> list[TestCase]:
    """Generate synthetic test cases from a task description."""
    config = config or Config.load()
    model = model or config.model
    client = AsyncAnthropic()

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": TEST_GENERATION_PROMPT.format(task=task)}],
    )

    content = response.content[0].text
    data = parse_json_response(content)

    if isinstance(data, list):
        return [TestCase(**tc) for tc in data]
    return [TestCase(**tc) for tc in data.get("cases", data)]


async def refine_skill(
    skill: Skill,
    failures: list[str],
    model: str | None = None,
    config: Config | None = None,
) -> Skill:
    """Refine a skill based on evaluation failures."""
    config = config or Config.load()
    model = model or config.model
    client = AsyncAnthropic()

    prompt = f"""Improve this skill based on failures:

Name: {skill.name}
Description: {skill.description}
Body: {skill.body[:500]}...

Failures:
{chr(10).join(f'- {f}' for f in failures[:3])}

Output improved skill as JSON (same structure, no code blocks)."""

    response = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=GENERATION_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    data = parse_json_response(response.content[0].text)

    return Skill(
        name=data.get("name", skill.name),
        description=data.get("description", skill.description),
        body=data["body"],
        references=data.get("references", skill.references),
        scripts=data.get("scripts", skill.scripts),
        metadata=SkillMetadata(
            generated_by=model,
            generated_at=datetime.now(timezone.utc),
            source_task=skill.metadata.source_task,
        ),
    )

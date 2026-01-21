"""Skill generation from task descriptions using FastAgent."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from fast_agent import FastAgent

from upskill.config import Config
from upskill.models import Skill, SkillMetadata, TestCase

# Few-shot examples for skill generation
SKILL_EXAMPLES = """
## Example 1: Git Commit Skill

Task: "Write good git commit messages"

Output:
```json
{
  "name": "git-commit-messages",
  "description": "Write clear, conventional commit messages that follow best practices.",
  "body": "## Overview\\n\\nThis skill teaches how to write effective git commit messages following the Conventional Commits specification.\\n\\n## Format\\n\\nCommit messages should follow this structure:\\n\\n```\\n<type>(<scope>): <subject>\\n\\n<body>\\n\\n<footer>\\n```\\n\\n## Types\\n\\n- `feat`: New feature\\n- `fix`: Bug fix\\n- `docs`: Documentation changes\\n- `style`: Code style changes (formatting, semicolons)\\n- `refactor`: Code refactoring\\n- `test`: Adding or updating tests\\n- `chore`: Build process or auxiliary tool changes\\n\\n## Examples\\n\\n### Simple feature commit\\n```\\nfeat(auth): add password reset functionality\\n```\\n\\n### Bug fix with explanation\\n```\\nfix(api): handle null response from user service\\n\\nThe user service can return null when the user is not found.\\nAdded proper null checking to prevent crashes.\\n\\nCloses #123\\n```\\n\\n### Breaking change\\n```\\nfeat(api)!: change authentication endpoint response format\\n\\nBREAKING CHANGE: The /auth/login endpoint now returns\\na different JSON structure with nested user object.\\n```\\n\\n## Guidelines\\n\\n1. Keep the subject line under 50 characters\\n2. Use imperative mood ('add' not 'added')\\n3. Don't end the subject with a period\\n4. Separate subject from body with a blank line\\n5. Use the body to explain what and why, not how"
}
```

## Example 2: API Error Handling Skill

Task: "Handle API errors gracefully in Python"

Output:
```json
{
  "name": "python-api-error-handling",
  "description": "Implement robust error handling for REST API calls in Python applications.",
  "body": "## Overview\\n\\nThis skill covers best practices for handling errors when making HTTP API calls in Python.\\n\\n## Key Principles\\n\\n1. **Catch specific exceptions** - Don't use bare `except:`\\n2. **Implement retry logic** - For transient failures\\n3. **Log appropriately** - Include context for debugging\\n4. **Fail gracefully** - Provide meaningful error messages\\n\\n## Example Implementation\\n\\n```python\\nimport requests\\nfrom requests.exceptions import RequestException, Timeout, HTTPError\\nimport time\\n\\ndef fetch_with_retry(url: str, max_retries: int = 3) -> dict:\\n    \\\"\\\"\\\"Fetch data from API with retry logic.\\\"\\\"\\\"\\n    for attempt in range(max_retries):\\n        try:\\n            response = requests.get(url, timeout=10)\\n            response.raise_for_status()\\n            return response.json()\\n        except Timeout:\\n            if attempt < max_retries - 1:\\n                time.sleep(2 ** attempt)  # Exponential backoff\\n                continue\\n            raise\\n        except HTTPError as e:\\n            if e.response.status_code >= 500:\\n                if attempt < max_retries - 1:\\n                    time.sleep(2 ** attempt)\\n                    continue\\n            raise\\n        except RequestException as e:\\n            raise RuntimeError(f\\\"API request failed: {e}\\\")\\n```\\n\\n## Error Categories\\n\\n| Status Code | Action |\\n|------------|--------|\\n| 4xx | Don't retry, fix request |\\n| 5xx | Retry with backoff |\\n| Timeout | Retry with backoff |\\n| Connection | Retry with backoff |\\n\\n## Best Practices\\n\\n- Always set timeouts on requests\\n- Use exponential backoff for retries\\n- Include request IDs in logs\\n- Return user-friendly error messages"
}
```

## Example 3: Data Validation Skill

Task: "Validate user input in web forms"

Output:
```json
{
  "name": "form-input-validation",
  "description": "Validate and sanitize user input in web forms to prevent errors and security issues.",
  "body": "## Overview\\n\\nProper input validation is critical for security and data integrity. This skill covers client and server-side validation patterns.\\n\\n## Validation Layers\\n\\n1. **Client-side** - Immediate feedback, better UX\\n2. **Server-side** - Security boundary, never trust client\\n\\n## Common Validations\\n\\n### Email\\n```python\\nimport re\\n\\ndef validate_email(email: str) -> bool:\\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$'\\n    return bool(re.match(pattern, email))\\n```\\n\\n### Phone Number\\n```python\\ndef validate_phone(phone: str) -> bool:\\n    # Remove common separators\\n    digits = re.sub(r'[\\\\s\\\\-\\\\(\\\\)]', '', phone)\\n    return digits.isdigit() and 10 <= len(digits) <= 15\\n```\\n\\n### Password Strength\\n```python\\ndef validate_password(password: str) -> tuple[bool, list[str]]:\\n    errors = []\\n    if len(password) < 8:\\n        errors.append('Must be at least 8 characters')\\n    if not re.search(r'[A-Z]', password):\\n        errors.append('Must contain uppercase letter')\\n    if not re.search(r'[a-z]', password):\\n        errors.append('Must contain lowercase letter')\\n    if not re.search(r'\\\\d', password):\\n        errors.append('Must contain a number')\\n    return len(errors) == 0, errors\\n```\\n\\n## Sanitization\\n\\nAlways sanitize before storing or displaying:\\n\\n```python\\nimport html\\n\\ndef sanitize_input(value: str) -> str:\\n    return html.escape(value.strip())\\n```\\n\\n## Security Notes\\n\\n- Never trust client-side validation alone\\n- Use parameterized queries for database input\\n- Escape output based on context (HTML, SQL, etc.)"
}
```
"""

# Meta-prompt for skill generation with few-shot examples
GENERATION_PROMPT = f"""You generate "skills" - instruction documents that teach AI coding agents how to perform tasks.

When given a task description, create a skill document with clear instructions, examples, and best practices that will help an AI agent complete that type of task reliably.

{SKILL_EXAMPLES}

## Output Format

Output ONLY a JSON object with this exact structure:
{{"name": "skill-name", "description": "What this skill teaches", "body": "Markdown instructions"}}

## Field Requirements

- **name**: lowercase alphanumeric with hyphens (e.g., "parse-yaml-files", "git-commit-messages")
- **description**: one sentence under 100 characters describing what the skill teaches
- **body**: 200-400 word markdown guide including:
  - Brief overview
  - Step-by-step instructions or key principles
  - 2-3 practical code examples
  - Best practices or common pitfalls

## Important

- Output ONLY the JSON object, no other text
- Use \\n for newlines in string fields
- Do NOT actually perform the task - create instructions FOR performing it
- Focus on practical, actionable guidance with real code examples"""

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

TEST_GENERATION_PROMPT = f"""Generate 3-5 test cases for evaluating if an AI agent can perform this task well.

{TEST_EXAMPLES}

## Your Task

Task: {TASK_PLACEHOLDER}

Generate test cases that verify the agent can apply the skill correctly.

Output ONLY valid JSON array (no markdown code blocks):
[{{"input": "prompt/question for the agent", "expected": {{"contains": "substring that should appear in good response"}}}}]

Focus on practical scenarios that test understanding of the core concepts."""


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
    fast = FastAgent(
        "upskill-generator",
        ignore_unknown_args=True,
        parse_cli_args=False,  # Disable CLI arg parsing to avoid conflicts with upskill CLI
        config_path=str(config_path),
    )

    instruction = GENERATION_PROMPT
    if model:

        @fast.agent(name="skill_gen", instruction=instruction, model=model)
        async def skill_gen():
            return None
    else:

        @fast.agent(name="skill_gen", instruction=instruction)
        async def skill_gen():
            return None

    return fast


def build_test_generator(config_path: Path, model: str | None = None) -> FastAgent:
    """Create a FastAgent instance for test generation."""
    fast = FastAgent(
        "upskill-test-generator",
        ignore_unknown_args=True,
        parse_cli_args=False,  # Disable CLI arg parsing to avoid conflicts with upskill CLI
        config_path=str(config_path),
    )

    instruction = "You generate test cases for evaluating AI agent skills. Output only valid JSON."
    if model:

        @fast.agent(name="test_gen", instruction=instruction, model=model)
        async def test_gen():
            return None
    else:

        @fast.agent(name="test_gen", instruction=instruction)
        async def test_gen():
            return None

    return fast


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
    for attempt in range(2):
        async with fast.run() as agent:
            response = await agent.skill_gen.send(prompt)

        try:
            data = parse_json_response(response)
            break
        except ValueError as e:
            last_error = e
            if attempt == 0:
                prompt = f"{prompt}\n\nIMPORTANT: Output only raw JSON, no code blocks."
                continue
            raise last_error

    # Validate required fields
    required = ["name", "description", "body"]
    missing = [k for k in required if k not in data]
    if missing:
        preview = {
            k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
            for k, v in data.items()
        }
        raise ValueError(
            f"Missing required fields: {missing}\nGot: {preview}\n\nRaw response:\n{response[:1000]}"
        )

    return Skill(
        name=data["name"],
        description=data["description"],
        body=data["body"],
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

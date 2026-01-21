"""Skill evaluation - compare agent performance with and without skills."""

from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from upskill.config import Config
from upskill.models import EvalResults, Skill, TestCase, TestResult


def extract_text_from_content(content: list) -> str:
    """Extract text from Anthropic response content, handling thinking models."""
    texts = []
    for block in content:
        # Standard text block
        if hasattr(block, "text"):
            texts.append(block.text)
        # Thinking block (from reasoning models)
        elif hasattr(block, "thinking"):
            texts.append(block.thinking)
        # Dict format (from some API responses)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                texts.append(block.get("thinking", ""))
    return "\n".join(texts)


async def run_test_anthropic(
    test_case: TestCase,
    skill: Skill | None,
    model: str,
    base_url: str | None = None,
) -> TestResult:
    """Run test using Anthropic API (or compatible endpoint like llama.cpp)."""
    client = AsyncAnthropic(base_url=base_url) if base_url else AsyncAnthropic()

    system = "You are a helpful AI assistant."
    if skill:
        system += f"\n\n## Skill: {skill.name}\n\n{skill.body}"

    user_content = test_case.input
    if test_case.context and "files" in test_case.context:
        for filename, content in test_case.context["files"].items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )

        output = extract_text_from_content(response.content)
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        success = check_expected(output, test_case.expected)

        return TestResult(
            test_case=test_case,
            success=success,
            output=output,
            tokens_used=tokens_used,
            turns=1,
        )

    except Exception as e:
        return TestResult(test_case=test_case, success=False, error=str(e))


async def run_test_openai(
    test_case: TestCase,
    skill: Skill | None,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> TestResult:
    """Run test using OpenAI API (or compatible endpoint like Ollama, vLLM)."""
    # For local endpoints, use a dummy key if none provided
    if base_url and not api_key:
        api_key = "sk-no-key-required"

    client = AsyncOpenAI(base_url=base_url, api_key=api_key) if base_url else AsyncOpenAI()

    system = "You are a helpful AI assistant."
    if skill:
        system += f"\n\n## Skill: {skill.name}\n\n{skill.body}"

    user_content = test_case.input
    if test_case.context and "files" in test_case.context:
        for filename, content in test_case.context["files"].items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    try:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )

        output = response.choices[0].message.content or ""
        tokens_used = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0

        success = check_expected(output, test_case.expected)

        return TestResult(
            test_case=test_case,
            success=success,
            output=output,
            tokens_used=tokens_used,
            turns=1,
        )

    except Exception as e:
        return TestResult(test_case=test_case, success=False, error=str(e))


def check_expected(output: str, expected: dict | None) -> bool:
    """Check if output matches expected conditions."""
    if not expected:
        return True

    if "contains" in expected:
        if expected["contains"].lower() not in output.lower():
            return False

    return True


async def run_test(
    test_case: TestCase,
    skill: Skill | None,
    model: str,
    provider: str = "anthropic",
    base_url: str | None = None,
    api_key: str | None = None,
) -> TestResult:
    """Run a single test case using the specified provider.

    Args:
        test_case: The test to run
        skill: Skill to use (None for baseline)
        model: Model name to use
        provider: "anthropic" or "openai"
        base_url: Custom API endpoint (for local models)
        api_key: API key (optional for local endpoints)

    Returns:
        TestResult with success/failure and metrics
    """
    if provider == "openai":
        return await run_test_openai(test_case, skill, model, base_url, api_key)
    else:
        return await run_test_anthropic(test_case, skill, model, base_url)


async def evaluate_skill(
    skill: Skill,
    test_cases: list[TestCase],
    model: str | None = None,
    config: Config | None = None,
    run_baseline: bool = True,
    provider: str = "anthropic",
    base_url: str | None = None,
    api_key: str | None = None,
) -> EvalResults:
    """Evaluate a skill against test cases.

    Args:
        skill: The skill to evaluate
        test_cases: Test cases to run
        model: Model to evaluate on (defaults to config.eval_model)
        config: Configuration
        run_baseline: Whether to also run without the skill
        provider: "anthropic" or "openai"
        base_url: Custom API endpoint (for local models)
        api_key: API key (optional for local endpoints)

    Returns:
        EvalResults comparing skill vs baseline
    """
    config = config or Config.load()
    model = model or config.effective_eval_model

    results = EvalResults(skill_name=skill.name, model=model)

    # Run with skill
    for tc in test_cases:
        result = await run_test(tc, skill, model, provider, base_url, api_key)
        results.with_skill_results.append(result)

    # Calculate with-skill metrics
    successes = sum(1 for r in results.with_skill_results if r.success)
    results.with_skill_success_rate = successes / len(test_cases) if test_cases else 0
    results.with_skill_total_tokens = sum(r.tokens_used for r in results.with_skill_results)
    results.with_skill_avg_turns = (
        sum(r.turns for r in results.with_skill_results) / len(test_cases) if test_cases else 0
    )

    # Run baseline if requested
    if run_baseline:
        for tc in test_cases:
            result = await run_test(tc, None, model, provider, base_url, api_key)
            results.baseline_results.append(result)

        successes = sum(1 for r in results.baseline_results if r.success)
        results.baseline_success_rate = successes / len(test_cases) if test_cases else 0
        results.baseline_total_tokens = sum(r.tokens_used for r in results.baseline_results)
        results.baseline_avg_turns = (
            sum(r.turns for r in results.baseline_results) / len(test_cases) if test_cases else 0
        )

    return results


def get_failure_descriptions(results: EvalResults) -> list[str]:
    """Extract descriptions of failed tests for refinement."""
    failures = []
    for result in results.with_skill_results:
        if not result.success:
            desc = f"Input: {result.test_case.input}"
            if result.error:
                desc += f" | Error: {result.error}"
            elif result.output:
                desc += f" | Output: {result.output[:200]}..."
            if result.test_case.expected:
                desc += f" | Expected: {result.test_case.expected}"
            failures.append(desc)
    return failures

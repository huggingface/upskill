"""Skill evaluation - compare agent performance with and without skills using FastAgent."""

from __future__ import annotations

import os
from pathlib import Path

from fast_agent import ConversationSummary, FastAgent

from upskill.config import Config
from upskill.logging import extract_stats_from_summary
from upskill.models import ConversationStats, EvalResults, Skill, TestCase, TestResult

PROMPT = """You are an evaluator of skills. You are given a skill and a test case. You need to evaluate the skill on the test case and return a score."""

def check_expected(output: str, expected: dict | None) -> bool:
    """Check if output matches expected conditions."""
    if not expected:
        return True

    if "contains" in expected:
        if expected["contains"].lower() not in output.lower():
            return False

    return True


def build_eval_agent(
    config_path: Path,
    skill: Skill | None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    instruction: str | None = PROMPT,
) -> FastAgent:
    """Create a FastAgent instance for running evaluation tests.

    Args:
        config_path: Path to fastagent.config.yaml
        skill: Skill to inject into system prompt (None for baseline)
        model: Model name (FastAgent format or plain name)
        provider: Provider name (anthropic, openai, generic) - prepended to model if not already present
        base_url: Custom API endpoint (set via environment variable)
    """
    # Determine effective provider - default to generic if base_url provided without provider
    effective_provider = provider
    if base_url and not provider:
        effective_provider = "generic"

    # Set base URL via environment if provided
    if base_url:
        if effective_provider == "openai":
            os.environ["OPENAI_API_BASE"] = base_url
        elif effective_provider == "generic":
            os.environ["GENERIC_BASE_URL"] = base_url
        else:
            os.environ["ANTHROPIC_BASE_URL"] = base_url

    fast = FastAgent(
        "upskill-evaluator",
        ignore_unknown_args=True,
        config_path=str(config_path),
    )

    if skill:
        instruction += f"\n\n## Skill: {skill.name}\n\n{skill.body}"

    # Build model string with provider prefix if needed
    # Check if model already has a known provider prefix
    known_providers = ("anthropic.", "openai.", "generic.", "google.", "tensorzero.")
    model_str = model
    if model and effective_provider and not model.startswith(known_providers):
        model_str = f"{effective_provider}.{model}"

    if model_str:

        @fast.agent(name="eval_agent", instruction=instruction, model=model_str)
        async def eval_agent():
            return None
    else:

        @fast.agent(name="eval_agent", instruction=instruction)
        async def eval_agent():
            return None

    return fast


async def run_test(
    test_case: TestCase,
    skill: Skill | None,
    model: str,
    config_path: Path,
    provider: str | None = None,
    base_url: str | None = None,
) -> TestResult:
    """Run a single test case using FastAgent."""
    user_content = test_case.input
    if test_case.context and "files" in test_case.context:
        for filename, content in test_case.context["files"].items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    try:
        fast = build_eval_agent(config_path, skill, model, provider, base_url)

        async with fast.run() as agent:
            output = await agent.eval_agent.send(user_content)

            # Extract stats from agent history using ConversationSummary
            try:
                history = agent.eval_agent.history()
                summary = ConversationSummary(history)
                stats = extract_stats_from_summary(summary)
            except Exception:
                # Fall back to empty stats if extraction fails
                stats = ConversationStats()

        success = check_expected(output, test_case.expected)

        return TestResult(
            test_case=test_case,
            success=success,
            output=output,
            tokens_used=stats.total_tokens,  # Legacy field
            turns=stats.turns,  # Legacy field
            stats=stats,
        )

    except Exception as e:
        return TestResult(test_case=test_case, success=False, error=str(e))


async def evaluate_skill(
    skill: Skill,
    test_cases: list[TestCase],
    model: str | None = None,
    config: Config | None = None,
    run_baseline: bool = True,
    provider: str | None = None,
    base_url: str | None = None,
) -> EvalResults:
    """Evaluate a skill against test cases using FastAgent.

    Args:
        skill: The skill to evaluate
        test_cases: Test cases to run
        model: Model to evaluate on (defaults to config.eval_model)
        config: Configuration
        run_baseline: Whether to also run without the skill
        provider: API provider (anthropic, openai, generic) - overrides config
        base_url: Custom API endpoint - overrides config

    Returns:
        EvalResults comparing skill vs baseline
    """
    config = config or Config.load()
    model = model or config.effective_eval_model
    config_path = config.effective_fastagent_config

    results = EvalResults(skill_name=skill.name, model=model)

    # Run with skill
    for tc in test_cases:
        result = await run_test(tc, skill, model, config_path, provider, base_url)
        results.with_skill_results.append(result)

    # Calculate with-skill metrics
    successes = sum(1 for r in results.with_skill_results if r.success)
    results.with_skill_success_rate = successes / len(test_cases) if test_cases else 0
    results.with_skill_total_tokens = sum(r.stats.total_tokens for r in results.with_skill_results)
    results.with_skill_avg_turns = (
        sum(r.stats.turns for r in results.with_skill_results) / len(test_cases)
        if test_cases
        else 0
    )

    # Run baseline if requested
    if run_baseline:
        for tc in test_cases:
            result = await run_test(tc, None, model, config_path, provider, base_url)
            results.baseline_results.append(result)

        successes = sum(1 for r in results.baseline_results if r.success)
        results.baseline_success_rate = successes / len(test_cases) if test_cases else 0
        results.baseline_total_tokens = sum(r.stats.total_tokens for r in results.baseline_results)
        results.baseline_avg_turns = (
            sum(r.stats.turns for r in results.baseline_results) / len(test_cases)
            if test_cases
            else 0
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

"""Skill evaluation - compare agent performance with and without skills using FastAgent."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fast_agent import ConversationSummary, FastAgent

from upskill.config import Config
from upskill.fastagent_integration import (
    build_agent_from_card,
    compose_instruction,
)
from upskill.logging import extract_stats_from_summary
from upskill.models import (
    ConversationStats,
    EvalResults,
    Skill,
    TestCase,
    TestResult,
    ValidationResult,
)
from upskill.validators import get_validator

logger = logging.getLogger(__name__)

PROMPT = (
    "You are an evaluator of skills. You are given a skill and a test case. "
    "You need to evaluate the skill on the test case and return a score."
)


@contextmanager
def isolated_workspace(base_dir: Path | None = None, cleanup: bool = True) -> Generator[Path]:
    """Create an isolated workspace for a test run.

    Args:
        base_dir: Optional parent directory for the workspace
        cleanup: Whether to clean up the workspace after (default True)

    Yields:
        Path to the temporary workspace directory
    """
    workspace = tempfile.mkdtemp(dir=base_dir, prefix="upskill_run_")
    workspace_path = Path(workspace)
    original_cwd = Path.cwd()
    try:
        os.chdir(workspace_path)
        yield workspace_path
    finally:
        os.chdir(original_cwd)
        if cleanup:
            try:
                shutil.rmtree(workspace_path, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors


def check_expected(
    output: str,
    expected: dict | None,
    workspace: Path | None = None,
    test_case: TestCase | None = None,
) -> tuple[bool, ValidationResult | None]:
    """Check if output matches expected conditions.

    Args:
        output: The agent's output string
        expected: Expected conditions dict (legacy format with "contains")
        workspace: Optional workspace directory for file-based validation
        test_case: Optional test case with custom validator config

    Returns:
        Tuple of (success, validation_result)
    """
    # Handle custom validator if specified
    if test_case and test_case.validator:
        validator = get_validator(test_case.validator)
        if validator and workspace:
            config = test_case.validator_config or {}
            result = validator(
                workspace=workspace,
                output_file=test_case.output_file or "",
                **config,
            )
            return result.passed, result

    # Legacy: simple contains check
    if not expected:
        return True, None

    if "contains" in expected:
        if expected["contains"].lower() not in output.lower():
            return False, None

    return True, None


def build_eval_agent(
    config_path: Path,
    skill: Skill | None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    instruction: str | None = None,
) -> FastAgent:
    """Create a FastAgent instance for running evaluation tests.

    Args:
        config_path: Path to fastagent.config.yaml
        skill: Skill to inject into system prompt (None for baseline)
        model: Model name (FastAgent format or plain name)
        provider: Provider name (anthropic, openai, generic)
        base_url: Custom API endpoint (set via environment variable)
    """


    fast = build_agent_from_card(
        "upskill-evaluator",
        config_path,
        agent_name="evaluator",
        model=model,
    )

    agent_data = fast.agents.get("evaluator")
    if not agent_data:
        raise ValueError("AgentCard 'evaluator' not found in upskill package")

    base_instruction = instruction or agent_data.get("instruction") or PROMPT
    agent_data["instruction"] = compose_instruction(base_instruction, skill)

    return fast


async def run_test(
    test_case: TestCase,
    skill: Skill | None,
    model: str,
    config_path: Path,
    provider: str | None = None,
    base_url: str | None = None,
    use_workspace: bool | None = None,
) -> TestResult:
    """Run a single test case using FastAgent.

    Args:
        test_case: The test case to run
        skill: Optional skill to inject (None for baseline)
        model: Model name
        config_path: Path to fastagent config
        provider: API provider override
        base_url: Custom API endpoint override
        use_workspace: Force workspace isolation (auto-detected from test_case.validator)
    """
    user_content = test_case.input
    if test_case.context and "files" in test_case.context:
        for filename, content in test_case.context["files"].items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    # Determine if we need workspace isolation
    needs_workspace = use_workspace if use_workspace is not None else bool(test_case.validator)

    async def _run_in_workspace(workspace: Path | None) -> TestResult:
        try:
            fast = build_eval_agent(config_path, skill, model, provider, base_url)

            output = None
            stats = ConversationStats()

            async with fast.run() as agent:
                output = await agent.evaluator.send(user_content)

                # Extract stats from agent history
                try:
                    history = agent.evaluator.message_history
                    summary = ConversationSummary(messages=history)
                    stats = extract_stats_from_summary(summary)
                except Exception as exc:
                    logger.exception("Failed to extract stats from evaluator history", exc_info=exc)

            # Check expected with custom validator support
            if workspace and test_case.validator:
                success, validation_result = check_expected(
                    output or "", test_case.expected, workspace, test_case
                )
            else:
                success, validation_result = check_expected(
                    output or "", test_case.expected
                )

            return TestResult(
                test_case=test_case,
                success=success,
                output=output,
                tokens_used=stats.total_tokens,
                turns=stats.turns,
                stats=stats,
                validation_result=validation_result,
            )

        except Exception as e:
            return TestResult(test_case=test_case, success=False, error=str(e))

    if needs_workspace:
        with isolated_workspace() as workspace:
            return await _run_in_workspace(workspace)
    else:
        return await _run_in_workspace(None)


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

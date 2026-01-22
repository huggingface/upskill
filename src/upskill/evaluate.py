"""Skill evaluation - compare agent performance with and without skills using FastAgent."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fast_agent import ConversationSummary
from fast_agent.agents.llm_agent import LlmAgent

from upskill.fastagent_integration import (
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


async def _run_test_with_evaluator(
    test_case: TestCase,
    evaluator: LlmAgent,
    instruction: str | None,
    *,
    use_workspace: bool | None = None,
    instance_name: str | None = None,
) -> TestResult:
    """Run a single test case using a provided evaluator agent."""
    user_content = test_case.input
    if test_case.context and "files" in test_case.context:
        for filename, content in test_case.context["files"].items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    # Determine if we need workspace isolation
    needs_workspace = use_workspace if use_workspace is not None else bool(test_case.validator)

    async def _run_in_workspace(workspace: Path | None) -> TestResult:
        clone: LlmAgent | None = None
        try:
            clone = await evaluator.spawn_detached_instance(name=instance_name)
            if instruction:
                clone.set_instruction(instruction)

            output = await clone.send(user_content)
            stats = ConversationStats()

            # Extract stats from agent history
            try:
                history = clone.message_history
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
        except Exception as exc:
            return TestResult(test_case=test_case, success=False, error=str(exc))
        finally:
            if clone is not None:
                try:
                    await clone.shutdown()
                except Exception as exc:
                    logger.exception("Failed to shutdown evaluator clone", exc_info=exc)

    if needs_workspace:
        with isolated_workspace() as workspace:
            return await _run_in_workspace(workspace)
    return await _run_in_workspace(None)


async def run_test(
    test_case: TestCase,
    evaluator: LlmAgent,
    skill: Skill | None,
    use_workspace: bool | None = None,
) -> TestResult:
    """Run a single test case using an evaluator agent.

    Args:
        test_case: The test case to run
        evaluator: Evaluator agent to run the test case
        skill: Optional skill to inject (None for baseline)
        use_workspace: Force workspace isolation (auto-detected from test_case.validator)
    """

    try:
        instruction = compose_instruction(evaluator.instruction, skill) if skill else None
        return await _run_test_with_evaluator(
            test_case,
            evaluator,
            instruction,
            use_workspace=use_workspace,
        )
    except Exception as exc:
        return TestResult(test_case=test_case, success=False, error=str(exc))


async def evaluate_skill(
    skill: Skill,
    test_cases: list[TestCase],
    evaluator: LlmAgent,
    model: str | None = None,
    run_baseline: bool = True,
) -> EvalResults:
    """Evaluate a skill against test cases using FastAgent.

    Args:
        skill: The skill to evaluate
        test_cases: Test cases to run
        evaluator: Evaluator agent to run the test cases
        model: Model to evaluate on (defaults to config.eval_model)
        run_baseline: Whether to also run without the skill

    Returns:
        EvalResults comparing skill vs baseline
    """
    # config = config or Config.load()
    # model = model or config.effective_eval_model
    # config_path = config.effective_fastagent_config

    results = EvalResults(skill_name=skill.name, model=model)


    base_instruction = evaluator.instruction

    async def _run_batch(
        instruction: str | None,
        label: str,
    ) -> list[TestResult]:
        tasks = []
        for index, tc in enumerate(test_cases, start=1):
            instance_name = f"{evaluator.name}[{label}-{index}]"
            tasks.append(
                _run_test_with_evaluator(
                    tc,
                    evaluator,
                    instruction,
                    instance_name=instance_name,
                )
            )
        return await asyncio.gather(*tasks)

    # Run with skill
    skill_instruction = compose_instruction(base_instruction, skill)
    results.with_skill_results = await _run_batch(skill_instruction, "skill")

    # Calculate with-skill metrics
    successes = sum(1 for r in results.with_skill_results if r.success)
    results.with_skill_success_rate = successes / len(test_cases) if test_cases else 0
    results.with_skill_total_tokens = sum(
        r.stats.total_tokens for r in results.with_skill_results
    )
    results.with_skill_avg_turns = (
        sum(r.stats.turns for r in results.with_skill_results) / len(test_cases)
        if test_cases
        else 0
    )

    # Run baseline if requested
    if run_baseline:
        results.baseline_results = await _run_batch(None, "baseline")

        successes = sum(1 for r in results.baseline_results if r.success)
        results.baseline_success_rate = successes / len(test_cases) if test_cases else 0
        results.baseline_total_tokens = sum(
            r.stats.total_tokens for r in results.baseline_results
        )
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

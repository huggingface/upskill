"""Skill evaluation orchestration backed by an execution backend."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from upskill.executors.base import Executor

from upskill.artifacts import ensure_directory, sanitize_artifact_name
from upskill.executors.contracts import ExecutionRequest
from upskill.models import (
    EvalResults,
    Skill,
    TestCase,
    TestResult,
    ValidationResult,
)
from upskill.verifiers import run_verifiers

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PendingEvaluationRequest:
    """A single evaluation request prepared for backend submission."""

    phase_label: str
    test_index: int
    request: ExecutionRequest


def _format_execution_error(
    error: str,
    *,
    metadata: dict[str, str | int | float | bool | None] | None,
) -> str:
    """Append useful execution identifiers to surfaced backend errors."""
    if metadata is None:
        return error

    job_id = metadata.get("job_id")
    if isinstance(job_id, str) and job_id:
        return f"{error} (job {job_id})"
    return error


def _write_test_result_summary(path: Path, result: TestResult) -> None:
    """Persist a per-test result summary alongside raw artifacts."""
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _load_test_result_summary(path: Path) -> TestResult | None:
    """Load a persisted per-test result summary."""
    if not path.exists():
        return None
    try:
        return TestResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def apply_eval_metrics(results: EvalResults, test_cases: list[TestCase]) -> EvalResults:
    """Populate aggregate metrics on an ``EvalResults`` instance."""
    successes = sum(1 for r in results.with_skill_results if r.success)
    results.with_skill_success_rate = successes / len(test_cases) if test_cases else 0
    results.with_skill_total_tokens = sum(r.stats.total_tokens for r in results.with_skill_results)
    results.with_skill_avg_turns = (
        sum(r.stats.turns for r in results.with_skill_results) / len(test_cases)
        if test_cases
        else 0
    )

    if results.baseline_results:
        successes = sum(1 for r in results.baseline_results if r.success)
        results.baseline_success_rate = successes / len(test_cases) if test_cases else 0
        results.baseline_total_tokens = sum(r.stats.total_tokens for r in results.baseline_results)
        results.baseline_avg_turns = (
            sum(r.stats.turns for r in results.baseline_results) / len(test_cases)
            if test_cases
            else 0
        )

    return results


def load_eval_results_from_artifact_root(
    *,
    skill_name: str,
    model: str,
    artifact_root: Path,
) -> EvalResults | None:
    """Reconstruct eval results from persisted per-test summaries."""
    if not artifact_root.exists():
        return None

    with_skill_results = [
        loaded
        for loaded in (
            _load_test_result_summary(summary_path)
            for summary_path in sorted(
                (artifact_root / "with-skill").glob("test_*/test_result.json")
            )
        )
        if loaded is not None
    ]
    baseline_results = [
        loaded
        for loaded in (
            _load_test_result_summary(summary_path)
            for summary_path in sorted((artifact_root / "baseline").glob("test_*/test_result.json"))
        )
        if loaded is not None
    ]

    if not with_skill_results and not baseline_results:
        return None

    reconstructed = EvalResults(
        skill_name=skill_name,
        model=model,
        with_skill_results=with_skill_results,
        baseline_results=baseline_results,
    )
    test_cases = [result.test_case for result in with_skill_results]
    if not test_cases:
        test_cases = [result.test_case for result in baseline_results]
    return apply_eval_metrics(reconstructed, test_cases)


def check_expected(
    output: str,
    test_case: TestCase,
    workspace: Path | None = None,
) -> ValidationResult:
    """Run normalized deterministic verifiers for one test case."""
    return run_verifiers(test_case, output=output, workspace=workspace)


def format_test_prompt(test_case: TestCase) -> str:
    """Build the evaluator prompt, preserving legacy inline file context."""
    prompt = test_case.input
    if test_case.context and test_case.context.files:
        for filename, content in test_case.context.files.items():
            prompt += f"\n\n```{filename}\n{content}\n```"
    return prompt


def build_eval_execution_request(
    test_case: TestCase,
    *,
    skill: Skill | None,
    model: str,
    fastagent_config_path: Path,
    cards_source_dir: Path,
    artifact_dir: Path,
    agent_name: str = "evaluator",
    instance_name: str | None = None,
    operation: str = "eval",
) -> ExecutionRequest:
    """Build the normalized execution request for a single evaluation test."""
    workspace_files = (
        dict(test_case.context.files) if test_case.context and test_case.context.files else {}
    )
    normalized_artifact_dir = artifact_dir.resolve()
    return ExecutionRequest(
        prompt=format_test_prompt(test_case),
        model=model,
        agent=agent_name,
        fastagent_config_path=fastagent_config_path.resolve(),
        artifact_dir=normalized_artifact_dir,
        cards_source_dir=cards_source_dir.resolve(),
        label=instance_name or (skill.name if skill else "baseline"),
        skill=skill,
        workspace_files=workspace_files,
        metadata={
            "instance_name": instance_name,
            "operation": operation,
            "skill_name": skill.name if skill else None,
            "has_validator": bool(test_case.effective_verifiers()),
        },
    )


def build_eval_requests(
    *,
    skill: Skill,
    test_cases: list[TestCase],
    model: str,
    fastagent_config_path: Path,
    cards_source_dir: Path,
    artifact_root: Path,
    run_baseline: bool = True,
    operation: str = "eval",
) -> list[PendingEvaluationRequest]:
    """Build all execution requests needed for an evaluation run."""
    requests: list[PendingEvaluationRequest] = []

    for phase_label, batch_skill in _iter_evaluation_phases(skill, run_baseline):
        batch_root = ensure_directory(artifact_root / sanitize_artifact_name(phase_label))
        for index, test_case in enumerate(test_cases, start=1):
            instance_name = f"eval ({phase_label} test {index})"
            requests.append(
                PendingEvaluationRequest(
                    phase_label=phase_label,
                    test_index=index,
                    request=build_eval_execution_request(
                        test_case,
                        skill=batch_skill,
                        model=model,
                        fastagent_config_path=fastagent_config_path,
                        cards_source_dir=cards_source_dir,
                        artifact_dir=batch_root / f"test_{index}",
                        instance_name=instance_name,
                        operation=operation,
                    ),
                )
            )

    return requests


def _iter_evaluation_phases(
    skill: Skill,
    run_baseline: bool,
) -> list[tuple[str, Skill | None]]:
    phases: list[tuple[str, Skill | None]] = [("with-skill", skill)]
    if run_baseline:
        phases.append(("baseline", None))
    return phases


async def _run_test_with_evaluator(
    test_case: TestCase,
    executor: Executor,
    *,
    skill: Skill | None,
    model: str,
    fastagent_config_path: Path,
    cards_source_dir: Path,
    artifact_dir: Path,
    agent_name: str = "evaluator",
    instance_name: str | None = None,
    operation: str = "eval",
) -> TestResult:
    """Run a single test case through the configured executor."""
    request = build_eval_execution_request(
        test_case,
        skill=skill,
        model=model,
        fastagent_config_path=fastagent_config_path,
        cards_source_dir=cards_source_dir,
        artifact_dir=artifact_dir,
        agent_name=agent_name,
        instance_name=instance_name,
        operation=operation,
    )
    normalized_artifact_dir = request.artifact_dir

    try:
        handle = await executor.execute(request)
        execution_result = await executor.collect(handle)
    except Exception as exc:
        logger.exception("Evaluation execution failed", exc_info=exc)
        result = TestResult(test_case=test_case, success=False, error=str(exc))
        _write_test_result_summary(normalized_artifact_dir / "test_result.json", result)
        return result

    if execution_result.error is not None:
        result = TestResult(
            test_case=test_case,
            success=False,
            output=execution_result.output_text,
            tokens_used=execution_result.stats.total_tokens,
            turns=execution_result.stats.turns,
            error=_format_execution_error(
                execution_result.error,
                metadata=execution_result.metadata,
            ),
            stats=execution_result.stats,
        )
        _write_test_result_summary(normalized_artifact_dir / "test_result.json", result)
        return result

    validation_result = None
    verification_error = None
    try:
        validation_result = check_expected(
            execution_result.output_text or "",
            test_case,
            execution_result.workspace_dir,
        )
    except Exception as exc:
        logger.exception("Evaluation verification failed", exc_info=exc)
        verification_error = f"Verification failed: {type(exc).__name__}: {exc}"

    result = TestResult(
        test_case=test_case,
        success=validation_result.passed if validation_result is not None else False,
        output=execution_result.output_text,
        tokens_used=execution_result.stats.total_tokens,
        turns=execution_result.stats.turns,
        stats=execution_result.stats,
        error=verification_error,
        validation_result=validation_result,
    )
    _write_test_result_summary(normalized_artifact_dir / "test_result.json", result)
    return result


async def run_test(
    test_case: TestCase,
    executor: Executor,
    skill: Skill | None,
    *,
    model: str,
    fastagent_config_path: Path,
    cards_source_dir: Path,
    artifact_dir: Path,
    instance_name: str | None = None,
    operation: str = "eval",
) -> TestResult:
    """Run a single test case via the execution backend.

    Args:
        test_case: The test case to run
        executor: Execution backend to use
        skill: Optional skill to inject (None for baseline)
        model: Model to evaluate with for this test case
        fastagent_config_path: Fast-agent config to pass through to execution
        cards_source_dir: Source directory for bundled agent cards
        artifact_dir: Output directory for raw execution artifacts
        instance_name: Optional evaluator instance display name
        operation: High-level command family for labeling submitted jobs
    """
    return await _run_test_with_evaluator(
        test_case,
        executor,
        skill=skill,
        model=model,
        fastagent_config_path=fastagent_config_path,
        cards_source_dir=cards_source_dir,
        artifact_dir=artifact_dir,
        instance_name=instance_name,
        operation=operation,
    )


async def evaluate_skill(
    skill: Skill,
    test_cases: list[TestCase],
    executor: Executor,
    *,
    model: str,
    fastagent_config_path: Path,
    cards_source_dir: Path,
    artifact_root: Path,
    run_baseline: bool = True,
    max_parallel: int = 5,
    progress_callback: Callable[[str], None] | None = None,
    operation: str = "eval",
) -> EvalResults:
    """Evaluate a skill against test cases using FastAgent.

    Args:
        skill: The skill to evaluate
        test_cases: Test cases to run
        executor: Execution backend to use
        model: Model to evaluate on
        fastagent_config_path: Fast-agent config path to propagate
        cards_source_dir: Source directory for evaluator cards
        artifact_root: Artifact root for preserved raw execution outputs
        run_baseline: Whether to also run without the skill
        max_parallel: Maximum number of concurrent test executions
        progress_callback: Optional callback for lightweight progress updates
        operation: High-level command family for labeling submitted jobs

    Returns:
        EvalResults comparing skill vs baseline
    """
    results = EvalResults(skill_name=skill.name, model=model)
    semaphore = asyncio.Semaphore(max_parallel)
    ensure_directory(artifact_root)

    async def _run_batch(
        batch_skill: Skill | None,
        label: str,
    ) -> list[TestResult]:
        batch_root = ensure_directory(artifact_root / sanitize_artifact_name(label))

        async def _run_single(index: int, test_case: TestCase) -> TestResult:
            instance_name = f"eval ({label} test {index})"
            test_artifact_dir = batch_root / f"test_{index}"
            if progress_callback is not None:
                progress_callback(f"starting {label} test {index}/{len(test_cases)}")
            async with semaphore:
                result = await run_test(
                    test_case,
                    executor,
                    batch_skill,
                    model=model,
                    fastagent_config_path=fastagent_config_path,
                    cards_source_dir=cards_source_dir,
                    artifact_dir=test_artifact_dir,
                    instance_name=instance_name,
                    operation=operation,
                )
            if progress_callback is not None:
                status = "ok" if result.success else "failed"
                progress_callback(f"finished {label} test {index}/{len(test_cases)} ({status})")
            return result

        tasks = [
            asyncio.create_task(_run_single(index, test_case))
            for index, test_case in enumerate(test_cases, start=1)
        ]
        return await asyncio.gather(*tasks)

    # Run with skill
    results.with_skill_results = await _run_batch(skill, "with-skill")

    # Run baseline if requested
    if run_baseline:
        results.baseline_results = await _run_batch(None, "baseline")
    return apply_eval_metrics(results, test_cases)


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

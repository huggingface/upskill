"""Test that ``upskill eval`` and ``upskill benchmark`` route per model via the router."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from upskill.cli import _eval_async
from upskill.config import Config
from upskill.executors.cli_agent import CliAgentExecutor
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.models import (
    EvalResults,
    ExpectedSpec,
    Skill,
    SkillRecord,
    SkillState,
    TestCase,
    TestResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill_fixture(path: Path) -> SkillRecord:
    record = SkillRecord(
        skill=Skill(
            name="example-skill",
            description="An example skill.",
            body="Always be concise.",
        ),
        state=SkillState(
            tests=[
                TestCase(input="prompt", expected=ExpectedSpec(contains=["concise"])),
            ],
        ),
    )
    record.save(path)
    return record


def _make_eval_results(*, model: str, skill: Skill, test_cases: list[TestCase]) -> EvalResults:
    with_skill = [
        TestResult(
            test_case=tc,
            success=True,
            output="concise answer",
            tokens_used=10,
            turns=1,
            validation_result=ValidationResult(
                passed=True,
                assertions_passed=1,
                assertions_total=1,
            ),
        )
        for tc in test_cases
    ]
    return EvalResults(
        skill_name=skill.name,
        model=model,
        with_skill_results=with_skill,
        baseline_results=[],
        with_skill_success_rate=1.0,
        baseline_success_rate=0.0,
    )


@pytest.mark.asyncio
async def test_mixed_model_benchmark_routes_per_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed CLI + fast-agent benchmark must hit the right executor per model."""
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    skill_record = _write_skill_fixture(tmp_path / "skill")
    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    executor_models: list[tuple[str, str]] = []

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args
        executor = kwargs["executor"]
        model = str(kwargs["model"])
        executor_models.append((model, type(executor).__name__))
        return _make_eval_results(
            skill=skill_record.skill,
            model=model,
            test_cases=skill_record.state.tests,
        )

    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context",
        lambda *_args, **_kwargs: _NoopAgentContext(),
    )

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["sonnet", "cli.claude-code", "cli.kiro"],
        test_gen_model=None,
        num_runs=1,
        no_baseline=False,
        verbose=False,
        executor_name="local",
        artifact_repo=None,
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=1,
        log_runs=False,
        runs_dir=str(config.runs_dir),
    )

    # Each model should have been routed to the right executor type.
    assert ("sonnet", LocalFastAgentExecutor.__name__) in executor_models
    assert ("cli.claude-code", CliAgentExecutor.__name__) in executor_models
    assert ("cli.kiro", CliAgentExecutor.__name__) in executor_models


@pytest.mark.asyncio
async def test_eval_with_cli_test_gen_model_skips_fast_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When test_gen is cli.*, _agent_session must skip _fast_agent_context.

    Concretely, this proves you can run ``upskill eval -m cli.claude-code
    --test-gen-model cli.claude-code`` with no Anthropic/OpenAI key set.
    """
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    skill_record = _write_skill_fixture(tmp_path / "skill")
    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    fast_agent_calls: list[bool] = []

    def boom(*_args: object, **_kwargs: object) -> object:
        fast_agent_calls.append(True)
        raise AssertionError("_fast_agent_context must not run when all models are cli.*")

    monkeypatch.setattr("upskill.cli._fast_agent_context", boom)

    # Drop persisted tests so test generation actually triggers.
    monkeypatch.setattr(
        "upskill.models.SkillRecord.load",
        lambda path: skill_record.model_copy(deep=True),
    )

    captured: list[str] = []

    async def fake_evaluate_skill(*_args: object, **kwargs: object) -> EvalResults:
        captured.append(str(kwargs["model"]))
        return _make_eval_results(
            skill=skill_record.skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
        )

    async def fake_generate_tests(_task: str, *, generator: object) -> list[TestCase]:
        del generator
        return list(skill_record.state.tests)

    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)
    monkeypatch.setattr("upskill.cli.generate_tests", fake_generate_tests)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["cli.claude-code"],
        test_gen_model="cli.claude-code",
        num_runs=1,
        no_baseline=True,
        verbose=False,
        executor_name="local",
        artifact_repo=None,
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=1,
        log_runs=False,
        runs_dir=str(config.runs_dir),
    )

    assert fast_agent_calls == []  # never reached
    assert captured == ["cli.claude-code"]


class _NoopAgentContext:
    """Minimal stand-in for ``_fast_agent_context`` when only eval is exercised."""

    async def __aenter__(self) -> object:
        return _NoopSession()

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _NoopSession:
    skill_gen = object()
    test_gen = object()

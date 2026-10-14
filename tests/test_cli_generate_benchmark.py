from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from upskill.cli import (
    _benchmark_async,
    _build_logged_run_result,
    _generate_async,
    _install_fast_agent_model_references,
    _submit_generate_jobs_eval,
    main,
)
from upskill.config import Config
from upskill.evaluate import apply_eval_metrics
from upskill.hf_jobs import JobsConfig
from upskill.logging import load_batch_summary
from upskill.models import (
    ConversationStats,
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
    from collections.abc import Callable
    from pathlib import Path


class _FakeAgentContext:
    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(skill_gen=object(), test_gen=object())

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False


def _make_eval_results(
    *,
    skill: Skill,
    model: str,
    test_cases: list[TestCase],
    run_baseline: bool,
) -> EvalResults:
    with_skill_results = [
        TestResult(
            test_case=test_case,
            success=True,
            stats=ConversationStats(total_tokens=10, turns=1),
        )
        for test_case in test_cases
    ]
    results = EvalResults(
        skill_name=skill.name,
        model=model,
        with_skill_results=with_skill_results,
    )
    if run_baseline:
        results.baseline_results = [
            TestResult(
                test_case=test_case,
                success=False,
                stats=ConversationStats(total_tokens=20, turns=1),
            )
            for test_case in test_cases
        ]
    return apply_eval_metrics(results, test_cases)


def test_build_logged_run_result_preserves_validator_assertion_counts() -> None:
    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))
    run_result = _build_logged_run_result(
        model="haiku",
        task="Write good pull request descriptions.",
        batch_id="batch-1",
        run_number=1,
        test_results=[
            TestResult(
                test_case=test_case,
                success=True,
                validation_result=ValidationResult(
                    passed=True,
                    assertions_passed=2,
                    assertions_total=3,
                ),
                stats=ConversationStats(total_tokens=10, turns=1),
            ),
            TestResult(
                test_case=test_case,
                success=True,
                stats=ConversationStats(total_tokens=12, turns=1),
            ),
        ],
        assertions_total=2,
        passed=False,
        run_type="with_skill",
        skill_name="pull-request-descriptions",
    )

    assert run_result.assertions_passed == 3
    assert run_result.assertions_total == 4


def test_generate_help_does_not_expose_removed_tool_option() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["generate", "--help"])

    assert result.exit_code == 0
    assert "--tool" not in result.output
    assert "--from PATH" in result.output
    assert "--artifact-repo TEXT" in result.output
    assert 'upskill generate "parse invoices"' in result.output
    assert "--artifact-repo" in result.output
    assert "<user>/upskill-tests" in result.output


def test_install_fast_agent_model_references_merges_existing_namespaces() -> None:
    fast = SimpleNamespace(
        app=SimpleNamespace(
            _config_or_path=SimpleNamespace(
                model_references={
                    "custom": {"router": "haiku"},
                    "system": {"existing": "keep"},
                }
            )
        )
    )

    _install_fast_agent_model_references(
        fast,
        model_references={"system": {"skill_gen": "sonnet", "test_gen": "opus"}},
    )

    assert fast.app._config_or_path.model_references == {
        "custom": {"router": "haiku"},
        "system": {
            "existing": "keep",
            "skill_gen": "sonnet",
            "test_gen": "opus",
        },
    }


@pytest.mark.asyncio
async def test_generate_persists_generated_tests_in_skill_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        skills_dir=tmp_path / "skills",
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    test_cases = [
        TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
        TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
    ]
    fake_executor = object()

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context", lambda *_args, **_kwargs: _FakeAgentContext()
    )

    monkeypatch.setattr("upskill.cli._build_executor", lambda *args, **kwargs: fake_executor)

    async def fake_generate_skill(**kwargs: object) -> SkillRecord:
        del kwargs
        return SkillRecord(
            skill=Skill(
                name="pull-request-descriptions",
                description="Write good pull request descriptions.",
                body="Use a clear structure.",
            )
        )

    async def fake_generate_tests(*args: object, **kwargs: object) -> list[TestCase]:
        del args, kwargs
        return test_cases

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        skill = args[0]
        assert isinstance(skill, Skill)
        assert kwargs["executor"] is fake_executor
        assert kwargs["operation"] == "generate"
        return _make_eval_results(
            skill=skill,
            model=str(kwargs["model"]),
            test_cases=test_cases,
            run_baseline=True,
        )

    monkeypatch.setattr("upskill.cli.generate_skill", fake_generate_skill)
    monkeypatch.setattr("upskill.cli.generate_tests", fake_generate_tests)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _generate_async(
        task="write good pull request descriptions",
        examples=None,
        from_skill=None,
        from_trace=None,
        model="haiku",
        test_gen_model=None,
        output=None,
        no_eval=False,
        eval_model=None,
        executor_name="local",
        artifact_repo=None,
        wait=False,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=2,
        runs_dir=str(config.runs_dir),
        log_runs=True,
    )

    saved = SkillRecord.load(config.skills_dir / "pull-request-descriptions")
    assert len(saved.state.tests) == 2
    assert saved.state.tests[0].input == "prompt one"


@pytest.mark.asyncio
async def test_generate_no_eval_still_persists_generated_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        skills_dir=tmp_path / "skills",
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    test_cases = [
        TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
        TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
    ]

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context", lambda *_args, **_kwargs: _FakeAgentContext()
    )

    async def fake_generate_skill(**kwargs: object) -> SkillRecord:
        del kwargs
        return SkillRecord(
            skill=Skill(
                name="pull-request-descriptions",
                description="Write good pull request descriptions.",
                body="Use a clear structure.",
            )
        )

    async def fake_generate_tests(*args: object, **kwargs: object) -> list[TestCase]:
        del args, kwargs
        return test_cases

    async def fail_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args, kwargs
        raise AssertionError("evaluate_skill should not be called when --no-eval is set")

    monkeypatch.setattr("upskill.cli.generate_skill", fake_generate_skill)
    monkeypatch.setattr("upskill.cli.generate_tests", fake_generate_tests)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fail_evaluate_skill)

    await _generate_async(
        task="write good pull request descriptions",
        examples=None,
        from_skill=None,
        from_trace=None,
        model="haiku",
        test_gen_model=None,
        output=None,
        no_eval=True,
        eval_model=None,
        executor_name="local",
        artifact_repo=None,
        wait=False,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=2,
        runs_dir=str(config.runs_dir),
        log_runs=True,
    )

    saved = SkillRecord.load(config.skills_dir / "pull-request-descriptions")
    assert len(saved.state.tests) == 2
    assert saved.state.tests[1].input == "prompt two"


@pytest.mark.asyncio
async def test_generate_prints_test_generation_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        skills_dir=tmp_path / "skills",
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    test_cases = [
        TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
    ]
    printed_messages: list[str] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context", lambda *_args, **_kwargs: _FakeAgentContext()
    )
    monkeypatch.setattr(
        "upskill.cli.console.print",
        lambda *args, **kwargs: printed_messages.append(" ".join(str(arg) for arg in args)),
    )

    async def fake_generate_skill(**kwargs: object) -> SkillRecord:
        del kwargs
        return SkillRecord(
            skill=Skill(
                name="pull-request-descriptions",
                description="Write good pull request descriptions.",
                body="Use a clear structure.",
            )
        )

    async def fake_generate_tests(*args: object, **kwargs: object) -> list[TestCase]:
        del args, kwargs
        return test_cases

    monkeypatch.setattr("upskill.cli.generate_skill", fake_generate_skill)
    monkeypatch.setattr("upskill.cli.generate_tests", fake_generate_tests)

    await _generate_async(
        task="write good pull request descriptions",
        examples=None,
        from_skill=None,
        from_trace=None,
        model="haiku",
        test_gen_model="opus",
        output=None,
        no_eval=True,
        eval_model=None,
        executor_name="local",
        artifact_repo=None,
        wait=False,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=2,
        runs_dir=str(config.runs_dir),
        log_runs=True,
    )

    assert any("Generating test cases with opus..." in message for message in printed_messages)


@pytest.mark.asyncio
async def test_generate_jobs_no_wait_submits_remote_eval_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        skills_dir=tmp_path / "skills",
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    test_cases = [
        TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
        TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
    ]
    submit_models: list[str] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context", lambda *_args, **_kwargs: _FakeAgentContext()
    )

    async def fake_generate_skill(**kwargs: object) -> SkillRecord:
        del kwargs
        return SkillRecord(
            skill=Skill(
                name="pull-request-descriptions",
                description="Write good pull request descriptions.",
                body="Use a clear structure.",
            )
        )

    async def fake_generate_tests(*args: object, **kwargs: object) -> list[TestCase]:
        del args, kwargs
        return test_cases

    async def fake_submit_generate_jobs_eval(**kwargs: object) -> list[str]:
        submit_models.append(str(kwargs["model"]))
        assert kwargs["test_cases"] == test_cases
        return ["evalstate/job-1", "evalstate/job-2"]

    def fail_build_executor(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("_build_executor should not be used for jobs --no-wait submission")

    async def fail_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args, kwargs
        raise AssertionError("evaluate_skill should not be called for jobs --no-wait submission")

    monkeypatch.setattr("upskill.cli.generate_skill", fake_generate_skill)
    monkeypatch.setattr("upskill.cli.generate_tests", fake_generate_tests)
    monkeypatch.setattr("upskill.cli._submit_generate_jobs_eval", fake_submit_generate_jobs_eval)
    monkeypatch.setattr("upskill.cli._build_executor", fail_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fail_evaluate_skill)

    await _generate_async(
        task="write good pull request descriptions",
        examples=None,
        from_skill=None,
        from_trace=None,
        model="haiku",
        test_gen_model=None,
        output=None,
        no_eval=False,
        eval_model=None,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=False,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=2,
        runs_dir=str(config.runs_dir),
        log_runs=True,
    )

    saved = SkillRecord.load(config.skills_dir / "pull-request-descriptions")
    assert len(saved.state.tests) == 2
    assert submit_models == ["haiku"]


@pytest.mark.asyncio
async def test_submit_generate_jobs_eval_marks_operation_as_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(fastagent_config=tmp_path / "fastagent.config.yaml")
    test_cases = [TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"]))]
    skill = Skill(
        name="pull-request-descriptions",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    operation_calls: list[str] = []

    async def fake_submit_remote_eval_jobs(**kwargs: object) -> list[str]:
        operation_calls.append(str(kwargs["operation"]))
        return ["evalstate/job-1"]

    monkeypatch.setattr("upskill.cli._submit_remote_eval_jobs", fake_submit_remote_eval_jobs)

    job_refs = await _submit_generate_jobs_eval(
        skill=skill,
        test_cases=test_cases,
        model="haiku",
        jobs_config=JobsConfig(artifact_repo="ns/repo"),
        config=config,
        cards_path=tmp_path / "cards",
        batch_folder=tmp_path / "runs" / "batch_1",
    )

    assert job_refs == ["evalstate/job-1"]
    assert operation_calls == ["generate"]


@pytest.mark.asyncio
async def test_benchmark_jobs_uses_remote_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
    )
    skill_record = SkillRecord(
        skill=Skill(
            name="pull-request-descriptions",
            description="Write good pull request descriptions.",
            body="Use a clear structure.",
        ),
        state=SkillState(
            tests=[
                TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
                TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
            ]
        ),
    )
    skill_dir = tmp_path / "skill"
    skill_record.save(skill_dir)
    fake_executor = object()
    build_calls: list[str] = []
    eval_calls: list[int] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli._fast_agent_context", lambda *_args, **_kwargs: _FakeAgentContext()
    )

    def fake_build_executor(name: str, **kwargs: object) -> object:
        del kwargs
        build_calls.append(name)
        return fake_executor

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        skill = args[0]
        assert isinstance(skill, Skill)
        assert kwargs["executor"] is fake_executor
        assert kwargs["operation"] == "benchmark"
        max_parallel = kwargs["max_parallel"]
        assert isinstance(max_parallel, int)
        eval_calls.append(max_parallel)
        return _make_eval_results(
            skill=skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
            run_baseline=False,
        )

    monkeypatch.setattr("upskill.cli._build_executor", fake_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _benchmark_async(
        skill_path=str(skill_dir),
        models=["haiku"],
        test_gen_model=None,
        num_runs=2,
        tests_path=None,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        output_dir=str(config.runs_dir),
        verbose=False,
        max_parallel=4,
    )

    assert build_calls == ["jobs"]
    assert eval_calls == [4, 4]
    batch_folder = next(config.runs_dir.iterdir())
    summary = load_batch_summary(batch_folder)
    assert summary is not None
    assert summary.total_runs == 2


@pytest.mark.asyncio
async def test_benchmark_uses_config_execution_defaults_when_cli_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
        executor="jobs",
        num_runs=2,
        max_parallel=6,
    )
    skill_record = SkillRecord(
        skill=Skill(
            name="pull-request-descriptions",
            description="Write good pull request descriptions.",
            body="Use a clear structure.",
        ),
        state=SkillState(
            tests=[
                TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
                TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
            ]
        ),
    )
    skill_dir = tmp_path / "skill"
    skill_record.save(skill_dir)
    fake_executor = object()
    build_calls: list[str] = []
    benchmark_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    def fake_build_executor(name: str, **kwargs: object) -> object:
        del kwargs
        build_calls.append(name)
        return fake_executor

    async def fake_run_with_skill_benchmark(*args: object, **kwargs: object):
        del args
        executor_factory = cast("Callable[[str], object]", kwargs["executor_factory"])
        assert executor_factory("haiku") is fake_executor
        num_runs = kwargs["num_runs"]
        max_parallel = kwargs["max_parallel"]
        assert isinstance(num_runs, int)
        assert isinstance(max_parallel, int)
        benchmark_calls.append((num_runs, max_parallel))
        return {}, []

    monkeypatch.setattr("upskill.cli._build_executor", fake_build_executor)
    monkeypatch.setattr("upskill.cli._run_with_skill_benchmark", fake_run_with_skill_benchmark)
    monkeypatch.setattr("upskill.cli._print_benchmark_summary", lambda _results: None)
    monkeypatch.setattr("upskill.cli._write_benchmark_summary", lambda **_kwargs: None)

    await _benchmark_async(
        skill_path=str(skill_dir),
        models=["haiku"],
        test_gen_model=None,
        num_runs=None,
        tests_path=None,
        executor_name=None,
        artifact_repo="ns/repo",
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        output_dir=str(config.runs_dir),
        verbose=False,
        max_parallel=None,
    )

    assert build_calls == ["jobs"]
    assert benchmark_calls == [(2, 6)]

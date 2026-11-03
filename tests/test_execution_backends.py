from __future__ import annotations

import asyncio
import json
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Never

import pytest
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from mcp.types import TextContent

from upskill.artifacts import materialize_workspace
from upskill.evaluate import (
    evaluate_skill,
    get_failure_descriptions,
    load_eval_results_from_artifact_root,
)
from upskill.executors.contracts import ExecutionHandle, ExecutionRequest, ExecutionResult
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.executors.remote_fast_agent import RemoteFastAgentExecutor
from upskill.fast_agent_cli import build_fast_agent_command
from upskill.hf_jobs import JobsConfig, SubmittedJob
from upskill.models import (
    ConversationStats,
    ExpectedSpec,
    Skill,
    TestCase,
    TestResult,
    VerifierSpec,
)
from upskill.result_parsing import parse_fast_agent_results
from upskill.validators import register_validator


class EvaluationInterrupted(BaseException):
    """Process control exception that must not become an evaluation result."""


def _write_result_history(path: Path, *, assistant_text: str) -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Do the task")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=assistant_text)],
        ),
    ]
    save_json(messages, str(path))


def _build_request(tmp_path: Path) -> ExecutionRequest:
    cards_dir = tmp_path / "cards-source"
    cards_dir.mkdir()
    (cards_dir / "evaluator.md").write_text("---\ndescription: evaluator\n---\n{{agentSkills}}\n")
    (cards_dir / "skill_gen.md").write_text(
        "---\ndescription: skill generator\n---\nGenerate skills\n"
    )
    (cards_dir / "test_gen.md").write_text(
        "---\ndescription: test generator\n---\nGenerate tests\n"
    )
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n")
    return ExecutionRequest(
        prompt="Do the task",
        model="haiku",
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=tmp_path / "artifacts" / "run_1",
        cards_source_dir=cards_dir,
        label="test run",
        skill=Skill(
            name="write-good-prs",
            description="Write good pull request descriptions.",
            body="Use a clear structure.",
        ),
        workspace_files={"context.txt": "hello"},
    )


def test_build_fast_agent_command_uses_explicit_contract(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    prompt_path = tmp_path / "bundle" / "prompt.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(request.prompt, encoding="utf-8")
    command = build_fast_agent_command(
        request,
        config_path=request.fastagent_config_path,
        cards_dir=tmp_path / "bundle" / "cards",
        skills_dir=tmp_path / "bundle" / "skills",
        prompt_path=prompt_path,
        results_path=tmp_path / "bundle" / "results.json",
        fast_agent_bin="fast-agent",
    )

    assert command[:2] == ["fast-agent", "go"]
    assert "--config-path" in command
    assert "--card" in command
    assert "--agent" in command
    assert "--model" in command
    assert "--skills-dir" in command
    assert "--prompt-file" in command
    assert "--results" in command
    assert "--quiet" in command


def test_build_fast_agent_command_omits_missing_config_path(tmp_path: Path) -> None:
    request = replace(_build_request(tmp_path), fastagent_config_path=tmp_path / "missing.yaml")
    prompt_path = tmp_path / "bundle" / "prompt.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(request.prompt, encoding="utf-8")

    command = build_fast_agent_command(
        request,
        config_path=None,
        cards_dir=tmp_path / "bundle" / "cards",
        skills_dir=tmp_path / "bundle" / "skills",
        prompt_path=prompt_path,
        results_path=tmp_path / "bundle" / "results.json",
    )

    assert "--config-path" not in command
    assert "--prompt-file" in command


def test_parse_fast_agent_results_extracts_output_text(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    _write_result_history(results_path, assistant_text="Structured answer")

    parsed = parse_fast_agent_results(results_path)

    assert parsed.output_text == "Structured answer"
    assert parsed.stats.turns == 1


@pytest.mark.asyncio
async def test_local_fast_agent_executor_preserves_artifacts_and_parses_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"assistant output\n", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        del kwargs
        results_index = args.index("--results") + 1
        results_path = Path(args[results_index])
        _write_result_history(results_path, assistant_text="Final answer")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text == "Final answer"
    assert result.raw_results_path == request.artifact_dir / "results.json"
    assert (request.artifact_dir / "request.json").exists()
    assert (request.artifact_dir / "stdout.txt").exists()
    assert (request.artifact_dir / "stderr.txt").exists()
    assert (request.artifact_dir / "workspace" / "context.txt").read_text(
        encoding="utf-8"
    ) == "hello"
    assert (request.artifact_dir / "workspace" / "fastagent.config.yaml").exists()
    assert (request.artifact_dir / "cards" / "evaluator.md").exists()
    assert not (request.artifact_dir / "cards" / "skill_gen.md").exists()
    assert not (request.artifact_dir / "cards" / "test_gen.md").exists()
    assert (request.artifact_dir / "skills" / "write-good-prs" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_local_fast_agent_executor_fails_when_results_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error == "fast-agent run did not produce a results artifact."
    assert result.raw_results_path is None


@pytest.mark.asyncio
async def test_local_fast_agent_executor_omits_missing_config_from_command_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(_build_request(tmp_path), fastagent_config_path=tmp_path / "missing.yaml")
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        del kwargs
        assert "--config-path" not in args
        assert "--prompt-file" in args
        results_index = args.index("--results") + 1
        _write_result_history(Path(args[results_index]), assistant_text="Final answer")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert not (request.artifact_dir / "fastagent.config.yaml").exists()
    assert not (request.artifact_dir / "workspace" / "fastagent.config.yaml").exists()


@pytest.mark.asyncio
async def test_remote_fast_agent_executor_preserves_artifacts_and_parses_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = RemoteFastAgentExecutor(jobs_config=JobsConfig(artifact_repo="ns/repo"))
    submitted_labels: dict[str, str] = {}

    def fake_submit_bundle_job(**kwargs: object) -> SubmittedJob:
        nonlocal submitted_labels
        labels = kwargs["labels"]
        assert isinstance(labels, dict)
        assert all(isinstance(key, str) and isinstance(value, str) for key, value in labels.items())
        submitted_labels = {str(key): str(value) for key, value in labels.items()}
        del kwargs
        return SubmittedJob(
            job_id="evalstate/job-123",
            run_id="run-456",
            artifact_repo="ns/repo",
        )

    def fake_wait_for_job_outputs(
        job: SubmittedJob,
        *,
        destination_root: Path,
        wait_timeout_seconds: float,
        progress_callback: object = None,
    ) -> Path:
        del wait_timeout_seconds, progress_callback
        output_dir = destination_root / "outputs" / job.run_id
        (output_dir / "results").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (output_dir / "status").mkdir(parents=True, exist_ok=True)
        (output_dir / "workspaces" / "request_1").mkdir(parents=True, exist_ok=True)
        _write_result_history(
            output_dir / "results" / "request_1.json", assistant_text="Remote answer"
        )
        (output_dir / "logs" / "request_1.out.txt").write_text("stdout\n", encoding="utf-8")
        (output_dir / "logs" / "request_1.err.txt").write_text("", encoding="utf-8")
        (output_dir / "status" / "request_1.exit_code.txt").write_text("0\n", encoding="utf-8")
        (output_dir / "workspaces" / "request_1" / "context.txt").write_text(
            "remote hello",
            encoding="utf-8",
        )
        return output_dir

    monkeypatch.setattr(
        "upskill.executors.remote_fast_agent._submit_bundle_job",
        fake_submit_bundle_job,
    )
    monkeypatch.setattr(
        "upskill.executors.remote_fast_agent._make_run_id",
        lambda *_args: "run-456",
    )
    monkeypatch.setattr(
        "upskill.executors.remote_fast_agent.wait_for_job_outputs",
        fake_wait_for_job_outputs,
    )

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text == "Remote answer"
    assert result.raw_results_path == request.artifact_dir / "results.json"
    assert result.metadata["job_id"] == "evalstate/job-123"
    assert (request.artifact_dir / "stdout.txt").exists()
    assert (request.artifact_dir / "stderr.txt").exists()
    assert (request.artifact_dir / "remote_output" / "results" / "request_1.json").exists()
    assert (request.artifact_dir / "workspace" / "context.txt").read_text(
        encoding="utf-8"
    ) == "remote hello"
    assert not (request.artifact_dir / "cards" / "skill_gen.md").exists()
    assert not (request.artifact_dir / "cards" / "test_gen.md").exists()
    assert submitted_labels == {
        "upskill-agent": "evaluator",
        "upskill-executor": "remote-fast-agent",
        "upskill-model": "haiku",
        "upskill-operation": "eval",
        "upskill-request": "test-run",
        "upskill-run-id": "run-456",
        "upskill-skill": "write-good-prs",
    }


@pytest.mark.asyncio
async def test_remote_fast_agent_executor_submit_preserves_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = RemoteFastAgentExecutor(jobs_config=JobsConfig(artifact_repo="ns/repo"))

    def fake_submit_bundle_job(**kwargs: object) -> SubmittedJob:
        del kwargs
        return SubmittedJob(
            job_id="evalstate/job-123",
            run_id="run-456",
            artifact_repo="ns/repo",
        )

    def fail_wait_for_job_outputs(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise AssertionError("submit() should not wait for job outputs")

    monkeypatch.setattr(
        "upskill.executors.remote_fast_agent._submit_bundle_job",
        fake_submit_bundle_job,
    )
    monkeypatch.setattr(
        "upskill.executors.remote_fast_agent.wait_for_job_outputs",
        fail_wait_for_job_outputs,
    )

    submission = await executor.submit(request)

    assert submission == SubmittedJob(
        job_id="evalstate/job-123",
        run_id="run-456",
        artifact_repo="ns/repo",
    )
    assert (request.artifact_dir / "request.json").exists()
    assert (request.artifact_dir / "prompt.txt").exists()
    assert (request.artifact_dir / "cards" / "evaluator.md").exists()
    assert not (request.artifact_dir / "cards" / "skill_gen.md").exists()
    assert not (request.artifact_dir / "cards" / "test_gen.md").exists()
    assert (request.artifact_dir / "skills" / "write-good-prs" / "SKILL.md").exists()
    submitted_job = json.loads(
        (request.artifact_dir / "submitted_job.json").read_text(encoding="utf-8")
    )
    assert submitted_job["job_id"] == "evalstate/job-123"
    assert submitted_job["run_id"] == "run-456"


def test_remote_fast_agent_executor_bundle_omits_missing_config(tmp_path: Path) -> None:
    request = replace(_build_request(tmp_path), fastagent_config_path=tmp_path / "missing.yaml")
    executor = RemoteFastAgentExecutor(jobs_config=JobsConfig(artifact_repo="ns/repo"))

    temp_root, bundle_archive = executor._create_bundle_archive(request)
    try:
        with tarfile.open(bundle_archive, "r:gz") as archive:
            assert "bundle/fastagent.config.yaml" not in archive.getnames()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_local_fast_agent_executor_normalizes_paths_and_preserves_file_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cards_dir = Path("cards-source")
    cards_dir.mkdir()
    (cards_dir / "evaluator.md").write_text("---\ndescription: evaluator\n---\n{{agentSkills}}\n")
    config_path = Path("fastagent.config.yaml")
    config_path.write_text("default_model: sonnet\n")
    request = ExecutionRequest(
        prompt="Base prompt\n\n```context.txt\nhello\n```",
        model="haiku",
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=Path("artifacts") / "run_1",
        cards_source_dir=cards_dir,
        label="test run",
        workspace_files={"context.txt": "hello"},
    )
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"assistant output\n", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        assert cwd.is_absolute()
        results_index = args.index("--results") + 1
        prompt_index = args.index("--prompt-file") + 1
        cards_index = args.index("--card") + 1
        skills_index = args.index("--skills-dir") + 1
        config_index = args.index("--config-path") + 1
        agent_index = args.index("--agent") + 1
        assert args[agent_index] == "evaluator"
        for index in (results_index, cards_index, skills_index, config_index, prompt_index):
            if index == prompt_index:
                continue
            assert Path(args[index]).is_absolute()
        prompt_text = Path(args[prompt_index]).read_text(encoding="utf-8")
        assert "```context.txt\nhello\n```" in prompt_text
        _write_result_history(Path(args[results_index]), assistant_text="Final answer")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None


def test_materialize_workspace_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not traverse parents"):
        materialize_workspace(tmp_path / "workspace", {"../pyproject.toml": "oops"})

    with pytest.raises(ValueError, match="must be relative"):
        materialize_workspace(tmp_path / "workspace", {"/tmp/pwned": "oops"})


def test_load_eval_results_from_artifact_root_reconstructs_metrics(tmp_path: Path) -> None:
    artifact_root = tmp_path / "eval"
    with_skill_dir = artifact_root / "with-skill" / "test_1"
    baseline_dir = artifact_root / "baseline" / "test_1"
    with_skill_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))
    with_skill_result = TestResult(test_case=test_case, success=True, output="answer")
    baseline_result = TestResult(test_case=test_case, success=False, output="miss")

    (with_skill_dir / "test_result.json").write_text(
        with_skill_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (baseline_dir / "test_result.json").write_text(
        baseline_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    reconstructed = load_eval_results_from_artifact_root(
        skill_name="write-good-prs",
        model="qwen35",
        artifact_root=artifact_root,
    )

    assert reconstructed is not None
    assert reconstructed.with_skill_success_rate == 1.0
    assert reconstructed.baseline_success_rate == 0.0


@pytest.mark.asyncio
async def test_evaluate_skill_emits_per_test_progress_messages(tmp_path: Path) -> None:
    skill = Skill(
        name="write-good-prs",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))
    messages: list[str] = []

    class FakeExecutor:
        async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = request.artifact_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task = asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=ExecutionResult(
                        output_text="answer",
                        raw_results_path=None,
                        stdout_path=request.artifact_dir / "stdout.txt",
                        stderr_path=request.artifact_dir / "stderr.txt",
                        artifact_dir=request.artifact_dir,
                        workspace_dir=workspace_dir,
                        stats=ConversationStats(),
                    ),
                )
            )
            return ExecutionHandle(request=request, task=task)

        async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
            return await handle.task

        async def cancel(self, handle: ExecutionHandle) -> None:
            handle.task.cancel()

    results = await evaluate_skill(
        skill,
        [test_case],
        FakeExecutor(),
        model="haiku",
        fastagent_config_path=tmp_path / "fastagent.config.yaml",
        cards_source_dir=tmp_path,
        artifact_root=tmp_path / "eval",
        progress_callback=messages.append,
    )

    assert results.with_skill_success_rate == 1.0
    assert "starting with-skill test 1/1" in messages
    assert "finished with-skill test 1/1 (ok)" in messages


@pytest.mark.asyncio
async def test_evaluate_skill_supports_verifier_only_test_cases(tmp_path: Path) -> None:
    skill = Skill(
        name="write-good-prs",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    test_case = TestCase(
        input="write a report",
        verifiers=[
            VerifierSpec(type="file_exists", path="report.txt"),
            VerifierSpec(type="file_contains", path="report.txt", text="answer"),
        ],
    )

    class FakeExecutor:
        async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = request.artifact_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "report.txt").write_text("answer in report", encoding="utf-8")
            task = asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=ExecutionResult(
                        output_text="done",
                        raw_results_path=None,
                        stdout_path=request.artifact_dir / "stdout.txt",
                        stderr_path=request.artifact_dir / "stderr.txt",
                        artifact_dir=request.artifact_dir,
                        workspace_dir=workspace_dir,
                        stats=ConversationStats(),
                    ),
                )
            )
            return ExecutionHandle(request=request, task=task)

        async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
            return await handle.task

        async def cancel(self, handle: ExecutionHandle) -> None:
            handle.task.cancel()

    results = await evaluate_skill(
        skill,
        [test_case],
        FakeExecutor(),
        model="haiku",
        fastagent_config_path=tmp_path / "fastagent.config.yaml",
        cards_source_dir=tmp_path,
        artifact_root=tmp_path / "eval",
    )

    test_result = results.with_skill_results[0]
    assert test_result.success is True
    assert test_result.validation_result is not None
    assert test_result.validation_result.assertions_passed == 2
    assert test_result.validation_result.assertions_total == 2


@pytest.mark.asyncio
async def test_evaluate_skill_includes_job_id_in_execution_errors(tmp_path: Path) -> None:
    skill = Skill(
        name="write-good-prs",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))

    class FakeExecutor:
        async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = request.artifact_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task = asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=ExecutionResult(
                        output_text=None,
                        raw_results_path=None,
                        stdout_path=request.artifact_dir / "stdout.txt",
                        stderr_path=request.artifact_dir / "stderr.txt",
                        artifact_dir=request.artifact_dir,
                        workspace_dir=workspace_dir,
                        stats=ConversationStats(),
                        error="fast-agent exited with code 1.",
                        metadata={"job_id": "evalstate/job-123"},
                    ),
                )
            )
            return ExecutionHandle(request=request, task=task)

        async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
            return await handle.task

        async def cancel(self, handle: ExecutionHandle) -> None:
            handle.task.cancel()

    results = await evaluate_skill(
        skill,
        [test_case],
        FakeExecutor(),
        model="haiku",
        fastagent_config_path=tmp_path / "fastagent.config.yaml",
        cards_source_dir=tmp_path,
        artifact_root=tmp_path / "eval",
        progress_callback=None,
    )

    assert (
        results.with_skill_results[0].error
        == "fast-agent exited with code 1. (job evalstate/job-123)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("binary", "UnicodeDecodeError"),
        ("directory", "IsADirectoryError"),
        ("timeout", "TimeoutExpired"),
        ("validator", "RuntimeError"),
        ("interrupted", "EvaluationInterrupted"),
    ],
)
async def test_evaluate_skill_preserves_verifier_errors_and_finishes_batches(
    tmp_path: Path, failure: str, error_type: str
) -> None:
    @register_validator("raises-during-evaluation")
    def broken_validator(**_: object) -> Never:
        if failure == "interrupted":
            raise EvaluationInterrupted("stop evaluation")
        raise RuntimeError("validator could not inspect artifact")

    spec = VerifierSpec(type="file_contains", path="report.txt", text="answer")
    if failure == "timeout":
        # A zero-second deadline deterministically times out the shell without
        # starting a long-lived child process.
        spec = VerifierSpec(type="command", cmd="exit 0", config={"timeout_seconds": 0})
    elif failure in ("validator", "interrupted"):
        spec = VerifierSpec(type="validator", name="raises-during-evaluation")
    cases = [
        TestCase(input="bad artifact", verifiers=[spec]),
        TestCase(input="good answer", expected=ExpectedSpec(contains=["answer"])),
        TestCase(input="wrong answer", expected=ExpectedSpec(contains=["missing text"])),
    ]
    stats = ConversationStats(total_tokens=17, turns=2, tool_calls=1)
    requests: list[ExecutionRequest] = []

    class ArtifactExecutor:
        async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
            requests.append(request)
            workspace = request.artifact_dir / "workspace"
            workspace.mkdir(parents=True)
            if failure == "directory":
                (workspace / "report.txt").mkdir()
            else:
                (workspace / "report.txt").write_bytes(b"\xff")
            task = asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=ExecutionResult(
                        output_text="answer",
                        raw_results_path=None,
                        stdout_path=request.artifact_dir / "stdout.txt",
                        stderr_path=request.artifact_dir / "stderr.txt",
                        artifact_dir=request.artifact_dir,
                        workspace_dir=workspace,
                        stats=stats,
                    ),
                )
            )
            return ExecutionHandle(request=request, task=task)

        async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
            return await handle.task

        async def cancel(self, handle: ExecutionHandle) -> None:
            handle.task.cancel()

    root = tmp_path / "evaluation"
    evaluation = evaluate_skill(
        Skill(name="write-report", description="Write reports", body="Write a report."),
        cases,
        ArtifactExecutor(),
        model="test-model",
        fastagent_config_path=tmp_path / "config.yaml",
        cards_source_dir=tmp_path,
        artifact_root=root,
        max_parallel=2,
    )
    if failure == "interrupted":
        with pytest.raises(EvaluationInterrupted, match="stop evaluation"):
            await evaluation
        assert not (root / "with-skill" / "test_1" / "test_result.json").exists()
        assert not (root / "baseline").exists()
        return
    results = await evaluation
    assert len(requests) == 6
    for batch in (results.with_skill_results, results.baseline_results):
        failed, successful, mismatch = batch
        assert not failed.success
        assert failed.error is not None and error_type in failed.error
        assert failed.output == "answer"
        assert failed.stats == stats
        assert failed.tokens_used == 17 and failed.turns == 2
        assert failed.validation_result is None
        assert successful.success and successful.error is None
        assert not mismatch.success and mismatch.error is None
        assert mismatch.validation_result is not None
        assert mismatch.validation_result.assertions_total == 1
    assert results.with_skill_success_rate == pytest.approx(1 / 3)
    assert results.baseline_success_rate == pytest.approx(1 / 3)
    assert results.with_skill_total_tokens == 51
    assert error_type in get_failure_descriptions(results)[0]
    restored = load_eval_results_from_artifact_root(
        skill_name="write-report", model="test-model", artifact_root=root
    )
    assert restored == results

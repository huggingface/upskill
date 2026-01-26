"""CLI interface for upskill."""
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import TypedDict

import click
from dotenv import load_dotenv
from fast_agent import FastAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from upskill.config import Config
from upskill.evaluate import evaluate_skill, get_failure_descriptions
from upskill.generate import generate_skill, generate_tests, improve_skill, refine_skill
from upskill.logging import (
    aggregate_conversation_stats,
    create_batch_folder,
    create_run_folder,
    load_batch_summary,
    load_run_result,
    summarize_runs_to_csv,
    write_batch_summary,
    write_run_metadata,
    write_run_result,
)
from upskill.models import (
    BatchSummary,
    RunMetadata,
    RunResult,
    Skill,
    TestCase,
    TestResult,
)

load_dotenv()

console = Console()


@asynccontextmanager
async def _fast_agent_context() -> AsyncIterator[object]:
    fast = FastAgent(
        "upskill",
        ignore_unknown_args=True,
    )

    @fast.agent()
    async def empty():
        pass

    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        fast.load_agents(cards_path)

    async with fast.run() as agent:
        yield agent


def _render_bar(value: float, width: int = 20) -> str:
    """Render a simple text bar for a 0-1 value."""
    if width <= 0:
        return ""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    empty = width - filled
    return "█" * filled + "░" * empty


class EvalPlotResult(TypedDict):
    """Structured plot data for eval runs."""

    model: str
    skill_name: str
    with_skill_rate: float
    with_skill_tokens: int
    baseline_rate: float
    baseline_tokens: int
    has_baseline: bool


def _success_rate(run: RunResult) -> float:
    """Compute a success rate from a run result."""
    if run.assertions_total == 0:
        return 0.0
    return run.assertions_passed / run.assertions_total


def _select_baseline_run(
    baseline_runs: list[RunResult],
    with_skill_run: RunResult,
) -> RunResult | None:
    """Select the most relevant baseline run for a with-skill run."""
    if not baseline_runs:
        return None
    with_number = with_skill_run.metadata.run_number
    eligible = [run for run in baseline_runs if run.metadata.run_number <= with_number]
    if eligible:
        return eligible[-1]
    return baseline_runs[-1]


def _load_eval_results(runs_path: Path) -> list[EvalPlotResult]:
    """Load eval results from batch summaries or run folders."""
    results: list[EvalPlotResult] = []
    if not runs_path.exists():
        return results

    batch_folders = sorted(p for p in runs_path.iterdir() if p.is_dir())
    for batch_folder in batch_folders:
        run_results: list[RunResult] = []
        summary = load_batch_summary(batch_folder)
        if summary:
            run_results = summary.results
        else:
            for run_folder in sorted(batch_folder.glob("run_*")):
                if not run_folder.is_dir():
                    continue
                run_result = load_run_result(run_folder)
                if run_result:
                    run_results.append(run_result)

        if not run_results:
            continue

        grouped: dict[tuple[str, str], dict[str, list[RunResult]]] = {}
        for run in run_results:
            skill_name = run.skill_name or "unknown"
            key = (run.metadata.model, skill_name)
            grouped.setdefault(key, {"baseline": [], "with_skill": []})
            if run.run_type == "baseline":
                grouped[key]["baseline"].append(run)
            else:
                grouped[key]["with_skill"].append(run)

        for (model, skill_name), runs in grouped.items():
            baseline_runs = sorted(runs["baseline"], key=lambda r: r.metadata.run_number)
            for with_skill_run in runs["with_skill"]:
                baseline_run = _select_baseline_run(baseline_runs, with_skill_run)
                result: EvalPlotResult = {
                    "model": model,
                    "skill_name": skill_name,
                    "with_skill_rate": _success_rate(with_skill_run),
                    "with_skill_tokens": with_skill_run.stats.total_tokens,
                    "baseline_rate": _success_rate(baseline_run) if baseline_run else 0.0,
                    "baseline_tokens": baseline_run.stats.total_tokens if baseline_run else 0,
                    "has_baseline": baseline_run is not None,
                }
                results.append(result)

    return results


@click.group()
@click.version_option()
def main():
    """upskill - Generate and evaluate agent skills."""


@main.command()
@click.argument("task")
@click.option("-e", "--example", multiple=True, help="Input -> output example")
@click.option("--tool", help="Generate from MCP tool schema (path#tool_name)")
@click.option(
    "-f",
    "--from",
    "from_source",
    type=click.Path(exists=True),
    help="Improve existing skill (directory) or generate from trace (file)",
)
@click.option(
    "-m",
    "--model",
    help="Model for generation (e.g., 'sonnet', 'anthropic.claude-sonnet-4-20250514')",
)
@click.option("-o", "--output", type=click.Path(), help="Output directory for skill")
@click.option("--no-eval", is_flag=True, help="Skip eval and refinement")
@click.option("--eval-model", help="Model to evaluate skill on (different from generation model)")
@click.option("--runs-dir", type=click.Path(), help="Directory for run logs (default: ./runs)")
@click.option("--log-runs/--no-log-runs", default=True, help="Log run data (default: enabled)")
def generate(
    task: str,
    example: tuple[str, ...],
    tool: str | None,  # noqa: ARG001
    from_source: str | None,
    model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    runs_dir: str | None,
    log_runs: bool,
):
    """Generate a skill from a task description, or improve an existing skill.

    Examples:

        upskill generate "parse JSON Schema files"

        upskill generate "write git commits" --model sonnet

        upskill generate "handle API errors" --eval-model haiku

        upskill generate "validate forms" -o ./my-skills/validation

        # Improve an existing skill (auto-detects directory):

        upskill generate "add more error handling examples" --from ./skills/api-errors/

        upskill generate "make it more concise" -f ./skills/my-skill/ -o ./skills/my-skill-v2/

        # Generate from trace file (auto-detects file):

        upskill generate "extract patterns" --from trace.json

        # Evaluate on a local model (Ollama):

        upskill generate "parse YAML" --eval-model llama3.2:latest \\
            --eval-base-url http://localhost:11434/v1

        upskill generate "document code" --no-log-runs
    """
    # Auto-detect if --from is a skill directory or trace file
    from_skill = None
    from_trace = None
    if from_source:
        source_path = Path(from_source)
        if source_path.is_dir():
            from_skill = from_source
        else:
            from_trace = from_source

    asyncio.run(
        _generate_async(
            task,
            list(example) if example else None,
            from_skill,
            from_trace,
            model,
            output,
            no_eval,
            eval_model,
            runs_dir,
            log_runs,
        )
    )


async def _generate_async(
    task: str,
    examples: list[str] | None,
    from_skill: str | None,
    from_trace: str | None,
    model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    runs_dir: str | None,
    log_runs: bool,
):
    """Async implementation of generate command."""
    config = Config.load()
    gen_model = model or config.model

    # Setup run logging if enabled
    batch_id = None
    batch_folder = None
    run_results: list[RunResult] = []

    if log_runs:
        runs_path = Path(runs_dir) if runs_dir else config.runs_dir
        batch_id, batch_folder = create_batch_folder(runs_path)
        console.print(f"Logging runs to: {batch_folder}", style="dim")


    skill: Skill | None = None
    results = None

    async with _fast_agent_context() as agent:

        # Generate from trace file
        if from_trace:
            console.print(f"Generating skill from trace: {from_trace}", style="dim")
            trace_path = Path(from_trace)
            with open(trace_path, encoding="utf-8") as f:
                trace_content = f.read()

            # Try to parse as JSON, otherwise use as plain text
            if trace_path.suffix.lower() == ".json":
                try:
                    trace_data = json.loads(trace_content)
                    trace_context = json.dumps(trace_data, indent=2)[:4000]
                except json.JSONDecodeError:
                    trace_context = trace_content[:4000]
            else:
                # Plain text, markdown, etc.
                trace_context = trace_content[:4000]

            task = f"{task}\n\nBased on this agent trace:\n\n{trace_context}"
            console.print(f"Generating skill with {gen_model}...", style="dim")
            skill = await generate_skill(
                task=task,
                examples=examples,
                generator=agent.skill_gen,
                model=model,
            )
        # Improve existing skill
        elif from_skill:
            existing_skill = Skill.load(Path(from_skill))
            console.print(
                f"Improving [bold]{existing_skill.name}[/bold] with {gen_model}...",
                style="dim",
            )
            skill = await improve_skill(
                existing_skill,
                instructions=task,
                generator=agent.skill_gen,
                model=model,
            )
        else:
            console.print(f"Generating skill with {gen_model}...", style="dim")
            skill = await generate_skill(
                task=task,
                examples=examples,
                generator=agent.skill_gen,
                model=model,
            )
        if no_eval:
            _save_and_display(skill, output, config)
            return

        console.print("Generating test cases...", style="dim")
        test_cases = await generate_tests(task, generator=agent.test_gen, model=model)

        # Eval loop with refinement (on generation model)
        prev_success_rate = 0.0
        results = None
        attempts = max(1, config.max_refine_attempts)
        for attempt in range(attempts):
            console.print(f"Evaluating on {gen_model}... (attempt {attempt + 1})", style="dim")

            # Create run folder for logging (2 folders per attempt: baseline + with_skill)
            run_folder = None
            if log_runs and batch_folder:
                baseline_run_num = attempt * 2 + 1
                run_folder = create_run_folder(batch_folder, baseline_run_num)
                write_run_metadata(
                    run_folder,
                    RunMetadata(
                        model=gen_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=baseline_run_num,
                    ),
                )

            console.print("[dim]Starting evaluation run...[/dim]")

            results = await evaluate_skill(
                skill,
                test_cases=test_cases,
                evaluator=agent.evaluator,
                model=gen_model,
            )

            # Log run results (both baseline and with-skill for plot command)
            if log_runs and run_folder:
                # Log baseline result
                baseline_result = RunResult(
                    metadata=RunMetadata(
                        model=gen_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=baseline_run_num,
                    ),
                    stats=aggregate_conversation_stats(results.baseline_results),
                    passed=results.baseline_success_rate > 0.5,
                    assertions_passed=int(results.baseline_success_rate * len(test_cases)),
                    assertions_total=len(test_cases),
                    run_type="baseline",
                    skill_name=skill.name,
                )
                write_run_result(run_folder, baseline_result)
                run_results.append(baseline_result)

                # Log with-skill result (in a separate folder)
                with_skill_folder = create_run_folder(batch_folder, attempt * 2 + 2)
                with_skill_result = RunResult(
                    metadata=RunMetadata(
                        model=gen_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=attempt * 2 + 2,
                    ),
                    stats=aggregate_conversation_stats(results.with_skill_results),
                    passed=results.is_beneficial,
                    assertions_passed=int(results.with_skill_success_rate * len(test_cases)),
                    assertions_total=len(test_cases),
                    run_type="with_skill",
                    skill_name=skill.name,
                )
                write_run_metadata(with_skill_folder, with_skill_result.metadata)
                write_run_result(with_skill_folder, with_skill_result)
                run_results.append(with_skill_result)

            lift = results.skill_lift
            lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"

            if results.is_beneficial:
                console.print(
                    f"  {results.baseline_success_rate:.0%} -> "
                    f"{results.with_skill_success_rate:.0%} ({lift_str}) [green]OK[/green]"
                )
                break

            console.print(
                f"  {results.baseline_success_rate:.0%} -> "
                f"{results.with_skill_success_rate:.0%} ({lift_str}) not good enough"
            )

            if abs(results.with_skill_success_rate - prev_success_rate) < 0.05:
                console.print("  [yellow]Plateaued, stopping[/yellow]")
                break

            prev_success_rate = results.with_skill_success_rate

            if attempt < attempts - 1:
                console.print("Refining...", style="dim")
                failures = get_failure_descriptions(results)
                skill = await refine_skill(
                    skill,
                    failures,
                    generator=agent.skill_gen,
                    model=model,
                )

        # If eval_model specified, also eval on that model
        eval_results = None
        if eval_model:
            console.print(f"Evaluating on {eval_model}...", style="dim")

            # Create run folder for eval model
            run_folder = None
            if log_runs and batch_folder:
                run_number = attempts + 1
                run_folder = create_run_folder(batch_folder, run_number)
                write_run_metadata(
                    run_folder,
                    RunMetadata(
                        model=eval_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=run_number,
                    ),
                )

            eval_results = await evaluate_skill(
                skill,
                test_cases,
                evaluator=agent.evaluator,
                model=eval_model,
            )

            # Log eval run results (both baseline and with-skill)
            if log_runs and run_folder:
                # Log baseline result
                baseline_result = RunResult(
                    metadata=RunMetadata(
                        model=eval_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=run_number,
                    ),
                    stats=aggregate_conversation_stats(eval_results.baseline_results),
                    passed=eval_results.baseline_success_rate > 0.5,
                    assertions_passed=int(eval_results.baseline_success_rate * len(test_cases)),
                    assertions_total=len(test_cases),
                    run_type="baseline",
                    skill_name=skill.name,
                )
                write_run_result(run_folder, baseline_result)
                run_results.append(baseline_result)

                # Log with-skill result
                with_skill_folder = create_run_folder(batch_folder, run_number + 1)
                with_skill_result = RunResult(
                    metadata=RunMetadata(
                        model=eval_model,
                        task=task,
                        batch_id=batch_id or "",
                        run_number=run_number + 1,
                    ),
                    stats=aggregate_conversation_stats(eval_results.with_skill_results),
                    passed=eval_results.is_beneficial,
                    assertions_passed=int(eval_results.with_skill_success_rate * len(test_cases)),
                    assertions_total=len(test_cases),
                    run_type="with_skill",
                    skill_name=skill.name,
                )
                write_run_metadata(with_skill_folder, with_skill_result.metadata)
                write_run_result(with_skill_folder, with_skill_result)
                run_results.append(with_skill_result)

            lift = eval_results.skill_lift
            lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
            console.print(
                f"  {eval_results.baseline_success_rate:.0%} -> "
                f"{eval_results.with_skill_success_rate:.0%} ({lift_str})"
            )

        # Write batch summary
        if log_runs and batch_folder and batch_id:
            summary = BatchSummary(
                batch_id=batch_id,
                model=gen_model,
                task=task,
                total_runs=len(run_results),
                passed_runs=sum(1 for r in run_results if r.passed),
                results=run_results,
            )
            write_batch_summary(batch_folder, summary)

    if not no_eval and skill is not None:
        if results:
            skill.metadata.test_pass_rate = results.with_skill_success_rate
        else:
            console.print(
                "[yellow]No evaluation results available; skipping report output.[/yellow]"
            )

        _save_and_display(skill, output, config, results, eval_results, gen_model, eval_model)





def _save_and_display(
    skill: Skill,
    output: str | None,
    config: Config,
    results=None,
    eval_results=None,
    gen_model: str | None = None,
    eval_model: str | None = None,
):
    """Save skill and display summary."""
    if output:
        output_path = Path(output)
    else:
        output_path = config.skills_dir / skill.name

    skill.save(output_path)

    console.print("[dim]Rendering report output...[/dim]")

    console.print()
    console.print(f"  [bold]{skill.name}[/bold]")
    console.print(f"  {skill.description}")
    console.print()

    skill_tokens = len(skill.body.split()) * 1.3
    console.print(f"  SKILL.md              ~{int(skill_tokens)} tokens")
    for name in skill.references:
        ref_tokens = len(skill.references[name].split()) * 1.3
        console.print(f"  references/{name}  ~{int(ref_tokens)} tokens")
    for name in skill.scripts:
        console.print(f"  scripts/{name}     (exec only)")

    # Show results with horizontal bars
    if results and eval_results:
        # Multiple models - show each with bars
        console.print()
        for model_name, r in [(gen_model or "gen", results), (eval_model or "eval", eval_results)]:
            console.print(f"  [bold]{model_name}[/bold]")
            baseline_bar = _render_bar(r.baseline_success_rate)
            with_skill_bar = _render_bar(r.with_skill_success_rate)
            lift = r.skill_lift
            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
            lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

            console.print(f"    baseline   {baseline_bar}  {r.baseline_success_rate:>5.0%}")
            console.print(
                f"    with skill {with_skill_bar}  {r.with_skill_success_rate:>5.0%}  "
                f"[{lift_style}]({lift_str})[/{lift_style}]"
            )
            console.print()

    elif results:
        # Single model
        console.print()
        baseline_bar = _render_bar(results.baseline_success_rate)
        with_skill_bar = _render_bar(results.with_skill_success_rate)
        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
        lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

        console.print(f"  baseline   {baseline_bar}  {results.baseline_success_rate:>5.0%}")
        console.print(
            f"  with skill {with_skill_bar}  {results.with_skill_success_rate:>5.0%}  "
            f"[{lift_style}]({lift_str})[/{lift_style}]"
        )

        if results.baseline_total_tokens > 0:
            savings = results.token_savings
            savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
            savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"
            console.print()
            console.print(
                f"  tokens: {results.baseline_total_tokens} → {results.with_skill_total_tokens}  "
                f"[{savings_style}]({savings_str})[/{savings_style}]"
            )

    console.print()
    console.print(f"Saved to {output_path}")


@main.command("eval")
@click.argument("skill_path", type=click.Path(exists=True))
@click.option("-t", "--tests", type=click.Path(exists=True), help="Test cases JSON file")
@click.option(
    "-m", "--model", "models", multiple=True, help="Model(s) to evaluate (repeatable)"
)
@click.option("--runs", "num_runs", type=int, default=1, help="Number of runs per model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "generic"]),
    help="API provider (auto-detected as 'generic' when --base-url is provided)",
)
@click.option(
    "--base-url", help="Custom API endpoint for local models (e.g., http://localhost:8080/v1)"
)
@click.option("--no-baseline", is_flag=True, help="Skip baseline comparison")
@click.option("-v", "--verbose", is_flag=True, help="Show per-test results")
@click.option("--log-runs/--no-log-runs", default=True, help="Log run data (default: enabled)")
@click.option("--runs-dir", type=click.Path(), help="Directory for run logs")
def eval_cmd(
    skill_path: str,
    tests: str | None,
    models: tuple[str, ...],
    num_runs: int,
    provider: str | None,
    base_url: str | None,
    no_baseline: bool,
    verbose: bool,
    log_runs: bool,
    runs_dir: str | None,
):
    """Evaluate a skill (compares with vs without).

    Examples:

        upskill eval ./skills/my-skill/

        upskill eval ./skills/my-skill/ --tests ./tests.json -v

        upskill eval ./skills/my-skill/ -m haiku

        upskill eval ./skills/my-skill/ -m haiku -m sonnet

        upskill eval ./skills/my-skill/ -m haiku --runs 5

        # Local model with llama.cpp server:

        upskill eval ./skills/my-skill/ -m my-model \\
            --base-url http://localhost:8080/v1

        # Local model with Ollama:

        upskill eval ./skills/my-skill/ -m llama3.2:latest \\
            --base-url http://localhost:11434/v1

        upskill eval ./skills/my-skill/ --no-log-runs
    """
    asyncio.run(
        _eval_async(
            skill_path,
            tests,
            list(models) if models else None,
            num_runs,
            provider,
            base_url,
            no_baseline,
            verbose,
            log_runs,
            runs_dir,
        )
    )


async def _eval_async(
    skill_path: str,
    tests: str | None,
    models: list[str] | None,
    num_runs: int,
    provider: str | None,
    base_url: str | None,
    no_baseline: bool,
    verbose: bool,
    log_runs: bool,
    runs_dir: str | None,
):
    """Async implementation of eval command."""
    from upskill.evaluate import run_test

    config = Config.load()
    skill_dir = Path(skill_path)

    try:
        skill = Skill.load(skill_dir)
    except FileNotFoundError:
        console.print(f"[red]No SKILL.md found in {skill_dir}[/red]")
        sys.exit(1)

    # Use default model if none specified
    if not models:
        models = [config.effective_eval_model]

    is_benchmark_mode = len(models) > 1 or num_runs > 1

    async with _fast_agent_context() as agent:
        # Load test cases
        test_cases: list[TestCase] = []
        if tests:
            with open(tests, encoding="utf-8") as f:
                data = json.load(f)
            if "cases" in data:
                test_cases = [TestCase(**tc) for tc in data["cases"]]
            else:
                test_cases = [TestCase(**tc) for tc in data]
        elif skill.tests:
            test_cases = skill.tests
        else:
            console.print("Generating test cases from skill...", style="dim")
            test_cases = await generate_tests(
                skill.description,
                generator=agent.test_gen,
                model=models[0],
            )

        # Setup run logging
        batch_id = None
        batch_folder = None
        if log_runs:
            runs_path = Path(runs_dir) if runs_dir else config.runs_dir
            batch_id, batch_folder = create_batch_folder(runs_path)
            console.print(f"Logging to: {batch_folder}", style="dim")

        provider_info = ""
        if provider:
            provider_info += f" via {provider}"
        if base_url:
            provider_info += f" @ {base_url}"

        if is_benchmark_mode:
            # Benchmark mode: multiple models and/or runs
            console.print(
                f"\nEvaluating [bold]{skill.name}[/bold] across {len(models)} model(s)"
            )
            console.print(
                f"  {len(test_cases)} test case(s), "
                f"{num_runs} run(s) per model{provider_info}\n"
            )

            model_results: dict[str, list[RunResult]] = {m: [] for m in models}
            all_run_results: list[RunResult] = []

            for model in models:
                console.print(f"[bold]{model}[/bold]")

                for run_num in range(1, num_runs + 1):
                    run_folder = None
                    if log_runs and batch_folder:
                        run_folder = create_run_folder(
                            batch_folder, len(all_run_results) + 1
                        )

                    # Run each test case
                    total_assertions_passed = 0
                    total_assertions = 0
                    all_passed = True
                    run_test_results: list[TestResult] = []

                    for tc_idx, tc in enumerate(test_cases, 1):
                        if verbose:
                            console.print(
                                f"  Running test {tc_idx}/{len(test_cases)}...",
                                style="dim",
                            )

                        try:
                            result = await run_test(
                                tc,
                                evaluator=agent.evaluator,
                                skill=skill,
                            )
                        except Exception as e:
                            console.print(f"  [red]Test error: {e}[/red]")
                            result = TestResult(test_case=tc, success=False, error=str(e))

                        # Extract assertion counts
                        if result.validation_result:
                            total_assertions_passed += result.validation_result.assertions_passed
                            total_assertions += result.validation_result.assertions_total
                            if verbose and result.validation_result.error_message:
                                console.print(
                                    f"    Validation: {result.validation_result.error_message}",
                                    style="dim",
                                )
                        elif result.error:
                            if verbose:
                                console.print(f"    Error: {result.error}", style="dim")
                            total_assertions += 1
                        else:
                            total_assertions += 1
                            if result.success:
                                total_assertions_passed += 1

                        run_test_results.append(result)
                        if not result.success:
                            all_passed = False

                    aggregated_stats = aggregate_conversation_stats(run_test_results)

                    run_result = RunResult(
                        metadata=RunMetadata(
                            model=model,
                            task=skill.description,
                            batch_id=batch_id or "",
                            run_number=run_num,
                        ),
                        stats=aggregated_stats,
                        passed=all_passed,
                        assertions_passed=total_assertions_passed,
                        assertions_total=total_assertions,
                        run_type="with_skill",
                        skill_name=skill.name,
                    )

                    if run_folder:
                        write_run_metadata(run_folder, run_result.metadata)
                        write_run_result(run_folder, run_result)

                    model_results[model].append(run_result)
                    all_run_results.append(run_result)

                    # Display progress
                    status = "[green]PASS[/green]" if all_passed else "[red]FAIL[/red]"
                    if verbose:
                        console.print(
                            f"  Run {run_num}: {status} "
                            f"({total_assertions_passed}/{total_assertions} assertions passed)"
                        )

                console.print()

            # Summary report
            console.print("\n[bold]Summary[/bold]\n")

            for model, results in model_results.items():
                total_runs = len(results)
                passed_runs = sum(1 for r in results if r.passed)
                avg_tokens = (
                    sum(r.stats.total_tokens for r in results) / total_runs if total_runs else 0
                )
                avg_turns = sum(r.stats.turns for r in results) / total_runs if total_runs else 0

                pass_rate = passed_runs / total_runs if total_runs else 0
                pass_rate_str = f"{pass_rate:.0%}"
                if pass_rate > 0.5:
                    pass_rate_style = "green"
                elif pass_rate > 0:
                    pass_rate_style = "yellow"
                else:
                    pass_rate_style = "red"

                console.print(f"[bold]{model}[/bold]")
                console.print(
                    "  Runs: "
                    f"{total_runs} | Passed: {passed_runs} ([{pass_rate_style}]"
                    f"{pass_rate_str}[/{pass_rate_style}])"
                )
                console.print(f"  Avg tokens: {avg_tokens:.0f} | Avg turns: {avg_turns:.1f}")
                console.print()

            # Write batch summary
            if log_runs and batch_folder and batch_id:
                summary = BatchSummary(
                    batch_id=batch_id,
                    model=", ".join(models),
                    task=skill.description,
                    total_runs=len(all_run_results),
                    passed_runs=sum(1 for r in all_run_results if r.passed),
                    results=all_run_results,
                )
                write_batch_summary(batch_folder, summary)

        else:
            # Simple eval mode: single model, single run
            model = models[0]
            console.print(f"Running {len(test_cases)} test cases{provider_info}...", style="dim")

            results = await evaluate_skill(
                skill,
                test_cases,
                evaluator=agent.evaluator,
                model=model,
                run_baseline=not no_baseline,
            )

            # Log results (both baseline and with-skill)
            run_results: list[RunResult] = []
            if log_runs and batch_folder:
                # Log baseline result
                if not no_baseline:
                    baseline_folder = create_run_folder(batch_folder, 1)
                    baseline_result = RunResult(
                        metadata=RunMetadata(
                            model=model,
                            task=skill.description,
                            batch_id=batch_id or "",
                            run_number=1,
                        ),
                        stats=aggregate_conversation_stats(results.baseline_results),
                        passed=results.baseline_success_rate > 0.5,
                        assertions_passed=int(results.baseline_success_rate * len(test_cases)),
                        assertions_total=len(test_cases),
                        run_type="baseline",
                        skill_name=skill.name,
                    )
                    write_run_metadata(baseline_folder, baseline_result.metadata)
                    write_run_result(baseline_folder, baseline_result)
                    run_results.append(baseline_result)

                # Log with-skill result
                with_skill_folder = create_run_folder(batch_folder, 2 if not no_baseline else 1)
                with_skill_result = RunResult(
                    metadata=RunMetadata(
                        model=model,
                        task=skill.description,
                        batch_id=batch_id or "",
                        run_number=2 if not no_baseline else 1,
                    ),
                    stats=aggregate_conversation_stats(results.with_skill_results),
                    passed=results.is_beneficial
                    if not no_baseline
                    else results.with_skill_success_rate > 0.5,
                    assertions_passed=int(results.with_skill_success_rate * len(test_cases)),
                    assertions_total=len(test_cases),
                    run_type="with_skill",
                    skill_name=skill.name,
                )
                write_run_metadata(with_skill_folder, with_skill_result.metadata)
                write_run_result(with_skill_folder, with_skill_result)
                run_results.append(with_skill_result)

                # Write batch summary
                summary = BatchSummary(
                    batch_id=batch_id or "",
                    model=model,
                    task=skill.description,
                    total_runs=len(run_results),
                    passed_runs=sum(1 for r in run_results if r.passed),
                    results=run_results,
                )
                write_batch_summary(batch_folder, summary)

            if verbose and not no_baseline:
                console.print()
                for i, (with_r, base_r) in enumerate(
                    zip(results.with_skill_results, results.baseline_results), 1
                ):
                    base_icon = "[green]OK[/green]" if base_r.success else "[red]FAIL[/red]"
                    skill_icon = "[green]OK[/green]" if with_r.success else "[red]FAIL[/red]"
                    input_preview = with_r.test_case.input[:40]
                    console.print(f"  {i}. {input_preview}  {base_icon} base  {skill_icon} skill")
                console.print()

            # Display results with horizontal bars
            console.print()
            if not no_baseline:
                baseline_rate = results.baseline_success_rate
                with_skill_rate = results.with_skill_success_rate
                lift = results.skill_lift

                baseline_bar = _render_bar(baseline_rate)
                with_skill_bar = _render_bar(with_skill_rate)

                lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
                lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

                console.print(f"  baseline   {baseline_bar}  {baseline_rate:>5.0%}")
                console.print(
                    f"  with skill {with_skill_bar}  {with_skill_rate:>5.0%}  "
                    f"[{lift_style}]({lift_str})[/{lift_style}]"
                )

                # Token comparison
                if results.baseline_total_tokens > 0:
                    savings = results.token_savings
                    savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
                    savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"
                    console.print()
                    token_line = (
                        f"  tokens: {results.baseline_total_tokens} → "
                        f"{results.with_skill_total_tokens}  "
                        f"[{savings_style}]({savings_str})[/{savings_style}]"
                    )
                    console.print(token_line)
            else:
                with_skill_rate = results.with_skill_success_rate
                with_skill_bar = _render_bar(with_skill_rate)
                console.print(f"  with skill {with_skill_bar}  {with_skill_rate:>5.0%}")
                console.print(f"  tokens: {results.with_skill_total_tokens}")

            if not no_baseline:
                if results.is_beneficial:
                    console.print("\n[green]Recommendation: keep skill[/green]")
                else:
                    console.print("\n[yellow]Recommendation: skill may not be beneficial[/yellow]")


@main.command("list")
@click.option("-d", "--dir", "skills_dir", type=click.Path(), help="Skills directory to list")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed skill structure")
def list_cmd(skills_dir: str | None, verbose: bool):
    """List generated skills."""
    config = Config.load()
    if skills_dir:
        path = Path(skills_dir)
    else:
        path = config.skills_dir

    if not path.exists():
        console.print(f"No skills directory found at {path}")
        return

    skills = [d for d in path.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

    if not skills:
        console.print(f"No skills found in {path}")
        return

    # Build tree view
    tree = Tree(f"[bold]Skills in {path}[/bold]")
    total_tokens = 0

    for skill_dir in sorted(skills):
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text()
        lines = content.split("\n")
        name = skill_dir.name

        # Estimate tokens
        skill_tokens = int(len(content.split()) * 1.3)
        total_tokens += skill_tokens

        skill_branch = tree.add(f"[bold]{name}[/bold]")

        if verbose:
            # Add SKILL.md with token count
            skill_branch.add(f"SKILL.md [dim](~{skill_tokens} tokens)[/dim]")

            # Check for references directory
            refs_dir = skill_dir / "references"
            if refs_dir.exists() and refs_dir.is_dir():
                refs_branch = skill_branch.add("references/")
                for ref_file in sorted(refs_dir.iterdir()):
                    if ref_file.is_file():
                        ref_tokens = int(len(ref_file.read_text().split()) * 1.3)
                        total_tokens += ref_tokens
                        refs_branch.add(f"{ref_file.name} [dim](~{ref_tokens} tokens)[/dim]")

            # Check for scripts directory
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists() and scripts_dir.is_dir():
                scripts_branch = skill_branch.add("scripts/")
                for script_file in sorted(scripts_dir.iterdir()):
                    if script_file.is_file():
                        scripts_branch.add(f"{script_file.name}")

            # Check for tests.json
            tests_file = skill_dir / "tests.json"
            if tests_file.exists():
                with open(tests_file, encoding="utf-8") as f:
                    tests_data = json.load(f)
                num_cases = len(tests_data.get("cases", tests_data))
                skill_branch.add(f"tests.json [dim]({num_cases} cases)[/dim]")
        else:
            # Simple view: just show description
            description = lines[2] if len(lines) > 2 else ""
            if description:
                skill_branch.add(f"[dim]{description[:60]}...[/dim]")

    console.print()
    console.print(tree)
    console.print()
    console.print(f"[dim]{len(skills)} skills, ~{total_tokens:,} total tokens[/dim]")


@main.command("benchmark")
@click.argument("skill_path", type=click.Path(exists=True))
@click.option("-m", "--model", "models", multiple=True, required=True, help="Model to benchmark")
@click.option("--runs", "num_runs", type=int, default=3, help="Runs per model (default: 3)")
@click.option("-t", "--tests", type=click.Path(exists=True), help="Test cases JSON file")
@click.option("-o", "--output", type=click.Path(), help="Output directory for results")
@click.option("-v", "--verbose", is_flag=True, help="Show per-run details")
def benchmark_cmd(
    skill_path: str,
    models: tuple[str, ...],
    num_runs: int,
    tests: str | None,
    output: str | None,
    verbose: bool,
):
    """Benchmark a skill across multiple models.

    Runs the skill's test cases multiple times per model and reports
    pass rates and assertion statistics. MCP servers from fastagent.config.yaml
    are automatically enabled.

    Examples:

        upskill benchmark ./skills/hf-eval-extraction/ -m haiku -m sonnet

        upskill benchmark ./skills/my-skill/ -m gpt-4o -m claude-sonnet --runs 5

        upskill benchmark ./skills/my-skill/ -m haiku -t ./custom_tests.json -v
    """
    asyncio.run(
        _benchmark_async(
            skill_path, list(models), num_runs, tests, output, verbose
        )
    )


async def _benchmark_async(
    skill_path: str,
    models: list[str],
    num_runs: int,
    tests_path: str | None,
    output_dir: str | None,
    verbose: bool,
):
    """Async implementation of benchmark command."""
    from upskill.evaluate import run_test

    config = Config.load()
    skill = Skill.load(Path(skill_path))

    async with _fast_agent_context() as agent:
        # Load test cases
        if tests_path:
            with open(tests_path, encoding="utf-8") as f:
                data = json.load(f)
            if "cases" in data:
                test_cases = [TestCase(**tc) for tc in data["cases"]]
            else:
                test_cases = [TestCase(**tc) for tc in data]
        elif skill.tests:
            test_cases = skill.tests
        else:
            console.print("Generating test cases from skill...", style="dim")
            test_cases = await generate_tests(
                skill.description,
                generator=agent.test_gen,
            )

        # Setup output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = config.runs_dir

        batch_id, batch_folder = create_batch_folder(out_path)
        console.print(f"Results will be saved to: {batch_folder}", style="dim")

        # Track results per model
        model_results: dict[str, list[RunResult]] = {m: [] for m in models}
        all_run_results: list[RunResult] = []

        console.print(f"\nBenchmarking [bold]{skill.name}[/bold] across {len(models)} model(s)")
        console.print(f"  {len(test_cases)} test case(s), {num_runs} run(s) per model\n")

        for model in models:
            console.print(f"[bold]{model}[/bold]")

            for run_num in range(1, num_runs + 1):
                run_folder = create_run_folder(batch_folder, len(all_run_results) + 1)

                # Run each test case
                total_assertions_passed = 0
                total_assertions = 0
                all_passed = True
                run_results: list[TestResult] = []

                for tc_idx, tc in enumerate(test_cases, 1):
                    if verbose:
                        console.print(f"  Running test {tc_idx}/{len(test_cases)}...", style="dim")

                    try:
                        result = await run_test(
                            tc,
                            evaluator=agent.evaluator,
                            skill=skill,
                        )
                    except Exception as e:
                        console.print(f"  [red]Test error: {e}[/red]")
                        result = TestResult(test_case=tc, success=False, error=str(e))

                    # Extract assertion counts from validation result
                    if result.validation_result:
                        total_assertions_passed += result.validation_result.assertions_passed
                        total_assertions += result.validation_result.assertions_total
                        if verbose and result.validation_result.error_message:
                            console.print(
                                f"    Validation: {result.validation_result.error_message}",
                                style="dim",
                            )
                    elif result.error:
                        if verbose:
                            console.print(f"    Error: {result.error}", style="dim")
                        # Legacy: count as 1 assertion (failed)
                        total_assertions += 1
                    else:
                        # Legacy: count as 1 assertion
                        total_assertions += 1
                        if result.success:
                            total_assertions_passed += 1

                    run_results.append(result)

                    if not result.success:
                        all_passed = False

                aggregated_stats = aggregate_conversation_stats(run_results)

                # Create run result
                run_result = RunResult(
                    metadata=RunMetadata(
                        model=model,
                        task=skill.description,
                        batch_id=batch_id,
                        run_number=run_num,
                    ),
                    stats=aggregated_stats,
                    passed=all_passed,
                    assertions_passed=total_assertions_passed,
                    assertions_total=total_assertions,
                    run_type="with_skill",
                    skill_name=skill.name,
                )

                write_run_metadata(run_folder, run_result.metadata)
                write_run_result(run_folder, run_result)
                model_results[model].append(run_result)
                all_run_results.append(run_result)

                # Display progress
                status = "[green]PASS[/green]" if all_passed else "[red]FAIL[/red]"
                if verbose:
                    console.print(
                        f"  Run {run_num}: {status} "
                        f"({total_assertions_passed}/{total_assertions} assertions passed)"
                    )

        console.print("\n[bold]Summary[/bold]\n")

        for model, results in model_results.items():
            total_runs = len(results)
            passed_runs = sum(1 for r in results if r.passed)
            avg_tokens = (
                sum(r.stats.total_tokens for r in results) / total_runs if total_runs else 0
            )
            avg_turns = (
                sum(r.stats.turns for r in results) / total_runs if total_runs else 0
            )

            pass_rate = passed_runs / total_runs if total_runs else 0
            pass_rate_str = f"{pass_rate:.0%}"
            pass_rate_style = "green" if pass_rate > 0.5 else "yellow" if pass_rate > 0 else "red"

            console.print(f"[bold]{model}[/bold]")
            console.print(
                "  Runs: "
                f"{total_runs} | Passed: {passed_runs} ([{pass_rate_style}]"
                f"{pass_rate_str}[/{pass_rate_style}])"
            )
            console.print(f"  Avg tokens: {avg_tokens:.0f} | Avg turns: {avg_turns:.1f}")
            console.print()

        summary = BatchSummary(
            batch_id=batch_id,
            model=", ".join(models),
            task=skill.description,
            total_runs=len(all_run_results),
            passed_runs=sum(1 for r in all_run_results if r.passed),
            results=all_run_results,
        )
        write_batch_summary(batch_folder, summary)

@main.command("runs")
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option("-m", "--model", "models", multiple=True, help="Filter by model(s)")
@click.option("--csv", "csv_output", type=click.Path(), help="Export to CSV file")
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to display",
)
def runs_cmd(
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    csv_output: str | None,
    metric: str,
):
    """Show evaluation history with visual comparison.

    By default shows a plot of baseline vs with-skill performance.
    Use --csv to export results to a CSV file.

    Examples:

        upskill runs

        upskill runs -s my-skill -m haiku -m sonnet

        upskill runs --csv results.csv

        upskill runs --metric tokens
    """
    config = Config.load()
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir

    if not runs_path.exists():
        console.print(f"[red]No runs directory found at {runs_path}[/red]")
        sys.exit(1)

    # If --csv is specified, export to CSV
    if csv_output:
        try:
            output_path = summarize_runs_to_csv(runs_path, Path(csv_output))
            console.print(f"Summary written to {output_path}")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        return

    # Otherwise, show plot
    all_results = _load_eval_results(runs_path)

    if not all_results:
        console.print("[yellow]No eval results with baseline comparisons found.[/yellow]")
        console.print("Run 'upskill eval <skill>' to generate comparable results.")
        console.print("\n[dim]--csv results.csv to export raw data[/dim]")
        sys.exit(0)

    # Filter by skills and models
    if skills:
        all_results = [r for r in all_results if r["skill_name"] in skills]
    if models:
        all_results = [r for r in all_results if r["model"] in models]

    if not all_results:
        console.print("[yellow]No results match the specified filters.[/yellow]")
        sys.exit(0)

    # Aggregate by model and skill (take most recent / highest)
    aggregated: dict[tuple[str, str], dict] = {}
    for r in all_results:
        key = (r["model"], r["skill_name"])
        if key not in aggregated or r["with_skill_rate"] > aggregated[key]["with_skill_rate"]:
            aggregated[key] = r

    results_list = list(aggregated.values())

    # Determine display mode
    unique_skills = set(r["skill_name"] for r in results_list)
    unique_models = set(r["model"] for r in results_list)

    console.print()

    if len(unique_skills) == 1 and len(unique_models) >= 1:
        # Single skill, multiple models - use Panel
        skill_name = list(unique_skills)[0]

        # Build content for panel
        content_lines = []
        for r in sorted(results_list, key=lambda x: x["model"]):
            content_lines.append(_format_comparison_bars(r, metric))

        panel_content = "\n".join(content_lines)
        panel_title = f"Evaluation History: {skill_name}"
        console.print(Panel(panel_content, title=panel_title, border_style="blue"))

    elif len(unique_models) == 1 and len(unique_skills) >= 1:
        # Single model, multiple skills - use Panel
        model_name = list(unique_models)[0]

        content_lines = []
        for r in sorted(results_list, key=lambda x: x["skill_name"]):
            content_lines.append(_format_comparison_bars(r, metric, label_field="skill_name"))

        panel_content = "\n".join(content_lines)
        panel_title = f"Evaluation History: {model_name}"
        console.print(Panel(panel_content, title=panel_title, border_style="blue"))

    else:
        # Multiple skills and models - matrix view with Panel
        _print_matrix_view(results_list, metric)

    console.print("\n[dim]--csv results.csv to export[/dim]")


@main.command("plot", hidden=True)
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option("-m", "--model", "models", multiple=True, help="Filter by model(s)")
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to plot",
)
@click.pass_context
def plot_cmd(
    ctx: click.Context,
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    metric: str,
):
    """[Deprecated] Use 'upskill runs' instead."""
    console.print("[yellow]Note: 'plot' is deprecated. Use 'upskill runs' instead.[/yellow]\n")
    ctx.invoke(
        runs_cmd,
        runs_dir=runs_dir,
        skills=skills,
        models=models,
        csv_output=None,
        metric=metric,
    )


def _format_comparison_bars(
    result: EvalPlotResult,
    metric: str,
    label_field: str = "model",
) -> str:
    """Format baseline vs with-skill comparison bars for a single result as string."""
    label = result[label_field]
    has_baseline = result["has_baseline"]
    lines = [f"[bold]{label}[/bold]"]

    if metric == "success":
        with_skill_val = result["with_skill_rate"]
        with_skill_bar = _render_bar(with_skill_val)

        if has_baseline:
            baseline_val = result["baseline_rate"]
            lift = with_skill_val - baseline_val
            baseline_bar = _render_bar(baseline_val)

            lines.append(f"  baseline   {baseline_bar}  {baseline_val:>5.0%}")

            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
            lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>5.0%}  "
                f"[{lift_style}]({lift_str})[/{lift_style}]"
            )
        else:
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>5.0%}  [dim](no baseline)[/dim]"
            )
    else:  # tokens
        with_skill_val = result["with_skill_tokens"]

        if has_baseline:
            baseline_val = result["baseline_tokens"]
            max_val = max(baseline_val, with_skill_val, 1)

            baseline_bar = _render_bar(baseline_val / max_val)
            with_skill_bar = _render_bar(with_skill_val / max_val)

            savings = (baseline_val - with_skill_val) / baseline_val if baseline_val else 0
            savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
            savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"

            lines.append(f"  baseline   {baseline_bar}  {baseline_val:>6}")
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>6}  "
                f"[{savings_style}]({savings_str})[/{savings_style}]"
            )
        else:
            with_skill_bar = _render_bar(1.0 if with_skill_val > 0 else 0)
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>6}  [dim](no baseline)[/dim]"
            )

    return "\n".join(lines)


def _print_comparison_bars(
    result: EvalPlotResult,
    metric: str,
    label_field: str = "model",
) -> None:
    """Print baseline vs with-skill comparison bars for a single result."""
    console.print(_format_comparison_bars(result, metric, label_field))
    console.print()


def _print_matrix_view(results: list[EvalPlotResult], metric: str) -> None:
    """Print a matrix view for multiple skills and models."""
    # Get unique skills and models
    skills = sorted(set(r["skill_name"] for r in results))
    models = sorted(set(r["model"] for r in results))

    # Build lookup
    lookup = {(r["model"], r["skill_name"]): r for r in results}

    # Create table
    table = Table(show_header=True, title="Skill Performance Matrix")
    table.add_column("skill", style="bold")

    for model in models:
        table.add_column(model, justify="center")

    for skill in skills:
        row = [skill]
        for model in models:
            r = lookup.get((model, skill))
            if r:
                has_baseline = r["has_baseline"]
                if metric == "success":
                    with_skill = r["with_skill_rate"]
                    if has_baseline:
                        baseline = r["baseline_rate"]
                        lift = with_skill - baseline
                        lift_style = "green" if lift > 0 else "red" if lift < 0 else ""
                        cell = f"{baseline:.0%}→{with_skill:.0%}"
                        if lift_style:
                            cell = f"[{lift_style}]{cell}[/{lift_style}]"
                    else:
                        cell = f"[dim]-[/dim]→{with_skill:.0%}"
                else:
                    with_skill = r["with_skill_tokens"]
                    if has_baseline:
                        baseline = r["baseline_tokens"]
                        savings = (baseline - with_skill) / baseline if baseline else 0
                        savings_style = "green" if savings > 0 else "red" if savings < 0 else ""
                        cell = f"{baseline}→{with_skill}"
                        if savings_style:
                            cell = f"[{savings_style}]{cell}[/{savings_style}]"
                    else:
                        cell = f"[dim]-[/dim]→{with_skill}"
                row.append(cell)
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()

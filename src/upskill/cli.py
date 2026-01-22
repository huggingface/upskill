"""CLI interface for upskill."""
from __future__ import annotations

import asyncio
import json
import sys
from importlib import resources
from pathlib import Path

import click
from dotenv import load_dotenv
from fast_agent import FastAgent
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from upskill.config import Config
from upskill.evaluate import evaluate_skill, get_failure_descriptions
from upskill.generate import generate_skill, generate_tests, improve_skill, refine_skill
from upskill.logging import (
    aggregate_conversation_stats,
    create_batch_folder,
    create_run_folder,
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
    "from_path",
    type=click.Path(exists=True),
    help="Improve from existing skill dir or agent trace file (auto-detected)",
)
@click.option(
    "-m",
    "--model",
    help="Model for generation (e.g., 'sonnet', 'anthropic.claude-sonnet-4-20250514')",
)
@click.option("-o", "--output", type=click.Path(), help="Output directory for skill")
@click.option("--no-eval", is_flag=True, help="Skip eval and refinement")
@click.option("--eval-model", help="Model to evaluate skill on (different from generation model)")
@click.option(
    "--eval-provider",
    type=click.Choice(["anthropic", "openai", "generic"]),
    help=(
        "API provider for eval model (auto-detected as 'generic' when "
        "--eval-base-url is provided)"
    ),
)
@click.option(
    "--eval-base-url", help="Custom API endpoint for eval model (e.g., http://localhost:11434/v1)"
)
@click.option("--runs-dir", type=click.Path(), help="Directory for run logs (default: ./runs)")
@click.option("--log-runs/--no-log-runs", default=True, help="Log run data (default: enabled)")
def generate(
    task: str,
    example: tuple[str, ...],
    tool: str | None,  # noqa: ARG001
    from_path: str | None,
    model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    eval_provider: str | None,
    eval_base_url: str | None,
    runs_dir: str | None,
    log_runs: bool,
):
    """Generate a skill from a task description, or improve an existing skill.

    The --from option auto-detects whether the path is a skill directory
    (contains SKILL.md) or an agent trace file.

    Examples:

        upskill generate "parse JSON Schema files"

        upskill generate "write git commits" --model sonnet

        upskill generate "handle API errors" --eval-model haiku

        upskill generate "validate forms" -o ./my-skills/validation

        # Improve an existing skill (auto-detected as directory):

        upskill generate "add more error handling examples" --from ./skills/api-errors/

        upskill generate "make it more concise" -f ./skills/my-skill/ -o ./skills/my-skill-v2/

        # Generate from agent trace (auto-detected as file):

        upskill generate "document the pattern" --from ./trace.json

        # Evaluate on a local model (Ollama):

        upskill generate "parse YAML" --eval-model llama3.2:latest \\
            --eval-base-url http://localhost:11434/v1

        upskill generate "document code" --no-log-runs
    """
    # Auto-detect whether from_path is a skill directory or trace file
    from_skill = None
    from_trace = None
    if from_path:
        path = Path(from_path)
        if path.is_dir() and (path / "SKILL.md").exists():
            from_skill = from_path
        elif path.is_file():
            from_trace = from_path
        elif path.is_dir():
            console.print(f"[red]Directory {path} does not contain SKILL.md[/red]")
            sys.exit(1)

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
            eval_provider,
            eval_base_url,
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
    eval_provider: str | None,
    eval_base_url: str | None,
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

    fast = FastAgent(
        "upskill",
        ignore_unknown_args=True,
    )

    @fast.agent()
    async def empty():
        pass

    # load agents from card files
    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        fast.load_agents(cards_path)

    async with fast.run() as agent:

        # Improve existing skill, generate from trace, or generate new
        if from_skill:
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
        elif from_trace:
            console.print(f"Generating skill from trace with {gen_model}...", style="dim")
            # Load trace file and extract relevant context
            with open(from_trace, encoding="utf-8") as f:
                trace_data = json.load(f)
            # Include trace content as an example for generation
            trace_context = json.dumps(trace_data, indent=2)[:5000]  # Limit size
            skill = await generate_skill(
                task=task,
                examples=[f"Agent trace:\n{trace_context}"] if not examples else examples,
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
        for attempt in range(config.max_refine_attempts):
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

            if attempt < config.max_refine_attempts - 1:
                console.print("Refining...", style="dim")
                failures = get_failure_descriptions(results)
                skill = await refine_skill(skill, failures, model=model, config=config)

        # If eval_model specified, also eval on that model
        eval_results = None
        if eval_model:
            provider_info = ""
            if eval_provider:
                provider_info += f" via {eval_provider}"
            if eval_base_url:
                provider_info += f" @ {eval_base_url}"
            console.print(f"Evaluating on {eval_model}{provider_info}...", style="dim")

            # Create run folder for eval model
            run_folder = None
            if log_runs and batch_folder:
                run_number = config.max_refine_attempts + 1
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

        if results:
            skill.metadata.test_pass_rate = results.with_skill_success_rate

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
    "-m",
    "--model",
    "models",
    multiple=True,
    help="Model(s) to evaluate against (repeatable for multi-model benchmarking)",
)
@click.option(
    "--runs", "num_runs", type=int, default=1, help="Number of runs per model (default: 1)"
)
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
    """Evaluate a skill against test cases.

    Supports single-model evaluation with baseline comparison, or multi-model
    benchmarking with multiple runs per model.

    Examples:

        upskill eval ./skills/my-skill/

        upskill eval ./skills/my-skill/ --tests ./tests.json -v

        upskill eval ./skills/my-skill/ -m haiku

        # Multiple models (benchmark mode):

        upskill eval ./skills/my-skill/ -m haiku -m sonnet

        # Multiple runs per model:

        upskill eval ./skills/my-skill/ -m haiku -m sonnet --runs 5

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
    """Async implementation of eval command.

    Supports two modes:
    - Single-model eval: standard baseline vs with-skill comparison
    - Benchmark mode: multiple models and/or multiple runs per model
    """
    from upskill.evaluate import run_test

    config = Config.load()
    skill_dir = Path(skill_path)

    try:
        skill = Skill.load(skill_dir)
    except FileNotFoundError:
        console.print(f"[red]No SKILL.md found in {skill_dir}[/red]")
        sys.exit(1)

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

    # Determine if this is benchmark mode or single-model eval mode
    model_list = models if models else [config.effective_eval_model]
    is_benchmark_mode = len(model_list) > 1 or num_runs > 1

    test_cases: list[TestCase] = []

    async with fast.run() as agent:
        # Load or generate test cases
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
                model=model_list[0] if model_list else None,
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
            # Multi-model benchmark mode
            await _run_benchmark_mode(
                skill,
                test_cases,
                model_list,
                num_runs,
                fast,
                config,
                batch_id,
                batch_folder,
                log_runs,
                verbose,
                provider_info,
            )
        else:
            # Single-model eval mode with baseline comparison
            model = model_list[0]
            console.print(
                f"Running {len(test_cases)} test cases{provider_info}...", style="dim"
            )

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
                        assertions_passed=int(
                            results.baseline_success_rate * len(test_cases)
                        ),
                        assertions_total=len(test_cases),
                        run_type="baseline",
                        skill_name=skill.name,
                    )
                    write_run_metadata(baseline_folder, baseline_result.metadata)
                    write_run_result(baseline_folder, baseline_result)
                    run_results.append(baseline_result)

                # Log with-skill result
                with_skill_folder = create_run_folder(
                    batch_folder, 2 if not no_baseline else 1
                )
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
                    assertions_passed=int(
                        results.with_skill_success_rate * len(test_cases)
                    ),
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
                    base_icon = (
                        "[green]OK[/green]" if base_r.success else "[red]FAIL[/red]"
                    )
                    skill_icon = (
                        "[green]OK[/green]" if with_r.success else "[red]FAIL[/red]"
                    )
                    input_preview = with_r.test_case.input[:40]
                    console.print(
                        f"  {i}. {input_preview}  {base_icon} base  {skill_icon} skill"
                    )
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
                    savings_str = (
                        f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
                    )
                    savings_style = (
                        "green" if savings > 0 else "red" if savings < 0 else "dim"
                    )
                    console.print()
                    console.print(
                        f"  tokens: {results.baseline_total_tokens} → "
                        f"{results.with_skill_total_tokens}  "
                        f"[{savings_style}]({savings_str})[/{savings_style}]"
                    )
            else:
                with_skill_rate = results.with_skill_success_rate
                with_skill_bar = _render_bar(with_skill_rate)
                console.print(f"  with skill {with_skill_bar}  {with_skill_rate:>5.0%}")
                console.print(f"  tokens: {results.with_skill_total_tokens}")

            if not no_baseline:
                if results.is_beneficial:
                    console.print("\n[green]Recommendation: keep skill[/green]")
                else:
                    console.print(
                        "\n[yellow]Recommendation: skill may not be beneficial[/yellow]"
                    )


async def _run_benchmark_mode(
    skill: Skill,
    test_cases: list[TestCase],
    models: list[str],
    num_runs: int,
    fast: FastAgent,
    config: Config,
    batch_id: str | None,
    batch_folder: Path | None,
    log_runs: bool,
    verbose: bool,
    provider_info: str,
):
    """Run multi-model benchmark mode."""
    from upskill.evaluate import run_test

    console.print(f"\nEvaluating [bold]{skill.name}[/bold] across {len(models)} model(s)")
    console.print(
        f"  {len(test_cases)} test case(s), {num_runs} run(s) per model{provider_info}\n"
    )

    # Track results per model
    model_results: dict[str, list[RunResult]] = {m: [] for m in models}
    run_folder_counter = 0

    for model in models:
        console.print(f"[bold]{model}[/bold]")

        for run_num in range(1, num_runs + 1):
            run_folder_counter += 1
            run_folder = None
            if log_runs and batch_folder:
                run_folder = create_run_folder(batch_folder, run_folder_counter)

            # Run each test case
            total_assertions_passed = 0
            total_assertions = 0
            all_passed = True
            test_results: list[TestResult] = []

            for tc_idx, tc in enumerate(test_cases, 1):
                if verbose:
                    console.print(
                        f"  Running test {tc_idx}/{len(test_cases)}...", style="dim"
                    )

                try:
                    result = await run_test(
                        tc,
                        fast=fast,
                        skill=skill,
                        model=model,
                        config_path=config.effective_fastagent_config,
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
                    total_assertions += 1
                else:
                    total_assertions += 1
                    if result.success:
                        total_assertions_passed += 1

                test_results.append(result)

                if not result.success:
                    all_passed = False

            aggregated_stats = aggregate_conversation_stats(test_results)

            # Create run result
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

            if log_runs and run_folder:
                write_run_metadata(run_folder, run_result.metadata)
                write_run_result(run_folder, run_result)

            model_results[model].append(run_result)

            # Display progress
            status = "[green]PASS[/green]" if all_passed else "[red]FAIL[/red]"
            if verbose:
                total_tokens = aggregated_stats.total_tokens
                console.print(
                    f"  Run {run_num}: {total_assertions_passed}/{total_assertions} "
                    f"assertions  {total_tokens} tokens  {status}"
                )

        # Summary for this model
        passes = sum(1 for r in model_results[model] if r.passed)
        avg_assertions = (
            sum(r.assertions_passed for r in model_results[model])
            / len(model_results[model])
            if model_results[model]
            else 0
        )
        total_possible = (
            model_results[model][0].assertions_total if model_results[model] else 0
        )
        console.print(
            f"  Pass rate: {passes}/{num_runs} ({passes / num_runs:.0%})  "
            f"Avg assertions: {avg_assertions:.1f}/{total_possible}"
        )
        console.print()

    # Write batch summary
    if log_runs and batch_folder:
        all_results = [r for results in model_results.values() for r in results]
        summary = BatchSummary(
            batch_id=batch_id or "",
            model=",".join(models),
            task=skill.description,
            total_runs=len(all_results),
            passed_runs=sum(1 for r in all_results if r.passed),
            results=all_results,
        )
        write_batch_summary(batch_folder, summary)

    # Final summary table
    console.print()
    table = Table(show_header=True, title="Evaluation Summary")
    table.add_column("Model")
    table.add_column("Pass Rate")
    table.add_column("Avg Assertions")
    table.add_column("Avg Tokens")

    for model in models:
        results = model_results[model]
        passes = sum(1 for r in results if r.passed)
        avg_assertions = (
            sum(r.assertions_passed for r in results) / len(results) if results else 0
        )
        total_possible = results[0].assertions_total if results else 0
        avg_tokens = (
            sum(r.stats.total_tokens for r in results) / len(results) if results else 0
        )

        pass_style = (
            "green" if passes == num_runs else "yellow" if passes > 0 else "red"
        )
        table.add_row(
            model,
            f"[{pass_style}]{passes}/{num_runs}[/{pass_style}]",
            f"{avg_assertions:.1f}/{total_possible}",
            f"{avg_tokens:.0f}",
        )

    console.print(table)
    if batch_folder:
        console.print(f"\nResults saved to: {batch_folder}")


@main.command("list")
@click.option("-d", "--dir", "skills_dir", type=click.Path(), help="Skills directory to list")
@click.option("-v", "--verbose", is_flag=True, help="Show skill contents preview")
def list_cmd(skills_dir: str | None, verbose: bool):
    """List generated skills.

    Examples:

        upskill list

        upskill list -d ./my-skills/

        upskill list -v
    """
    config = Config.load()
    if skills_dir:
        path = Path(skills_dir)
    else:
        path = config.skills_dir

    if not path.exists():
        console.print(f"[yellow]No skills directory found at {path}[/yellow]")
        return

    skills = [d for d in path.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

    if not skills:
        console.print(f"[yellow]No skills found in {path}[/yellow]")
        return

    # Build tree view
    tree = Tree(f"[bold]{path}[/bold]")

    for skill_dir in sorted(skills):
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text()
        lines = content.split("\n")
        name = skill_dir.name
        description = lines[2] if len(lines) > 2 else ""

        # Add skill to tree
        skill_branch = tree.add(f"[bold cyan]{name}[/bold cyan]")
        if description:
            skill_branch.add(f"[dim]{description[:70]}{'...' if len(description) > 70 else ''}[/dim]")

        # Add file details
        files_branch = skill_branch.add("[dim]files[/dim]")
        files_branch.add("SKILL.md")

        # Check for references
        refs_dir = skill_dir / "references"
        if refs_dir.exists():
            for ref_file in sorted(refs_dir.iterdir()):
                files_branch.add(f"references/{ref_file.name}")

        # Check for scripts
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.exists():
            for script_file in sorted(scripts_dir.iterdir()):
                files_branch.add(f"scripts/{script_file.name}")

        # Show preview if verbose
        if verbose:
            # Extract first meaningful content after frontmatter
            in_frontmatter = False
            preview_lines = []
            for line in lines:
                if line.strip() == "---":
                    in_frontmatter = not in_frontmatter
                    continue
                if not in_frontmatter and line.strip() and not line.startswith("#"):
                    preview_lines.append(line)
                    if len(preview_lines) >= 3:
                        break
            if preview_lines:
                preview = "\n".join(preview_lines)
                skill_branch.add(Panel(Syntax(preview, "markdown", theme="monokai"), title="preview", border_style="dim"))

    console.print(tree)


@main.command("runs")
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option("-m", "--model", "models", multiple=True, help="Filter by model(s)")
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to display (default: success)",
)
@click.option("--csv", "csv_output", type=click.Path(), help="Export to CSV instead of plot")
def runs_cmd(
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    metric: str,
    csv_output: str | None,
):
    """View run results as a plot, or export to CSV.

    By default, shows a visual comparison of baseline vs with-skill performance.
    Use --csv to export data to CSV format instead.

    Examples:

        upskill runs

        upskill runs -d ./runs/

        upskill runs -s my-skill -m haiku -m sonnet

        upskill runs --metric tokens

        upskill runs --csv ./results.csv
    """
    config = Config.load()
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir

    if not runs_path.exists():
        console.print(f"[red]No runs directory found at {runs_path}[/red]")
        sys.exit(1)

    # CSV export mode
    if csv_output:
        try:
            output_path = summarize_runs_to_csv(runs_path, Path(csv_output))
            console.print(f"Summary written to {output_path}")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        return

    # Plot mode (default)
    all_results = _load_eval_results(runs_path)

    if not all_results:
        console.print("[yellow]No eval results with baseline comparisons found.[/yellow]")
        console.print("Run 'upskill eval <skill>' to generate comparable results.")
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
        # Single skill, multiple models
        skill_name = list(unique_skills)[0]
        console.print(f"[bold]skill: {skill_name}[/bold]\n")

        for r in sorted(results_list, key=lambda x: x["model"]):
            _print_comparison_bars(r, metric)

    elif len(unique_models) == 1 and len(unique_skills) >= 1:
        # Single model, multiple skills
        model_name = list(unique_models)[0]
        console.print(f"[bold]model: {model_name}[/bold]\n")

        for r in sorted(results_list, key=lambda x: x["skill_name"]):
            _print_comparison_bars(r, metric, label_field="skill_name")

    else:
        # Multiple skills and models - matrix view
        _print_matrix_view(results_list, metric)


def _render_bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    """Render a horizontal progress bar."""
    filled_count = int(value * width)
    return filled * filled_count + empty * (width - filled_count)


def _load_eval_results(runs_path: Path) -> list[dict]:
    """Load eval results from batch summaries, extracting baseline vs with-skill pairs."""
    results = []

    for batch_dir in sorted(runs_path.iterdir()):
        if not batch_dir.is_dir():
            continue

        summary_file = batch_dir / "batch_summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)

        # Group results by model and skill
        baseline_by_key: dict[tuple[str, str], dict] = {}
        with_skill_by_key: dict[tuple[str, str], dict] = {}

        for run in summary.get("results", []):
            model = run.get("metadata", {}).get("model", summary.get("model", "unknown"))
            skill_name = run.get("skill_name")
            run_type = run.get("run_type", "with_skill")

            if not skill_name:
                continue

            key = (model, skill_name)
            assertions_total = run.get("assertions_total", 1)
            success_rate = (
                run.get("assertions_passed", 0) / assertions_total if assertions_total else 0
            )

            entry = {
                "model": model,
                "skill_name": skill_name,
                "success_rate": success_rate,
                "tokens": run.get("stats", {}).get("total_tokens", 0),
                "batch_id": summary.get("batch_id"),
            }

            if run_type == "baseline":
                baseline_by_key[key] = entry
            else:
                with_skill_by_key[key] = entry

        # Pair up baseline and with-skill results
        for key, with_skill in with_skill_by_key.items():
            baseline = baseline_by_key.get(key)
            results.append({
                "model": key[0],
                "skill_name": key[1],
                "baseline_rate": baseline["success_rate"] if baseline else None,
                "with_skill_rate": with_skill["success_rate"],
                "baseline_tokens": baseline["tokens"] if baseline else None,
                "with_skill_tokens": with_skill["tokens"],
                "batch_id": with_skill["batch_id"],
                "has_baseline": baseline is not None,
            })

    return results


def _print_comparison_bars(result: dict, metric: str, label_field: str = "model") -> None:
    """Print baseline vs with-skill comparison bars for a single result."""
    label = result[label_field]
    has_baseline = result.get("has_baseline", True)
    console.print(f"[bold]{label}[/bold]")

    if metric == "success":
        with_skill_val = result["with_skill_rate"]
        with_skill_bar = _render_bar(with_skill_val)

        if has_baseline:
            baseline_val = result["baseline_rate"]
            lift = with_skill_val - baseline_val
            baseline_bar = _render_bar(baseline_val)

            console.print(f"  baseline   {baseline_bar}  {baseline_val:>5.0%}")

            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
            lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"
            console.print(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>5.0%}  "
                f"[{lift_style}]({lift_str})[/{lift_style}]"
            )
        else:
            # Benchmark-only (no baseline)
            console.print(
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

            console.print(f"  baseline   {baseline_bar}  {baseline_val:>6}")
            console.print(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>6}  "
                f"[{savings_style}]({savings_str})[/{savings_style}]"
            )
        else:
            # Benchmark-only (no baseline)
            with_skill_bar = _render_bar(1.0 if with_skill_val > 0 else 0)
            console.print(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>6}  [dim](no baseline)[/dim]"
            )

    console.print()


def _print_matrix_view(results: list[dict], metric: str) -> None:
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
                has_baseline = r.get("has_baseline", True)
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

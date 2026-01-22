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
from rich.table import Table

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
@click.option("--trace", type=click.Path(exists=True), help="Generate from agent trace")
@click.option(
    "-f",
    "--from",
    "from_skill",
    type=click.Path(exists=True),
    help="Existing skill to improve (path to skill directory)",
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
    trace: str | None,  # noqa: ARG001
    from_skill: str | None,
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

    Examples:

        upskill generate "parse JSON Schema files"

        upskill generate "write git commits" --model sonnet

        upskill generate "handle API errors" --eval-model haiku

        upskill generate "validate forms" -o ./my-skills/validation

        # Improve an existing skill:

        upskill generate "add more error handling examples" --from ./skills/api-errors/

        upskill generate "make it more concise" -f ./skills/my-skill/ -o ./skills/my-skill-v2/

        # Evaluate on a local model (Ollama):

        upskill generate "parse YAML" --eval-model llama3.2:latest \\
            --eval-base-url http://localhost:11434/v1

        upskill generate "document code" --no-log-runs
    """
    asyncio.run(
        _generate_async(
            task,
            list(example) if example else None,
            from_skill,
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


    # todo -- legacy in f-a.

    
    fast = FastAgent(
        "upskill",
        ignore_unknown_args=True,
        ## NB - at the moment we let fast-agent see CLI arguments, check for conflicts/consistency/behaviour required
#        parse_cli_args=False,        
#        config_path=str(CONFIG_PATH),
#        environment_dir=environment_dir,
#        skills_directory=[skills_manifest_dir],
    )

    @fast.agent()
    async def empty():
        pass
        

    # load agents from card files
    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        fast.load_agents(cards_path)

    async with fast.run() as agent:

        # Either improve existing skill or generate new one
        if from_skill:
            existing_skill = Skill.load(Path(from_skill))
            console.print(
                f"Improving [bold]{existing_skill.name}[/bold] with {gen_model}...",
                style="dim",
            )
            skill = await improve_skill(existing_skill, instructions=task, generator=agent.skill_gen, model=model)
        else:
            console.print(f"Generating skill with {gen_model}...", style="dim")
            print(agent._agents)
            skill = await generate_skill(task=task, examples=examples, generator=agent.skill_gen, model=model)
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
                skill = await refine_skill(
                    skill,
                    failures,
                    generator=agent.skill_gen,
                    model=model,
                )

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
@click.option("-m", "--model", help="Model to evaluate against (e.g., 'sonnet', 'llama3.2:latest')")
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
    model: str | None,
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
            skill_path, tests, model, provider, base_url, no_baseline, verbose, log_runs, runs_dir
        )
    )


async def _eval_async(
    skill_path: str,
    tests: str | None,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    no_baseline: bool,
    verbose: bool,
    log_runs: bool,
    runs_dir: str | None,
):
    """Async implementation of eval command."""
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

    results = None
    test_cases: list[TestCase] = []

    async with fast.run() as agent:
        if tests:
            with open(tests, encoding="utf-8") as f:
                data = json.load(f)
            if "cases" in data:
                test_cases = [TestCase(**tc) for tc in data["cases"]]
            else:
                test_cases = [TestCase(**tc) for tc in data]
        else:
            console.print("Generating test cases from skill...", style="dim")
            test_cases = await generate_tests(
                skill.description,
                generator=agent.test_gen,
                model=model,
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
        console.print(f"Running {len(test_cases)} test cases{provider_info}...", style="dim")

        results = await evaluate_skill(
            skill,
            test_cases,
            evaluator=agent.evaluator,
            model=model,
            run_baseline=not no_baseline,
        )

        # Log results (both baseline and with-skill for plot command)
        run_results: list[RunResult] = []
        eval_model = model or config.effective_eval_model
        if log_runs and batch_folder:
            # Log baseline result
            if not no_baseline:
                baseline_folder = create_run_folder(batch_folder, 1)
                baseline_result = RunResult(
                    metadata=RunMetadata(
                        model=eval_model,
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
                    model=eval_model,
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
                model=eval_model,
                task=skill.description,
                total_runs=len(run_results),
                passed_runs=sum(1 for r in run_results if r.passed),
                results=run_results,
            )
            write_batch_summary(batch_folder, summary)

    if results is None:
        return

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
            console.print(
                f"  tokens: {results.baseline_total_tokens} → {results.with_skill_total_tokens}  "
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
            console.print("\n[yellow]Recommendation: skill may not be beneficial[/yellow]")


@main.command("list")
@click.option("-d", "--dir", "skills_dir", type=click.Path(), help="Skills directory to list")
def list_cmd(skills_dir: str | None):
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

    console.print(f"Skills in {path}:\n")
    for skill_dir in sorted(skills):
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text()
        lines = content.split("\n")
        name = skill_dir.name
        description = lines[2] if len(lines) > 2 else ""
        console.print(f"  [bold]{name}[/bold]")
        if description:
            console.print(f"    {description[:60]}...")
        console.print()


@main.command("runs")
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("--csv", "csv_output", type=click.Path(), help="Output CSV path")
def runs_cmd(runs_dir: str | None, csv_output: str | None):
    """Summarize run logs to CSV."""
    config = Config.load()
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir

    if not runs_path.exists():
        console.print(f"[red]No runs directory found at {runs_path}[/red]")
        sys.exit(1)

    try:
        output_path = summarize_runs_to_csv(runs_path, Path(csv_output) if csv_output else None)
        console.print(f"Summary written to {output_path}")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


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

        console.print(f"\nBenchmarking [bold]{skill.name}[/bold] across {len(models)} model(s)")
        console.print(f"  {len(test_cases)} test case(s), {num_runs} run(s) per model\n")

        for model in models:
            console.print(f"[bold]{model}[/bold]")

            for run_num in range(1, num_runs + 1):
                run_folder = create_run_folder(batch_folder, len(model_results[model]) + 1)

                # Run each test case
                total_assertions_passed = 0
                total_assertions = 0
                total_tokens = 0
                total_turns = 0
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

                    total_tokens += result.stats.total_tokens
                    total_turns += result.stats.turns

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

                # Display progress
                status = "[green]PASS[/green]" if all_passed else "[red]FAIL[/red]"
                if verbose:
                    console.print(
                        f"  Run {run_num}: {status} "
                        f"({total_assertions_passed}/{total_assertions} assertions passed)"
                    )

            console.print()

        # Summary report
        console.print("
[bold]Benchmark Summary[/bold]
")

        for model, results in model_results.items():
            total_runs = len(results)
            passed_runs = sum(1 for r in results if r.passed)
            avg_tokens = sum(r.stats.total_tokens for r in results) / total_runs if total_runs else 0
            avg_turns = sum(r.stats.turns for r in results) / total_runs if total_runs else 0

            pass_rate = passed_runs / total_runs if total_runs else 0
            pass_rate_str = f"{pass_rate:.0%}"
            pass_rate_style = "green" if pass_rate > 0.5 else "yellow" if pass_rate > 0 else "red"

            console.print(f"[bold]{model}[/bold]")
            console.print(f"  Runs: {total_runs} | Passed: {passed_runs} ({pass_rate_str})")
            console.print(f"  Avg tokens: {avg_tokens:.0f} | Avg turns: {avg_turns:.1f}")
            console.print()

        # Save summary to file
        summary = BatchSummary(
            batch_id=batch_id,
            model=", ".join(models),
            task=skill.description,
            total_runs=sum(len(r) for r in model_results.values()),
            passed_runs=sum(1 for results in model_results.values() for r in results if r.passed),
            results=[r for results in model_results.values() for r in results],
        )
        write_batch_summary(batch_folder, summary)



@main.command("plot")
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option("-m", "--model", "models", multiple=True, help="Filter by model(s)")
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to plot",
)
def plot_cmd(
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    metric: str,
):
    """Plot model performance: baseline vs with-skill.

    Shows horizontal bar charts comparing performance with and without skills.

    Examples:

        upskill plot

        upskill plot -d ./runs/

        upskill plot -s my-skill -m haiku -m sonnet

        upskill plot --metric tokens
    """
    config = Config.load()
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir

    if not runs_path.exists():
        console.print(f"[red]No runs directory found at {runs_path}[/red]")
        sys.exit(1)

    # Load all eval results
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

"""CLI interface for upskill."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from upskill.config import Config
from upskill.evaluate import evaluate_skill, get_failure_descriptions
from upskill.generate import generate_skill, generate_tests, refine_skill
from upskill.logging import (
    create_batch_folder,
    create_run_folder,
    summarize_runs_to_csv,
    write_batch_summary,
    write_run_metadata,
    write_run_result,
)
from upskill.models import (
    BatchSummary,
    ConversationStats,
    RunMetadata,
    RunResult,
    Skill,
    TestCase,
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
    help="API provider for eval model (auto-detected as 'generic' when --eval-base-url is provided)",
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
    model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    eval_provider: str | None,
    eval_base_url: str | None,
    runs_dir: str | None,
    log_runs: bool,
):
    """Generate a skill from a task description.

    Examples:

        upskill generate "parse JSON Schema files"

        upskill generate "write git commits" --model sonnet

        upskill generate "handle API errors" --eval-model haiku

        upskill generate "validate forms" -o ./my-skills/validation

        # Evaluate on a local model (Ollama):

        upskill generate "parse YAML" --eval-model llama3.2:latest \\
            --eval-base-url http://localhost:11434/v1

        # Evaluate on a local model (llama.cpp server):

        upskill generate "parse YAML" --eval-model my-model \\
            --eval-base-url http://localhost:8080/v1

        upskill generate "document code" --no-log-runs
    """
    asyncio.run(
        _generate_async(
            task,
            list(example) if example else None,
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

    console.print(f"Generating skill with {gen_model}...", style="dim")
    skill = await generate_skill(task=task, examples=examples, model=model, config=config)

    if no_eval:
        _save_and_display(skill, output, config)
        return

    console.print("Generating test cases...", style="dim")
    test_cases = await generate_tests(task, model=model, config=config)

    # Eval loop with refinement (on generation model)
    prev_success_rate = 0.0
    results = None
    for attempt in range(config.max_refine_attempts):
        console.print(f"Evaluating on {gen_model}... (attempt {attempt + 1})", style="dim")

        # Create run folder for logging
        run_folder = None
        if log_runs and batch_folder:
            run_folder = create_run_folder(batch_folder, attempt + 1)
            write_run_metadata(
                run_folder,
                RunMetadata(
                    model=gen_model,
                    task=task,
                    batch_id=batch_id or "",
                    run_number=attempt + 1,
                ),
            )

        results = await evaluate_skill(skill, test_cases, model=gen_model, config=config)

        # Log run result
        if log_runs and run_folder:
            run_result = RunResult(
                metadata=RunMetadata(
                    model=gen_model,
                    task=task,
                    batch_id=batch_id or "",
                    run_number=attempt + 1,
                ),
                stats=ConversationStats(
                    tokens=results.with_skill_total_tokens,
                    turns=int(results.with_skill_avg_turns * len(test_cases)),
                ),
                passed=results.is_beneficial,
                assertions_passed=int(results.with_skill_success_rate * len(test_cases)),
                assertions_total=len(test_cases),
            )
            write_run_result(run_folder, run_result)
            run_results.append(run_result)

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
            model=eval_model,
            config=config,
            provider=eval_provider,
            base_url=eval_base_url,
        )

        # Log eval run result
        if log_runs and run_folder:
            run_result = RunResult(
                metadata=RunMetadata(
                    model=eval_model,
                    task=task,
                    batch_id=batch_id or "",
                    run_number=config.max_refine_attempts + 1,
                ),
                stats=ConversationStats(
                    tokens=eval_results.with_skill_total_tokens,
                    turns=int(eval_results.with_skill_avg_turns * len(test_cases)),
                ),
                passed=eval_results.is_beneficial,
                assertions_passed=int(eval_results.with_skill_success_rate * len(test_cases)),
                assertions_total=len(test_cases),
            )
            write_run_result(run_folder, run_result)
            run_results.append(run_result)

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

    # Show results for both models if we have eval_results
    if results and eval_results:
        console.print()
        table = Table(show_header=True, title="Skill Impact by Model")
        table.add_column("model")
        table.add_column("baseline")
        table.add_column("with skill")
        table.add_column("lift")

        # Generation model results
        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
        lift_style = "green" if lift > 0 else "red" if lift < 0 else ""
        table.add_row(
            gen_model or "gen",
            f"{results.baseline_success_rate:.0%}",
            f"{results.with_skill_success_rate:.0%}",
            f"[{lift_style}]{lift_str}[/{lift_style}]" if lift_style else lift_str,
        )

        # Eval model results
        lift = eval_results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
        lift_style = "green" if lift > 0 else "red" if lift < 0 else ""
        table.add_row(
            eval_model or "eval",
            f"{eval_results.baseline_success_rate:.0%}",
            f"{eval_results.with_skill_success_rate:.0%}",
            f"[{lift_style}]{lift_str}[/{lift_style}]" if lift_style else lift_str,
        )

        console.print(table)

    elif results:
        console.print()
        table = Table(show_header=True)
        table.add_column("")
        table.add_column("baseline")
        table.add_column("with skill")
        table.add_column("")

        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
        table.add_row(
            "success",
            f"{results.baseline_success_rate:.0%}",
            f"{results.with_skill_success_rate:.0%}",
            lift_str,
        )

        savings = results.token_savings
        savings_str = f"-{savings:.0%}" if savings > 0 else f"+{-savings:.0%}"
        table.add_row(
            "tokens",
            str(results.baseline_total_tokens),
            str(results.with_skill_total_tokens),
            savings_str,
        )

        console.print(table)

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

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        console.print(f"[red]No SKILL.md found in {skill_dir}[/red]")
        sys.exit(1)

    content = skill_md.read_text()
    lines = content.split("\n")
    name = lines[0].lstrip("# ").strip() if lines else "unknown"
    description = lines[2] if len(lines) > 2 else ""
    body = "\n".join(lines[4:]) if len(lines) > 4 else content

    skill = Skill(name=name, description=description, body=body)

    if tests:
        with open(tests, encoding="utf-8") as f:
            data = json.load(f)
        if "cases" in data:
            test_cases = [TestCase(**tc) for tc in data["cases"]]
        else:
            test_cases = [TestCase(**tc) for tc in data]
    else:
        console.print("Generating test cases from skill...", style="dim")
        test_cases = await generate_tests(skill.description, config=config)

    # Setup run logging
    batch_id = None
    batch_folder = None
    if log_runs:
        runs_path = Path(runs_dir) if runs_dir else config.runs_dir
        batch_id, batch_folder = create_batch_folder(runs_path)
        run_folder = create_run_folder(batch_folder, 1)
        write_run_metadata(
            run_folder,
            RunMetadata(
                model=model or config.effective_eval_model,
                task=skill.description,
                batch_id=batch_id,
                run_number=1,
            ),
        )
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
        model=model,
        config=config,
        run_baseline=not no_baseline,
        provider=provider,
        base_url=base_url,
    )

    # Log results
    if log_runs and batch_folder:
        run_folder = batch_folder / "run_1"
        run_result = RunResult(
            metadata=RunMetadata(
                model=model or config.effective_eval_model,
                task=skill.description,
                batch_id=batch_id or "",
                run_number=1,
            ),
            stats=ConversationStats(
                tokens=results.with_skill_total_tokens,
                turns=int(results.with_skill_avg_turns * len(test_cases)),
            ),
            passed=results.is_beneficial
            if not no_baseline
            else results.with_skill_success_rate > 0.5,
            assertions_passed=int(results.with_skill_success_rate * len(test_cases)),
            assertions_total=len(test_cases),
        )
        write_run_result(run_folder, run_result)

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

    table = Table(show_header=True)
    table.add_column("")
    table.add_column("baseline")
    table.add_column("with skill")
    table.add_column("")

    n = len(test_cases)
    if not no_baseline:
        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
        base_pass = int(results.baseline_success_rate * n)
        skill_pass = int(results.with_skill_success_rate * n)
        table.add_row(
            "success",
            f"{base_pass}/{n} ({results.baseline_success_rate:.0%})",
            f"{skill_pass}/{n} ({results.with_skill_success_rate:.0%})",
            lift_str,
        )

        savings = results.token_savings
        savings_str = f"-{savings:.0%}" if savings > 0 else f"+{-savings:.0%}"
        table.add_row(
            "tokens",
            str(results.baseline_total_tokens),
            str(results.with_skill_total_tokens),
            savings_str,
        )
    else:
        skill_pass = int(results.with_skill_success_rate * n)
        table.add_row(
            "success",
            "-",
            f"{skill_pass}/{n} ({results.with_skill_success_rate:.0%})",
            "",
        )
        table.add_row("tokens", "-", str(results.with_skill_total_tokens), "")

    console.print(table)

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


if __name__ == "__main__":
    main()

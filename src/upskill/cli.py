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
from upskill.models import Skill, TestCase

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
@click.option("-m", "--model", help="Model to use for generation")
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.option("--no-eval", is_flag=True, help="Skip eval and refinement")
@click.option("--eval-model", help="Different model to evaluate skill on")
@click.option("--eval-provider", type=click.Choice(["anthropic", "openai"]),
              help="API provider for eval model")
@click.option("--eval-base-url", help="Custom API endpoint for eval model")
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
):
    """Generate a skill from a task description.

    Examples:

        upskill generate "parse JSON Schema files"

        # Generate with Sonnet, eval on local Ollama model
        upskill generate "parse YAML files" \\
            --model claude-sonnet-4-20250514 \\
            --eval-model llama3.2 \\
            --eval-provider openai \\
            --eval-base-url http://localhost:11434/v1
    """
    asyncio.run(
        _generate_async(
            task, list(example) if example else None, model, output, no_eval,
            eval_model, eval_provider or "anthropic", eval_base_url
        )
    )


async def _generate_async(
    task: str,
    examples: list[str] | None,
    model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    eval_provider: str,
    eval_base_url: str | None,
):
    """Async implementation of generate command."""
    config = Config.load()
    gen_model = model or config.model

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
        results = await evaluate_skill(skill, test_cases, model=gen_model, config=config)

        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"

        if results.is_beneficial:
            console.print(
                f"  {results.baseline_success_rate:.0%} -> "
                f"{results.with_skill_success_rate:.0%} ({lift_str}) [green]✓[/green]"
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
        console.print(f"Evaluating on {eval_model}...", style="dim")
        eval_results = await evaluate_skill(
            skill, test_cases, model=eval_model, config=config,
            provider=eval_provider, base_url=eval_base_url
        )
        lift = eval_results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
        console.print(
            f"  {eval_results.baseline_success_rate:.0%} -> "
            f"{eval_results.with_skill_success_rate:.0%} ({lift_str})"
        )

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
@click.option("-t", "--tests", type=click.Path(exists=True), help="Test cases JSON")
@click.option("-m", "--model", help="Model to evaluate against")
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic",
              help="API provider (anthropic or openai-compatible)")
@click.option("--base-url", help="Custom API endpoint (e.g., http://localhost:8080/v1)")
@click.option("--no-baseline", is_flag=True, help="Skip baseline comparison")
@click.option("-v", "--verbose", is_flag=True, help="Show per-test results")
def eval_cmd(
    skill_path: str,
    tests: str | None,
    model: str | None,
    provider: str,
    base_url: str | None,
    no_baseline: bool,
    verbose: bool,
):
    """Evaluate a skill (compares with vs without).

    Examples:

        upskill eval ./skill/

        upskill eval ./skill/ --tests ./tests.json -v

        # Evaluate with local llama.cpp (Anthropic API)
        upskill eval ./skill/ -m qwen3 --base-url http://localhost:8080

        # Evaluate with Ollama (OpenAI API)
        upskill eval ./skill/ -m llama3.2 --provider openai --base-url http://localhost:11434/v1
    """
    asyncio.run(_eval_async(skill_path, tests, model, provider, base_url, no_baseline, verbose))


async def _eval_async(
    skill_path: str,
    tests: str | None,
    model: str | None,
    provider: str,
    base_url: str | None,
    no_baseline: bool,
    verbose: bool,
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
        # Always generate tests with Anthropic (even when evaluating on local models)
        console.print("Generating test cases from skill...", style="dim")
        test_cases = await generate_tests(skill.description, config=config)

    provider_info = f" via {provider}"
    if base_url:
        provider_info += f" @ {base_url}"
    console.print(f"Running {len(test_cases)} test cases{provider_info}...", style="dim")

    results = await evaluate_skill(
        skill, test_cases, model=model, config=config, run_baseline=not no_baseline,
        provider=provider, base_url=base_url
    )

    if verbose and not no_baseline:
        console.print()
        for i, (with_r, base_r) in enumerate(
            zip(results.with_skill_results, results.baseline_results), 1
        ):
            base_icon = "[green]✓[/green]" if base_r.success else "[red]✗[/red]"
            skill_icon = "[green]✓[/green]" if with_r.success else "[red]✗[/red]"
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
def list_cmd():
    """List generated skills."""
    config = Config.load()
    skills_dir = config.skills_dir

    if not skills_dir.exists():
        console.print("No skills directory found.")
        return

    skills = [
        d for d in skills_dir.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    ]

    if not skills:
        console.print("No skills found.")
        return

    console.print(f"Skills in {skills_dir}:\n")
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


if __name__ == "__main__":
    main()

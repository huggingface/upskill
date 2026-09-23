"""Model resolution helpers for CLI commands.

This module centralizes command-specific model fallback and mode selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from upskill.config import Config

CommandName = Literal["generate", "eval", "benchmark"]


@dataclass(frozen=True)
class ResolvedModels:
    """Resolved model plan for a command invocation."""

    skill_generation_model: str | None = None
    test_generation_model: str | None = None
    evaluation_models: list[str] = field(default_factory=list)
    extra_eval_model: str | None = None
    is_benchmark_mode: bool = False
    run_baseline: bool = True


def build_fastagent_model_references(
    *,
    config: Config,
    resolved: ResolvedModels,
) -> dict[str, dict[str, str]]:
    """Build fast-agent model references for the standard upskill card slots."""

    skill_generation_model = resolved.skill_generation_model or config.skill_generation_model
    test_generation_model = (
        resolved.test_generation_model or config.test_gen_model or skill_generation_model
    )
    return {
        "system": {
            "default": skill_generation_model,
            "skill_gen": skill_generation_model,
            "test_gen": test_generation_model,
        }
    }


CLI_MODEL_PREFIX = "cli."


def _is_cli(model: str) -> bool:
    return model.startswith(CLI_MODEL_PREFIX)


def _resolve_eval_test_gen_model(
    *,
    cli_test_gen_model: str | None,
    config: Config,
    evaluation_models: list[str],
) -> str:
    """Pick the test-generation model for ``eval``/``benchmark`` runs.

    Resolution order (first match wins):

      1. Explicit CLI override (``--test-gen-model``), if provided.
      2. When every evaluation model is ``cli.<provider>`` and the configured
         ``test_gen_model`` is unset (or itself a non-CLI alias), use the first
         CLI evaluation model so the run stays key-free end-to-end.
      3. Config ``test_gen_model``, when set (covers both non-CLI and CLI
         values, including the all-CLI case where the user explicitly opted
         into a specific CLI test-gen model).
      4. Config ``skill_generation_model`` as the final fallback.
    """
    if cli_test_gen_model is not None:
        return cli_test_gen_model

    all_eval_cli = bool(evaluation_models) and all(_is_cli(m) for m in evaluation_models)
    config_test_gen = config.test_gen_model

    if all_eval_cli and (config_test_gen is None or not _is_cli(config_test_gen)):
        return evaluation_models[0]

    if config_test_gen is not None:
        return config_test_gen

    return config.skill_generation_model


def resolve_models(
    command: CommandName,
    *,
    config: Config,
    cli_model: str | None = None,
    cli_models: list[str] | tuple[str, ...] | None = None,
    cli_eval_model: str | None = None,
    cli_test_gen_model: str | None = None,
    num_runs: int = 1,
    no_baseline: bool = False,
) -> ResolvedModels:
    """Resolve all models and mode flags for a command.

    Args:
        command: CLI command name.
        config: Loaded upskill configuration.
        cli_model: Single ``--model`` value for ``generate``.
        cli_models: Repeatable ``-m/--model`` values for ``eval``/``benchmark``.
        cli_eval_model: Optional ``--eval-model`` value for ``generate``.
        cli_test_gen_model: Optional ``--test-gen-model`` override.
        num_runs: ``--runs`` value.
        no_baseline: Whether ``--no-baseline`` was passed.

    Returns:
        A ``ResolvedModels`` instance containing command-specific resolved model fields.
    """

    if command == "generate":
        skill_generation_model = cli_model or config.skill_generation_model
        test_generation_model = (
            cli_test_gen_model or config.test_gen_model or skill_generation_model
        )
        return ResolvedModels(
            skill_generation_model=skill_generation_model,
            test_generation_model=test_generation_model,
            evaluation_models=[skill_generation_model],
            extra_eval_model=cli_eval_model,
            is_benchmark_mode=False,
            run_baseline=True,
        )

    if command == "eval":
        evaluation_models = list(cli_models) if cli_models else [config.effective_eval_model]
        is_benchmark_mode = len(evaluation_models) > 1 or num_runs > 1
        run_baseline = (not no_baseline) if not is_benchmark_mode else False
        return ResolvedModels(
            test_generation_model=_resolve_eval_test_gen_model(
                cli_test_gen_model=cli_test_gen_model,
                config=config,
                evaluation_models=evaluation_models,
            ),
            evaluation_models=evaluation_models,
            is_benchmark_mode=is_benchmark_mode,
            run_baseline=run_baseline,
        )

    if command == "benchmark":
        evaluation_models = list(cli_models) if cli_models else []
        if not evaluation_models:
            raise ValueError("benchmark requires at least one model")
        return ResolvedModels(
            test_generation_model=_resolve_eval_test_gen_model(
                cli_test_gen_model=cli_test_gen_model,
                config=config,
                evaluation_models=evaluation_models,
            ),
            evaluation_models=evaluation_models,
            is_benchmark_mode=True,
            run_baseline=False,
        )

    raise ValueError(f"Unsupported command: {command}")

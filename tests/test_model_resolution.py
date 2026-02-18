from __future__ import annotations

import pytest

from upskill.config import Config
from upskill.model_resolution import resolve_models


def test_resolve_generate_uses_generation_model_for_test_gen_by_default() -> None:
    config = Config(skill_generation_model="sonnet", eval_model="haiku", test_gen_model=None)

    resolved = resolve_models("generate", config=config)

    assert resolved.skill_generation_model == "sonnet"
    assert resolved.test_generation_model == "sonnet"
    assert resolved.extra_eval_model is None
    assert resolved.is_benchmark_mode is False
    assert resolved.skill_generation_model is not None
    assert resolved.test_generation_model is not None


def test_resolve_generate_honors_test_gen_model_config() -> None:
    config = Config(skill_generation_model="sonnet", test_gen_model="haiku")

    resolved = resolve_models("generate", config=config, cli_model="opus", cli_eval_model="haiku")

    assert resolved.skill_generation_model == "opus"
    assert resolved.test_generation_model == "haiku"
    assert resolved.extra_eval_model == "haiku"


def test_resolve_generate_cli_test_gen_model_overrides_config() -> None:
    config = Config(skill_generation_model="sonnet", test_gen_model="haiku")

    resolved = resolve_models(
        "generate",
        config=config,
        cli_model="sonnet",
        cli_test_gen_model="opus",
    )

    assert resolved.test_generation_model == "opus"


def test_resolve_eval_defaults_and_simple_mode() -> None:
    config = Config(skill_generation_model="sonnet", eval_model="haiku", test_gen_model=None)

    resolved = resolve_models("eval", config=config, cli_models=None, num_runs=1, no_baseline=False)

    assert resolved.evaluation_models == ["haiku"]
    assert resolved.test_generation_model == "sonnet"
    assert resolved.is_benchmark_mode is False
    assert resolved.run_baseline is True
    assert resolved.evaluation_models


def test_resolve_eval_cli_test_gen_model_overrides_config() -> None:
    config = Config(skill_generation_model="sonnet", test_gen_model="haiku")

    resolved = resolve_models(
        "eval",
        config=config,
        cli_models=["kimi"],
        cli_test_gen_model="opus",
    )

    assert resolved.test_generation_model == "opus"


def test_resolve_eval_benchmark_mode_disables_baseline() -> None:
    config = Config(skill_generation_model="sonnet", eval_model="haiku")

    resolved = resolve_models(
        "eval",
        config=config,
        cli_models=["haiku", "sonnet"],
        num_runs=3,
        no_baseline=False,
    )

    assert resolved.evaluation_models == ["haiku", "sonnet"]
    assert resolved.is_benchmark_mode is True
    assert resolved.run_baseline is False


def test_resolve_eval_simple_mode_respects_no_baseline() -> None:
    config = Config(skill_generation_model="sonnet", eval_model="haiku")

    resolved = resolve_models(
        "eval",
        config=config,
        cli_models=["haiku"],
        num_runs=1,
        no_baseline=True,
    )

    assert resolved.is_benchmark_mode is False
    assert resolved.run_baseline is False


def test_resolve_benchmark_requires_models() -> None:
    config = Config(skill_generation_model="sonnet")

    with pytest.raises(ValueError):
        resolve_models("benchmark", config=config, cli_models=[])


def test_resolve_benchmark_uses_config_test_generation_fallback() -> None:
    config = Config(skill_generation_model="sonnet", test_gen_model="opus")

    resolved = resolve_models("benchmark", config=config, cli_models=["haiku"], num_runs=2)

    assert resolved.evaluation_models == ["haiku"]
    assert resolved.test_generation_model == "opus"
    assert resolved.is_benchmark_mode is True
    assert resolved.run_baseline is False


def test_resolve_benchmark_cli_test_gen_model_overrides_config() -> None:
    config = Config(skill_generation_model="sonnet", test_gen_model="haiku")

    resolved = resolve_models(
        "benchmark",
        config=config,
        cli_models=["kimi"],
        cli_test_gen_model="opus",
    )

    assert resolved.test_generation_model == "opus"


def test_resolve_eval_prefers_cli_models_over_config_default() -> None:
    config = Config(skill_generation_model="sonnet", eval_model="haiku")

    resolved = resolve_models(
        "eval",
        config=config,
        cli_models=["opus"],
    )

    assert resolved.evaluation_models == ["opus"]


def test_resolve_unsupported_command_raises() -> None:
    config = Config(skill_generation_model="sonnet")

    with pytest.raises(ValueError, match="Unsupported command"):
        resolve_models("not-a-command", config=config)  # type: ignore[arg-type]


def test_config_legacy_model_key_maps_to_skill_generation_model() -> None:
    config = Config(model="haiku")

    assert config.skill_generation_model == "haiku"
    assert config.model == "haiku"


def test_config_dump_uses_skill_generation_model_key() -> None:
    config = Config(skill_generation_model="sonnet")

    dumped = config.model_dump(mode="json")
    assert dumped["skill_generation_model"] == "sonnet"
    assert "model" not in dumped

from __future__ import annotations

import yaml

from upskill.config import (
    UPSKILL_CONFIG_ENV,
    Config,
    find_upskill_config_path,
    resolve_upskill_config_path,
)


def test_find_upskill_config_path_uses_env_override_when_file_is_missing(
    tmp_path, monkeypatch
) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))
    monkeypatch.chdir(tmp_path)

    assert find_upskill_config_path() == override_path


def test_resolve_upskill_config_path_reports_missing_env_override(tmp_path, monkeypatch) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))

    resolution = resolve_upskill_config_path()

    assert resolution.path == override_path
    assert resolution.source == f"{UPSKILL_CONFIG_ENV} env var"
    assert resolution.exists is False


def test_config_save_uses_env_override_path_when_file_is_missing(tmp_path, monkeypatch) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))
    monkeypatch.chdir(tmp_path)

    config = Config(skill_generation_model="haiku")
    config.save()

    assert override_path.exists()
    assert (tmp_path / "upskill.config.yaml").exists() is False

    with open(override_path, encoding="utf-8") as f:
        saved = yaml.safe_load(f) or {}

    assert saved["skill_generation_model"] == "haiku"

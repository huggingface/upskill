from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts import cpd
from scripts.cpd import build_cpd_command, resolve_cli_exit_code, resolve_platform

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_platform_maps_common_linux_labels() -> None:
    platform_config = resolve_platform(system="linux", arch="x86_64")

    assert platform_config.system == "linux"
    assert platform_config.arch == "x86_64"
    assert platform_config.os_label == "linux"
    assert platform_config.arch_label == "x64"
    assert platform_config.java_name == "java"
    assert platform_config.pmd_name == "pmd"


def test_build_cpd_command_includes_expected_arguments(tmp_path: Path) -> None:
    platform_config = resolve_platform(system="linux", arch="x86_64")
    pmd_dir = tmp_path / "pmd-bin"
    src_dir = tmp_path / "src"
    excluded_path = src_dir / "skip_me.py"

    command = build_cpd_command(
        platform_config=platform_config,
        pmd_dir=pmd_dir,
        src_dir=src_dir,
        excluded_paths=[excluded_path],
        min_tokens=120,
        output_format="xml",
    )

    assert command == [
        str(pmd_dir / "bin" / "pmd"),
        "cpd",
        "--language",
        "python",
        "--minimum-tokens",
        "120",
        "--dir",
        str(src_dir),
        "--format",
        "xml",
        "--exclude",
        str(excluded_path),
    ]


def test_resolve_cli_exit_code_honors_check_mode() -> None:
    assert resolve_cli_exit_code(cpd_exit_code=0, check=False) == 0
    assert resolve_cli_exit_code(cpd_exit_code=4, check=False) == 0
    assert resolve_cli_exit_code(cpd_exit_code=4, check=True) == 1
    assert resolve_cli_exit_code(cpd_exit_code=7, check=True) == 7


@pytest.mark.parametrize("unsafe_member", ["../escape.txt", "..\\escape.txt", "C:\\tmp\\escape.txt"])
def test_ensure_pmd_rejects_zip_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_member: str,
) -> None:
    tools_dir = tmp_path / "tools"
    pmd_dir = tools_dir / f"pmd-bin-{cpd.PMD_VERSION}"
    tools_dir.mkdir()
    archive_path = tools_dir / f"pmd-{cpd.PMD_VERSION}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"pmd-bin-{cpd.PMD_VERSION}/bin/pmd", "#!/bin/sh\n")
        archive.writestr(unsafe_member, "bad")

    monkeypatch.setattr(cpd, "TOOLS_DIR", tools_dir)
    monkeypatch.setattr(cpd, "PMD_DIR", pmd_dir)
    monkeypatch.setattr(cpd, "download_file", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Unsafe path in archive"):
        cpd.ensure_pmd(resolve_platform(system="linux", arch="x86_64"))

    assert not (tmp_path / "escape.txt").exists()

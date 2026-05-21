#!/usr/bin/env python3
"""Copy/Paste Detector (CPD) runner for upskill.

Uses PMD's CPD tool to detect duplicated code in the Python source tree.
If Java or PMD are not already available, the script downloads them into
``~/tools`` and reuses them on later runs.

Usage:
    uv run scripts/cpd.py [--min-tokens N] [--format FORMAT] [--report FILE]

Options:
    --min-tokens N   Minimum token count for duplication (default: 100)
    --format FORMAT  Output format: text, csv, xml (default: text)
    --report FILE    Write report to file (default: stdout)
    --check          Exit with error code if duplications are found
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

JRE_VERSION: Final = "17.0.9+9"
PMD_VERSION: Final = "7.9.0"

TOOLS_DIR: Final = Path.home() / "tools"
JRE_DIR: Final = TOOLS_DIR / f"jdk-{JRE_VERSION}-jre"
PMD_DIR: Final = TOOLS_DIR / f"pmd-bin-{PMD_VERSION}"

JRE_URL_TEMPLATE: Final = (
    "https://github.com/adoptium/temurin17-binaries/releases/download/"
    "jdk-{version}/OpenJDK17U-jre_{arch}_{os}_hotspot_{archive_version}.tar.gz"
)
PMD_URL: Final = (
    "https://github.com/pmd/pmd/releases/download/pmd_releases%2F"
    f"{PMD_VERSION}/pmd-dist-{PMD_VERSION}-bin.zip"
)

CPD_EXCLUSIONS: Final[dict[str, str]] = {}


@dataclass(frozen=True)
class PlatformConfig:
    """Resolved platform labels for tool downloads."""

    system: str
    arch: str
    os_label: str
    arch_label: str

    @property
    def archive_version(self) -> str:
        return JRE_VERSION.replace("+", "_")

    @property
    def version_label(self) -> str:
        return JRE_VERSION.replace("+", "%2B")

    @property
    def java_name(self) -> str:
        return "java.exe" if self.system == "windows" else "java"

    @property
    def pmd_name(self) -> str:
        return "pmd.bat" if self.system == "windows" else "pmd"

    @property
    def jre_filename(self) -> str:
        return f"OpenJDK17U-jre_{self.arch_label}_{self.os_label}_hotspot_{self.archive_version}"

    @property
    def jre_url(self) -> str:
        return JRE_URL_TEMPLATE.format(
            version=self.version_label,
            arch=self.arch_label,
            os=self.os_label,
            archive_version=self.archive_version,
        )


def resolve_platform(*, system: str | None = None, arch: str | None = None) -> PlatformConfig:
    """Resolve download labels for the current platform."""
    normalized_system = (system or platform.system()).lower()
    normalized_arch = (arch or platform.machine()).lower()

    arch_label = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(normalized_arch, normalized_arch)

    os_label = {
        "darwin": "mac",
        "linux": "linux",
        "windows": "windows",
    }.get(normalized_system, normalized_system)

    return PlatformConfig(
        system=normalized_system,
        arch=normalized_arch,
        os_label=os_label,
        arch_label=arch_label,
    )


def download_file(url: str, destination: Path, description: str) -> None:
    """Download a file with simple progress reporting."""
    print(f"Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as error:  # pragma: no cover - network failures are environment-specific
        print(f"  Failed to download: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"  Downloaded to {destination}")


def is_unsafe_archive_member(member_name: str) -> bool:
    """Detect archive paths that are absolute or traverse parents on POSIX or Windows."""
    posix_path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    return (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )


def extract_tar_archive(archive_path: Path, destination: Path) -> None:
    """Extract a tar.gz archive while guarding against path traversal."""
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if is_unsafe_archive_member(member.name):
                message = f"Unsafe path in archive {archive_path}: {member.name}"
                raise RuntimeError(message)
            member_path = (destination_root / member.name).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as error:
                message = f"Unsafe path in archive {archive_path}: {member.name}"
                raise RuntimeError(message) from error
        archive.extractall(destination_root)


def extract_zip_archive(archive_path: Path, destination: Path) -> None:
    """Extract a zip archive while guarding against path traversal."""
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if is_unsafe_archive_member(member.filename):
                message = f"Unsafe path in archive {archive_path}: {member.filename}"
                raise RuntimeError(message)
            member_path = (destination_root / member.filename).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as error:
                message = f"Unsafe path in archive {archive_path}: {member.filename}"
                raise RuntimeError(message) from error
        archive.extractall(destination_root)


def ensure_jre(platform_config: PlatformConfig) -> Path:
    """Ensure Java is available, downloading a JRE if required."""
    java_bin = JRE_DIR / "bin" / platform_config.java_name
    if java_bin.exists():
        return JRE_DIR

    system_java = shutil.which("java")
    if system_java:
        try:
            result = subprocess.run(
                [system_java, "-version"],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            pass
        else:
            version_output = result.stderr + result.stdout
            if "17" in version_output or "21" in version_output:
                print(f"Using system Java: {system_java}")
                return Path(system_java).resolve().parent.parent

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / f"{platform_config.jre_filename}.tar.gz"
    if not archive_path.exists():
        download_file(platform_config.jre_url, archive_path, f"Java JRE {JRE_VERSION}")

    print("Extracting Java JRE...")
    extract_tar_archive(archive_path, TOOLS_DIR)

    if JRE_DIR.exists():
        return JRE_DIR

    for candidate in TOOLS_DIR.iterdir():
        if candidate.is_dir() and candidate.name.startswith("jdk-17"):
            return candidate

    message = f"Unable to locate extracted Java runtime under {TOOLS_DIR}"
    raise RuntimeError(message)


def ensure_pmd(platform_config: PlatformConfig) -> Path:
    """Ensure PMD is available, downloading it if required."""
    pmd_bin = PMD_DIR / "bin" / platform_config.pmd_name
    if pmd_bin.exists():
        return PMD_DIR

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / f"pmd-{PMD_VERSION}.zip"
    if not archive_path.exists():
        download_file(PMD_URL, archive_path, f"PMD {PMD_VERSION}")

    print("Extracting PMD...")
    extract_zip_archive(archive_path, TOOLS_DIR)

    if platform_config.system != "windows":
        pmd_bin.chmod(0o755)

    return PMD_DIR


def build_cpd_command(
    *,
    platform_config: PlatformConfig,
    pmd_dir: Path,
    src_dir: Path,
    excluded_paths: list[Path],
    min_tokens: int,
    output_format: str,
) -> list[str]:
    """Build the PMD CPD command line."""
    pmd_bin = pmd_dir / "bin" / platform_config.pmd_name
    command = [
        str(pmd_bin),
        "cpd",
        "--language",
        "python",
        "--minimum-tokens",
        str(min_tokens),
        "--dir",
        str(src_dir),
        "--format",
        output_format,
    ]
    for excluded_path in excluded_paths:
        command.extend(["--exclude", str(excluded_path)])
    return command


def run_cpd(
    *,
    platform_config: PlatformConfig,
    java_home: Path,
    pmd_dir: Path,
    src_dir: Path,
    excluded_paths: list[Path],
    min_tokens: int = 100,
    output_format: str = "text",
) -> tuple[int, str]:
    """Run CPD and return its exit code and combined output."""
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = f"{java_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    command = build_cpd_command(
        platform_config=platform_config,
        pmd_dir=pmd_dir,
        src_dir=src_dir,
        excluded_paths=excluded_paths,
        min_tokens=min_tokens,
        output_format=output_format,
    )
    result = subprocess.run(command, capture_output=True, check=False, text=True, env=env)
    return result.returncode, result.stdout + result.stderr


def resolve_cli_exit_code(*, cpd_exit_code: int, check: bool) -> int:
    """Map PMD CPD exit codes to the script's CLI exit codes."""
    if cpd_exit_code == 4:
        return 1 if check else 0
    return cpd_exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect duplicated code in upskill source")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum token count for duplication (default: 100)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "xml"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with error code if duplications are found",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    excluded_paths = [project_root / relative_path for relative_path in CPD_EXCLUSIONS]
    platform_config = resolve_platform()

    print("Checking dependencies...")
    java_home = ensure_jre(platform_config)
    pmd_dir = ensure_pmd(platform_config)
    print()

    print(f"Running CPD on {src_dir} (min-tokens={args.min_tokens})...")
    if excluded_paths:
        print("Excluding intentional duplicates:")
        for relative_path, reason in CPD_EXCLUSIONS.items():
            print(f"  - {relative_path}: {reason}")
    print()

    cpd_exit_code, output = run_cpd(
        platform_config=platform_config,
        java_home=java_home,
        pmd_dir=pmd_dir,
        src_dir=src_dir,
        excluded_paths=excluded_paths,
        min_tokens=args.min_tokens,
        output_format=args.format,
    )

    if args.report:
        args.report.write_text(output, encoding="utf-8")
        print(f"Report written to {args.report}")
    else:
        print(output)

    if cpd_exit_code == 4:
        print("\n⚠️  Duplicated code detected!")
    elif cpd_exit_code == 0:
        print("\n✅ No duplicated code found.")
    else:
        print(f"\n❌ CPD failed with exit code {cpd_exit_code}", file=sys.stderr)

    return resolve_cli_exit_code(cpd_exit_code=cpd_exit_code, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())

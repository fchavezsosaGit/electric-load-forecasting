"""Bootstrap and verify a local Python environment from `pyproject.toml`.

This module is the repository's environment contract helper. It is used when a
developer needs to:

- resolve the dependency list from project metadata
- install the runtime or dev toolchain into the active interpreter
- run a lightweight smoke check that confirms critical imports and parquet support

It intentionally avoids project-specific logic beyond dependency discovery so it
can stay stable as the modeling code evolves.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
MIN_PYTHON = (3, 11)
CORE_SMOKE_IMPORTS: tuple[str, ...] = (
    "joblib",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "statsmodels",
    "threadpoolctl",
)
ACCELERATION_SMOKE_IMPORTS: tuple[str, ...] = ("xgboost",)
DEV_SMOKE_IMPORTS: tuple[str, ...] = (
    "pytest",
    "nbconvert",
    "nbformat",
    "jupyter",
    "ipykernel",
)
PARQUET_BACKENDS: tuple[str, ...] = ("pyarrow", "fastparquet")


def _load_pyproject() -> dict[str, object]:
    """Load raw project metadata from the repository's `pyproject.toml` file."""
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _unique(values: Iterable[str]) -> list[str]:
    """Preserve first-seen ordering while removing duplicate dependency strings."""
    return list(dict.fromkeys(values))


def assert_python_version() -> None:
    """Fail fast when the active interpreter is below the repository minimum."""
    if sys.version_info < MIN_PYTHON:
        version = ".".join(str(part) for part in MIN_PYTHON)
        raise RuntimeError(
            f"Python {version}+ is required. Current interpreter: {sys.executable} "
            f"({sys.version.split()[0]})."
        )


def dependency_specifiers(*, include_dev: bool) -> list[str]:
    """Return normalized dependency specifiers from `pyproject.toml`.

    When `include_dev` is true, the `project.optional-dependencies.dev` group is
    appended so notebook and test tooling are installed alongside the runtime set.
    """
    return dependency_specifiers_with_options(
        include_dev=include_dev,
        include_acceleration=False,
    )


def dependency_specifiers_with_options(
    *,
    include_dev: bool,
    include_acceleration: bool,
) -> list[str]:
    """Return normalized dependency specifiers from `pyproject.toml`."""
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise RuntimeError("Invalid pyproject.toml: [project] table missing or malformed.")

    dependencies = list(project.get("dependencies", []))
    if include_dev:
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise RuntimeError("Invalid pyproject.toml: optional-dependencies table malformed.")
        dependencies.extend(optional.get("dev", []))
        if include_acceleration:
            dependencies.extend(optional.get("acceleration", []))
    elif include_acceleration:
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise RuntimeError("Invalid pyproject.toml: optional-dependencies table malformed.")
        dependencies.extend(optional.get("acceleration", []))

    normalized = _unique([str(dep) for dep in dependencies if str(dep).strip()])
    if not normalized:
        raise RuntimeError("No dependency specifiers found in pyproject.toml.")
    return normalized


def smoke_imports(*, include_dev: bool) -> tuple[str, ...]:
    """Return the import names used by the post-install smoke check."""
    return smoke_imports_with_options(
        include_dev=include_dev,
        include_acceleration=False,
    )


def smoke_imports_with_options(
    *,
    include_dev: bool,
    include_acceleration: bool,
) -> tuple[str, ...]:
    """Return the import names used by the post-install smoke check."""
    imports = list(CORE_SMOKE_IMPORTS)
    if include_acceleration:
        imports.extend(ACCELERATION_SMOKE_IMPORTS)
    if include_dev:
        imports.extend(DEV_SMOKE_IMPORTS)
    return tuple(_unique(imports))


def install_dependencies(*, include_dev: bool, include_acceleration: bool) -> None:
    """Install resolved dependencies into the active interpreter environment."""
    dependencies = dependency_specifiers_with_options(
        include_dev=include_dev,
        include_acceleration=include_acceleration,
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies], cwd=PROJECT_ROOT)


def smoke_check(*, include_dev: bool, include_acceleration: bool) -> None:
    """Verify that critical imports and at least one parquet backend are available."""
    missing = [
        name
        for name in smoke_imports_with_options(
            include_dev=include_dev,
            include_acceleration=include_acceleration,
        )
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(f"Missing dependencies after install: {missing}")
    if not any(importlib.util.find_spec(name) is not None for name in PARQUET_BACKENDS):
        raise RuntimeError("Missing parquet backend. Install either pyarrow or fastparquet.")


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for environment bootstrap and smoke validation."""
    parser = argparse.ArgumentParser(description="Install and verify dependencies from pyproject.toml.")
    parser.add_argument("--install", action="store_true", help="Install dependencies into the current interpreter environment.")
    parser.add_argument("--with-dev", action="store_true", help="Include development dependencies.")
    parser.add_argument(
        "--with-acceleration",
        action="store_true",
        help="Include optional accelerated modeling dependencies when available.",
    )
    parser.add_argument("--check", action="store_true", help="Run a smoke import check after install or against the current environment.")
    parser.add_argument("--print-deps", action="store_true", help="Print resolved dependency specifiers and exit.")
    return parser.parse_args()


def main() -> int:
    """Execute the requested bootstrap action for the active interpreter."""
    args = parse_args()
    assert_python_version()

    if args.print_deps:
        for dependency in dependency_specifiers_with_options(
            include_dev=args.with_dev,
            include_acceleration=args.with_acceleration,
        ):
            print(dependency)
        return 0

    if not any((args.install, args.check)):
        raise RuntimeError("No action requested. Use --install, --check, or --print-deps.")

    if args.install:
        install_dependencies(
            include_dev=args.with_dev,
            include_acceleration=args.with_acceleration,
        )
    if args.check:
        smoke_check(
            include_dev=args.with_dev,
            include_acceleration=args.with_acceleration,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

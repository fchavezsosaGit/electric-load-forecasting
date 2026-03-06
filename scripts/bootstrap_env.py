"""Bootstrap and verify a local Python environment from pyproject metadata."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
MIN_PYTHON = (3, 11)
CORE_SMOKE_IMPORTS = (
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "statsmodels",
)
DEV_SMOKE_IMPORTS = (
    "pytest",
    "nbconvert",
    "nbformat",
    "jupyter",
    "ipykernel",
)
PARQUET_BACKENDS = ("pyarrow", "fastparquet")


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def assert_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        version = ".".join(str(part) for part in MIN_PYTHON)
        raise RuntimeError(
            f"Python {version}+ is required. Current interpreter: {sys.executable} "
            f"({sys.version.split()[0]})."
        )


def dependency_specifiers(*, include_dev: bool) -> list[str]:
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

    normalized = _unique([str(dep) for dep in dependencies if str(dep).strip()])
    if not normalized:
        raise RuntimeError("No dependency specifiers found in pyproject.toml.")
    return normalized


def smoke_imports(*, include_dev: bool) -> tuple[str, ...]:
    imports = list(CORE_SMOKE_IMPORTS)
    if include_dev:
        imports.extend(DEV_SMOKE_IMPORTS)
    return tuple(_unique(imports))


def install_dependencies(*, include_dev: bool) -> None:
    dependencies = dependency_specifiers(include_dev=include_dev)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies], cwd=PROJECT_ROOT)


def smoke_check(*, include_dev: bool) -> None:
    missing = [name for name in smoke_imports(include_dev=include_dev) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing dependencies after install: {missing}")
    if not any(importlib.util.find_spec(name) is not None for name in PARQUET_BACKENDS):
        raise RuntimeError("Missing parquet backend. Install either pyarrow or fastparquet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and verify dependencies from pyproject.toml.")
    parser.add_argument("--install", action="store_true", help="Install dependencies into the current interpreter environment.")
    parser.add_argument("--with-dev", action="store_true", help="Include development dependencies.")
    parser.add_argument("--check", action="store_true", help="Run a smoke import check after install or against the current environment.")
    parser.add_argument("--print-deps", action="store_true", help="Print resolved dependency specifiers and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_python_version()

    if args.print_deps:
        for dependency in dependency_specifiers(include_dev=args.with_dev):
            print(dependency)
        return 0

    if not any((args.install, args.check)):
        raise RuntimeError("No action requested. Use --install, --check, or --print-deps.")

    if args.install:
        install_dependencies(include_dev=args.with_dev)
    if args.check:
        smoke_check(include_dev=args.with_dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

# Purpose: bootstrap a local Python environment for this repository.
# Source of truth for dependencies: pyproject.toml (`project.dependencies` + `project.optional-dependencies.dev`).
# Last reviewed: 2026-03-04
#
# Bootstrap a local development environment for this repository.
# Usage:
#   ./setup.sh
#   VENV_DIR=.venv ./setup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: '$PYTHON_BIN' was not found. Set PYTHON_BIN to a valid Python 3.11+ executable."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -c "import pathlib, subprocess, sys, tomllib; cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); project = cfg.get('project', {}); deps = list(project.get('dependencies', [])); deps.extend(project.get('optional-dependencies', {}).get('dev', [])); assert deps, 'No dependencies declared in pyproject.toml'; subprocess.check_call([sys.executable, '-m', 'pip', 'install', *deps])"
python -c "import importlib.util; modules=['numpy','pandas','matplotlib','seaborn','sklearn','statsmodels','jupyter','fastparquet']; missing=[m for m in modules if importlib.util.find_spec(m) is None]; assert not missing, f'Missing dependencies after install: {missing}'; assert importlib.util.find_spec('pyarrow') or importlib.util.find_spec('fastparquet'), 'Missing parquet backend (pyarrow or fastparquet)'"

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"

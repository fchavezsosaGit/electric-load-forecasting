#!/usr/bin/env bash
set -euo pipefail

# Purpose: bootstrap a local Python environment for this repository.
# Source of truth for dependencies: pyproject.toml via scripts/bootstrap_env.py.
# Last reviewed: 2026-03-06
#
# Usage:
#   ./setup.sh
#   ./setup.sh --venv-dir .venv311
#   ./setup.sh --no-venv
#   PYTHON_BIN=python3.12 ./setup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
NO_VENV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --no-venv)
            NO_VENV=1
            shift
            ;;
        *)
            echo "Error: unknown argument '$1'"
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: '$PYTHON_BIN' was not found. Set PYTHON_BIN to a valid Python 3.11+ executable."
    exit 1
fi

if [[ "$NO_VENV" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/bootstrap_env.py --install --with-dev --check
    echo "Setup complete (no virtual environment)."
    exit 0
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python scripts/bootstrap_env.py --install --with-dev --check

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"

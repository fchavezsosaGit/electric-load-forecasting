#!/usr/bin/env bash
set -euo pipefail

# Purpose: bootstrap a local Python environment for this repository.
# Source of truth for dependencies: pyproject.toml (`.[dev]` extras).
# Last reviewed: 2026-02-20
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
python -m pip install -e ".[dev]"

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"

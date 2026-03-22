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

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
NO_VENV=0
WITH_ACCELERATION=0

resolve_python_bin() {
    if [[ -n "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi

    local candidates=("python3.13" "python3.12" "python3")
    local candidate
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

test_venv_healthy() {
    local venv_python="$1"
    [[ -x "$venv_python" ]] || return 1
    "$venv_python" -c "import sys; print(sys.executable)" >/dev/null 2>&1
}

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
        --with-acceleration)
            WITH_ACCELERATION=1
            shift
            ;;
        *)
            echo "Error: unknown argument '$1'"
            exit 1
            ;;
    esac
done

PYTHON_BIN="$(resolve_python_bin)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: '$PYTHON_BIN' was not found. Set PYTHON_BIN to a valid Python 3.12/3.13 executable."
    exit 1
fi

BOOTSTRAP_ARGS=(--install --with-dev --check)
if [[ "$WITH_ACCELERATION" -eq 1 ]]; then
    BOOTSTRAP_ARGS+=(--with-acceleration)
fi

if [[ "$NO_VENV" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/bootstrap_env.py "${BOOTSTRAP_ARGS[@]}"
    echo "Setup complete (no virtual environment)."
    exit 0
fi

if [[ -d "$VENV_DIR" ]] && ! test_venv_healthy "$VENV_DIR/bin/python"; then
    echo "Existing virtual environment is not runnable on this machine. Recreating $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python scripts/bootstrap_env.py "${BOOTSTRAP_ARGS[@]}"

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"

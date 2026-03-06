#!/usr/bin/env bash
set -euo pipefail

# Purpose: run the repository E2E workflow from repo root.
# Usage:
#   ./run_e2e.sh --mode full
#   PYTHON_BIN=python3.12 ./run_e2e.sh --mode quick

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: '$PYTHON_BIN' was not found. Set PYTHON_BIN to a valid Python 3.11+ executable."
    exit 1
fi

exec "$PYTHON_BIN" scripts/run_e2e.py "$@"

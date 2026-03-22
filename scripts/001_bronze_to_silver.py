"""Compatibility entrypoint for stage-1 bronze-to-silver transformation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "stages/bronze_to_silver.py")


if __name__ == "__main__":
    bronze_to_silver()

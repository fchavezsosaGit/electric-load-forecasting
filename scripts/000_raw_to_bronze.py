"""Compatibility entrypoint for stage-0 raw-to-bronze ingestion."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "stages/raw_to_bronze.py")


if __name__ == "__main__":
    raw_to_bronze()

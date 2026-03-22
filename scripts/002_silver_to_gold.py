"""Compatibility entrypoint for stage-2 silver-to-gold transformation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "stages/silver_to_gold.py")


if __name__ == "__main__":
    silver_to_gold()

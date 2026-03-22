"""Compatibility entrypoint for stage-3 model dataset creation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "stages/create_model_datasets.py")


if __name__ == "__main__":
    create_model_datasets()
